# Cali_EU Branch — Execution Plan
## EU SFDR/CSRD Agent + California SB 54 Agent + Regulatory Knowledge Base

**Branch:** `Cali_EU`  
**Produced:** 2026-04-21  
**Scope:** Two new compliance agents, a shared regulatory knowledge base with daily delta-checking, per-user dashboard notifications, and architecture cleanup.

---

## LangChain Assessment

**Recommendation: Do not adopt LangChain.**

The existing stack is clean, purpose-built, and doing everything LangChain would provide — direct Anthropic SDK streaming, sentence-transformers embeddings, NumPy cosine retrieval, Pydantic schemas. Introducing LangChain would:

- Add 20+ transitive dependencies with a history of breaking API changes
- Require rewriting tested, working code to fit LangChain's abstractions
- Make debugging harder (chains obscure what prompt is actually being sent)
- Provide no capability the current codebase lacks

The one genuine gap is **task scheduling**. The right tool for that is `APScheduler` — a single-purpose, well-maintained library that embeds directly in the FastAPI process. It is the only new library this plan introduces.

The new EU and CA agents follow the exact same pattern as the existing analyzer: a new Python module, the same Pydantic verdict schema, the same streaming interface. No framework change needed.

---

## Architecture Overview

### How the New Agents Fit

```
User uploads deck
        ↓
[Extractor] — already parameterized by selected modules
  ├─ SEC modules selected  → extract financial/business claims (existing)
  ├─ EU module selected    → also extract ESG fields (Scope 1/2/3, board %, AI risk flag)
  └─ CA module selected    → also extract founder demographic fields
        ↓
Analysis Config Card — user selects jurisdictions (existing card, new checkboxes added)
        ↓
[/verify/stream] — dispatches to per-module analyzer
  ├─ sec_compliance     → analyzer.py        (existing — retrieves company EDGAR filings)
  ├─ eu_sfdr_csrd       → analyzer_sfdr.py  (new — checks against EU regulatory KB)
  └─ ca_sb54            → analyzer_sb54.py  (new — checks against CA regulatory KB)
        ↓
Combined SSE stream → single report with jurisdiction sections
```

### Regulatory Knowledge Base

Each agent has a knowledge base: a set of regulatory source documents (chunked, embedded) stored on disk. The SEC agent keeps its existing per-company EDGAR retrieval *unchanged* — that evidence is inherently company-specific and correct as-is. What is added is a cached layer of *regulatory standard* text (what the rule actually requires) that makes all three analyzers more precise.

```
regulatory_kb/
  manifest.json          ← tracks all sources: url, etag, last_modified, sha256, version_label
  sec/
    chunks.jsonl         ← text chunks from SEC Reg D/A/CF, S-1 requirements
    embeddings.npy       ← sentence-transformer embeddings matrix
  eu/
    chunks.jsonl
    embeddings.npy
  ca/
    chunks.jsonl
    embeddings.npy
```

**Delta-check strategy (most efficient):**
1. Send `HEAD` request with `If-None-Match: <stored_etag>` and `If-Modified-Since: <stored_date>`
2. `304 Not Modified` → skip, total cost: one HTTP header exchange
3. `200` with new ETag or Last-Modified → re-download, rechunk, re-embed, update manifest
4. Sources without HTTP cache headers (e.g., DFPI rulemaking page, some EUR-Lex pages) → SHA-256 the response body, compare to stored hash
5. Log every check result regardless of outcome (for audit trail)

**Rebuild on change:**
- Only the changed source's chunk file and embedding slice is rebuilt, not the full corpus
- Rebuilding one source typically takes < 10 seconds

---

## Epic 0 — Architecture Cleanup

*Clean up dead code and establish the standard interface all three analyzers must implement. Do this first so every subsequent epic builds on a consistent foundation.*

---

### Story 0.1 — Remove `auth_old.py`

**Context:** `auth_old.py` is dead code. It is not imported anywhere. It contains an older authentication scheme that predates the current HMAC session + Google One Tap flow.

**Implementation:**
- Delete `/compliance-agent/auth_old.py`
- Run `grep -r "auth_old" .` to confirm no imports remain

**Success Criteria:**
- `auth_old.py` does not exist in the repository
- All tests pass
- `grep -r "auth_old" .` returns no results

---

### Story 0.2 — Resolve `chromadb` Dependency

**Context:** `chromadb>=0.4.18` is in `requirements.txt` but is never imported anywhere in the codebase. It adds ~400 MB of transitive dependencies to the Docker image for no benefit.

**Decision required before implementing:** Use ChromaDB for the persistent regulatory KB (replacing the NumPy `.npy` file approach), or remove it entirely and stick with the existing NumPy pattern.

*Recommendation: Remove it. The NumPy pattern is lighter, already tested, and sufficient for the corpus sizes involved (regulatory text is small — typically < 500 chunks per jurisdiction).*

**Implementation:**
- Remove `chromadb>=0.4.18` from `requirements.txt`
- Confirm no import exists: `grep -r "chromadb" .`

**Success Criteria:**
- `chromadb` is not in `requirements.txt`
- Docker image build succeeds
- `grep -r "chromadb" .` returns no results

---

### Story 0.3 — Define Standard Analyzer Protocol

**Context:** The existing `analyzer.py` exposes `analyze_claim()` and `analyze_industry_claim()`. The two new analyzer modules must implement the same interface so `agent.py` can dispatch to any of them without branching logic.

**Implementation:**  
Create `analyzer_protocol.py`:

```python
from typing import Protocol, runtime_checkable
from pydantic import BaseModel

class ClaimAssessment(BaseModel):
    """Shared verdict schema — all three analyzers return this."""
    verdict: str            # CONSISTENT | CONTRADICTS | UNSUPPORTED | INSUFFICIENT_EVIDENCE
    severity: str           # HIGH | MEDIUM | LOW | NONE
    forward_looking: bool
    explanation: str
    cited_passages: list[int]
    missing_information: str | None = None
    jurisdiction: str       # "sec" | "eu_sfdr_csrd" | "ca_sb54"
    # Jurisdiction-specific structured output (optional)
    red_flags: list[str] = []     # 🔴 blockers
    warnings: list[str] = []      # 🟡 needs clarification
    verified: list[str] = []      # 🟢 confirmed present
    action_items: list[str] = []  # 📋 questions to ask founder

@runtime_checkable
class AnalyzerModule(Protocol):
    """Protocol every analyzer module must satisfy."""
    def analyze_claim(
        self,
        client,
        claim: str,
        hits: list,
        deck_context,
        model: str,
        **kwargs,
    ) -> ClaimAssessment: ...
```

Add `jurisdiction` field to the existing `ClaimAssessment` in `analyzer.py` (default `"sec"`) so it is backward compatible.

**Success Criteria:**
- `analyzer_protocol.py` exists and is importable
- `analyzer.py` satisfies `AnalyzerModule` protocol (verified with `isinstance` check in a unit test)
- No existing behavior changes

---

### Story 0.4 — Extract Shared Whitelist Check Utility

**Context:** The whitelist check logic (DB query + env-var lookup + open-if-no-restriction) is duplicated between `POST /auth/google/one-tap` and the old Google OAuth callback. A third copy would be needed for any future auth provider.

**Implementation:**
- Extract to `def _is_email_allowed(email: str, db: Session) -> bool` in `web.py`
- Replace both existing call sites with the single function
- Add unit test covering: allowed by env var, allowed by DB domain, allowed by DB email, denied, open (no restrictions)

**Success Criteria:**
- Whitelist check code appears exactly once in `web.py`
- All four whitelist scenarios covered by tests
- Login flow unchanged (no regression)

---

### Story 0.5 — Add `APScheduler` to Dependencies

**Context:** The daily regulatory update job needs a scheduler embedded in the FastAPI process. APScheduler is the standard lightweight choice for this.

**Implementation:**
- Add `apscheduler>=3.10.0` to `requirements.txt`
- Create `scheduler.py` with a stub `start_scheduler()` function (to be filled in Epic 1)
- Call `start_scheduler()` in `web.py` at startup (after `init_db()`)
- Confirm import works in Docker build

**Success Criteria:**
- `apscheduler` installs without conflict with existing packages
- `start_scheduler()` can be called at FastAPI startup with no error
- `scheduler.py` is importable

---

## Epic 1 — Regulatory Knowledge Base & Update Pipeline

*Build the shared infrastructure that all three agents (SEC, EU, CA) will use for their regulatory standard text. This epic must be complete before Epics 2 and 3.*

---

### Story 1.1 — `RegulationSource` and `Notification` DB Models

**Context:** The knowledge base manifest needs to be durable across container restarts. Two new SQLAlchemy models handle this.

**Implementation:**  
Add to `models.py`:

```python
class RegulationSource(Base):
    __tablename__ = "regulation_sources"
    id           = Column(Integer, primary_key=True)
    module       = Column(String(32), nullable=False)   # "sec" | "eu_sfdr_csrd" | "ca_sb54"
    name         = Column(String(256), nullable=False)  # human label, e.g. "SFDR Level 1"
    url          = Column(String(1024), nullable=False)
    etag         = Column(String(256), nullable=True)
    last_modified = Column(String(256), nullable=True)
    content_sha256 = Column(String(64), nullable=True)
    version_label  = Column(String(128), nullable=True) # e.g. "ELI:32021R2088"
    last_fetched   = Column(DateTime(timezone=True), nullable=True)
    last_changed   = Column(DateTime(timezone=True), nullable=True)
    chunk_count    = Column(Integer, default=0)
    created_at     = Column(DateTime(timezone=True), default=utc_now)

class Notification(Base):
    __tablename__ = "notifications"
    id           = Column(Integer, primary_key=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    module       = Column(String(32), nullable=False)
    source_name  = Column(String(256), nullable=False)
    title        = Column(String(512), nullable=False)
    body         = Column(Text, nullable=False)
    created_at   = Column(DateTime(timezone=True), default=utc_now)
    dismissed_at = Column(DateTime(timezone=True), nullable=True)
```

Run `init_db()` to create both tables. Add to `migrate_schema()` for safe column additions in future.

**Success Criteria:**
- Both tables created on fresh DB init
- `migrate_schema()` runs without error on an existing DB
- Models importable from `models.py`

---

### Story 1.2 — Knowledge Base Ingestion Pipeline (`regulatory_kb.py`)

**Context:** Core module responsible for fetching regulation text, chunking it, building embeddings, and storing everything to disk. All three agents call into this.

**Implementation:**  
Create `regulatory_kb.py`:

```
Key functions:
  load_sources() -> list[RegulationSource]   — reads from DB
  fetch_if_changed(source) -> (bool, str)    — HEAD → 304 skip; 200 → download
  ingest_source(source, raw_text)            — chunk → embed → write to disk
  build_retriever(module) -> DenseRetriever  — load chunks.jsonl + embeddings.npy
  get_retriever(module) -> DenseRetriever    — in-memory cache, rebuilt on update
```

Delta-check logic:
```python
def fetch_if_changed(source: RegulationSource) -> tuple[bool, str | None]:
    headers = {}
    if source.etag:
        headers["If-None-Match"] = source.etag
    if source.last_modified:
        headers["If-Modified-Since"] = source.last_modified
    resp = httpx.head(source.url, headers=headers, timeout=15, follow_redirects=True)
    if resp.status_code == 304:
        return False, None          # unchanged — zero cost
    # Changed or no cache headers: do full GET
    resp = httpx.get(source.url, timeout=60, follow_redirects=True)
    new_hash = hashlib.sha256(resp.content).hexdigest()
    if new_hash == source.content_sha256:
        return False, None          # content identical despite header miss
    return True, resp.text
```

Chunking: reuse `sec.chunk_text()` (1200 chars, 200 overlap). Embedding: reuse `DenseRetriever.add()` + `build()` from `retriever.py`.

Disk layout:
```
{DATA_ROOT or BASE}/regulatory_kb/{module}/chunks.jsonl
{DATA_ROOT or BASE}/regulatory_kb/{module}/embeddings.npy
```

**Success Criteria:**
- `fetch_if_changed()` returns `(False, None)` for a 304 response
- `fetch_if_changed()` returns `(False, None)` when content hash matches even on 200
- `fetch_if_changed()` returns `(True, text)` when content genuinely changed
- Chunks written to `chunks.jsonl`; embeddings to `embeddings.npy`
- `get_retriever("eu")` returns a `DenseRetriever` with correct chunk count
- In-memory retriever cache is invalidated and rebuilt after an ingestion

---

### Story 1.3 — Notification Fan-Out

**Context:** When a regulation update is detected, every active user gets a `Notification` row so they can see the change in their dashboard.

**Implementation:**  
In `regulatory_kb.py`, after a successful ingestion:

```python
def _notify_all_users(db: Session, module: str, source: RegulationSource, old_version: str):
    users = db.query(User).all()
    for user in users:
        db.add(Notification(
            user_id=user.id,
            module=module,
            source_name=source.name,
            title=f"Regulatory update: {source.name}",
            body=(
                f"{source.name} was updated on {utc_now().date()}. "
                f"Previous version: {old_version or 'unknown'}. "
                f"Reports run before this date may need re-review under the new text."
            ),
        ))
    db.commit()
```

**Success Criteria:**
- One `Notification` row per user created in DB after a successful ingestion
- Notification `body` includes source name, date of change, and guidance to re-review
- No notification created when delta-check finds no change

---

### Story 1.4 — Notification API Endpoints

**Context:** Frontend needs to fetch and dismiss notifications.

**Implementation:**  
Add to `web.py`:

```python
GET  /notifications        → list undismissed for current user (requires auth)
POST /notifications/{id}/dismiss → set dismissed_at = utcnow()
```

Response shape:
```json
[{
  "id": 1,
  "module": "eu_sfdr_csrd",
  "source_name": "SFDR Level 1 (EU 2019/2088)",
  "title": "Regulatory update: SFDR Level 1",
  "body": "...",
  "created_at": "2026-04-22T07:00:00Z"
}]
```

**Success Criteria:**
- `GET /notifications` requires valid session; returns 401 without one
- Returns only notifications for the authenticated user
- Returns only undismissed notifications (dismissed_at IS NULL)
- `POST /notifications/{id}/dismiss` sets `dismissed_at`; subsequent GET omits it
- Dismissing another user's notification returns 403

---

### Story 1.5 — Daily Scheduler (`scheduler.py`)

**Context:** APScheduler job that fires once a day (configurable time, default 07:00 UTC) and runs delta-checks for all regulation sources.

**Implementation:**

```python
# scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

def run_regulatory_update():
    """Called by APScheduler daily. Checks all sources; ingests any that changed."""
    db = next(get_db())
    try:
        for source in load_sources(db):
            changed, raw_text = fetch_if_changed(source)
            if changed:
                old_version = source.version_label
                ingest_source(source, raw_text, db)
                _notify_all_users(db, source.module, source, old_version)
                logger.info(f"[scheduler] Updated {source.name}")
            else:
                source.last_fetched = utc_now()
                db.commit()
                logger.debug(f"[scheduler] No change: {source.name}")
    finally:
        db.close()

def start_scheduler():
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_regulatory_update,
        CronTrigger(hour=7, minute=0),   # 07:00 UTC daily
        id="regulatory_update",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    return scheduler
```

`REGULATORY_CHECK_HOUR` env var overrides the default 07:00 UTC.

**Success Criteria:**
- Scheduler starts without error at FastAPI startup
- Job fires at 07:00 UTC (verified by setting `REGULATORY_CHECK_HOUR=test` and advancing system time in test)
- `misfire_grace_time=3600` ensures missed jobs (e.g., container restart) run within 1 hour
- If a source fetch raises an exception, the job logs the error and continues to the next source (no full crash)
- Scheduler shuts down cleanly on FastAPI shutdown event

---

### Story 1.6 — Dashboard Notification Banner (UI)

**Context:** Users need to see regulatory updates in the dashboard without navigating to a separate page.

**Implementation:**  
Add above the existing dashboard content in `index.html`:

```html
<div id="notification-banner" style="display:none">
  <!-- Populated by JS on page load -->
</div>
```

JS on `DOMContentLoaded`:
```javascript
async function loadNotifications() {
  const resp = await fetch('/notifications', { credentials: 'same-origin' });
  if (!resp.ok) return;
  const notes = await resp.json();
  if (!notes.length) return;
  const banner = document.getElementById('notification-banner');
  banner.innerHTML = notes.map(n => `
    <div class="notification-chip" data-id="${n.id}">
      <span class="material-symbols-rounded">update</span>
      <span>${escapeHtml(n.title)}</span>
      <span class="note-body">${escapeHtml(n.body)}</span>
      <button onclick="dismissNote(${n.id})">Dismiss</button>
    </div>
  `).join('');
  banner.style.display = 'block';
}

async function dismissNote(id) {
  await fetch(`/notifications/${id}/dismiss`, { method: 'POST', credentials: 'same-origin' });
  document.querySelector(`.notification-chip[data-id="${id}"]`)?.remove();
}
```

Style: amber/yellow chip (matches 🟡 warning palette), dismissible per notification.

**Success Criteria:**
- Banner appears after login when undismissed notifications exist
- Banner does not appear when no notifications exist
- Each notification chip shows source name and body text
- "Dismiss" removes only that chip from the UI and marks it dismissed in DB
- Page refresh after dismissal does not show the dismissed notification
- Notification is visible on both direct app URL and saasless.ai iframe

---

## Epic 2 — EU SFDR/CSRD Agent

*Adds the European ESG compliance module. Requires Epic 0 and Epic 1 to be complete.*

---

### Story 2.1 — Ingest EU Regulatory Sources

**Context:** The EU knowledge base needs the key regulation texts. These are the documents the agent reasons against when a deck claims to be ESG-compliant.

**Sources to ingest (seed list — extend as needed):**

| Source | URL | Version identifier |
|--------|-----|-------------------|
| SFDR Level 1 (EU 2019/2088) | `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R2088` | ELI code |
| SFDR RTS Delegated Regulation (EU 2022/1288) | `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R1288` | ELI code |
| CSRD Directive (EU 2022/2464) | `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022L2464` | ELI code |
| EU AI Act (EU 2024/1689) — High-Risk Annex III | `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689` | ELI code |
| ESRS Set 1 (CSRD Technical Standards) | `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023R2772` | ELI code |

**Implementation:**
- Create `seed_regulatory_sources()` function called once at startup if `RegulationSource` table is empty
- Each source seeded as a `RegulationSource` row with the URL and module `"eu_sfdr_csrd"`
- Immediately trigger an initial ingestion (blocking, at startup) so the knowledge base is populated before the first user request

**Success Criteria:**
- All 5 sources appear in the `regulation_sources` table after first startup
- `regulatory_kb/eu/chunks.jsonl` exists and contains > 200 chunks
- `regulatory_kb/eu/embeddings.npy` exists and has matching row count
- `get_retriever("eu")` returns a working `DenseRetriever`
- A search for "Scope 1 emissions" returns relevant CSRD passages

---

### Story 2.2 — EU Extraction Schema Extensions

**Context:** The extractor must pull ESG-specific fields when the EU module is selected. The deck may or may not contain these; the analyzer flags absence as a red flag.

**Implementation:**  
Add optional fields to `DeckExtraction` in `extractor.py`:

```python
class EsgMetrics(BaseModel):
    scope_1_emissions: str | None = None      # e.g. "12,000 tCO2e (2025)"
    scope_2_emissions: str | None = None
    scope_3_emissions: str | None = None
    has_third_party_audit: bool | None = None  # True if claims verified externally
    audit_body: str | None = None              # e.g. "Bureau Veritas"
    board_diversity_pct: str | None = None     # e.g. "40% women on board"
    supply_chain_disclosure: str | None = None
    ai_risk_sector: bool | None = None         # True if AI used in Health/Finance/HR/Infra
    ai_transparency_statement: str | None = None
    sfdr_article_claim: str | None = None      # "Article 8" or "Article 9" if claimed

class DeckExtraction(BaseModel):
    # ... existing fields ...
    esg_metrics: EsgMetrics | None = None      # populated when EU module selected
```

Extractor prompt addendum (injected when `"eu_sfdr_csrd"` in `requested_metrics`):
```
Additionally extract ESG-specific data:
- Scope 1, 2, 3 GHG emissions figures with measurement year
- Whether a third-party sustainability audit is referenced (and by whom)
- Board diversity percentage if stated
- Whether the company uses AI in a regulated sector (Health, Finance, HR, Infrastructure)
- Any SFDR Article 8 or Article 9 self-classification the company claims
Extract only what is explicitly stated. Use null for missing fields.
```

**Success Criteria:**
- `DeckExtraction.esg_metrics` is populated when `eu_sfdr_csrd` module selected
- `esg_metrics` is `null` when EU module not selected
- A deck with "carbon neutral" but no audit body yields `has_third_party_audit: false` and `audit_body: null`
- A deck with "40% female board members" yields `board_diversity_pct: "40%"`
- Extractor does not invent values; missing fields are `null` not placeholder strings

---

### Story 2.3 — `analyzer_sfdr.py`

**Context:** The core EU compliance analyzer. Takes extracted claims + ESG fields, retrieves relevant passages from the EU regulatory KB, and returns structured verdicts with 🔴/🟡/🟢 categorization.

**Implementation:**  
Create `analyzer_sfdr.py` implementing `AnalyzerModule` protocol:

```python
SYSTEM_PROMPT_SFDR = """
You are a Venture Capital Compliance Auditor specializing in EU sustainable finance law.
You audit pitch deck claims against SFDR (EU 2019/2088), the SFDR Delegated Regulation
(EU 2022/1288), CSRD (EU 2022/2464), and the EU AI Act (EU 2024/1689).

Your task: given a claim from a startup pitch deck and the most relevant passages from
EU regulatory text, classify the claim as:
  CONSISTENT          — claim is supported by and aligns with regulatory requirements
  CONTRADICTS         — claim contradicts a specific regulatory requirement
  UNSUPPORTED         — claim makes a regulatory assertion but lacks required evidence
  INSUFFICIENT_EVIDENCE — regulatory relevance cannot be determined from available data

Additional classifications for ESG absence:
  CRITICAL_ABSENT     — required data (e.g. Scope 1/2/3 emissions) is entirely missing
  GREENWASHING_RISK   — qualitative ESG claim without quantifiable, audited data

Output structured JSON matching ClaimAssessment schema.
ZERO HALLUCINATION: if data is absent, state DATA_ABSENT. Do not infer.
Clarify that this is an AI audit, not formal legal advice.
"""

def analyze_claim(client, claim, hits, deck_context, model, esg_metrics=None, **kwargs) -> ClaimAssessment:
    # Build prompt with regulatory passages (same structure as analyzer.py)
    # Include esg_metrics fields as structured context
    # Return ClaimAssessment with jurisdiction="eu_sfdr_csrd"
    ...

def analyze_esg_completeness(client, esg_metrics, model) -> list[ClaimAssessment]:
    """
    Separate pass: check for required-but-absent ESG fields.
    Returns one ClaimAssessment per missing required field.
    For example: if scope_1_emissions is null → CRITICAL_ABSENT verdict.
    """
    ...
```

Required field checks (always run when EU module active, even if no ESG claims extracted):
- Scope 1 absent → 🔴 CRITICAL: "SFDR Art. 4 requires PAI disclosure including GHG emissions"
- No third-party audit + ESG claim present → 🟡 WARNING: "Greenwashing risk — qualitative claim without CSRD-compliant audit"
- AI in regulated sector + no transparency statement → 🔴 CRITICAL: "EU AI Act Annex III High-Risk classification requires transparency documentation"
- Article 8/9 claim without CSRD-compliant data → 🔴 CRITICAL: "SFDR self-classification without required PAI indicators"

**Success Criteria:**
- `analyze_claim()` returns a `ClaimAssessment` with `jurisdiction="eu_sfdr_csrd"`
- `analyze_esg_completeness()` returns CRITICAL_ABSENT for a deck with `scope_1_emissions=null`
- A deck claiming "carbon neutral" without audit body receives `GREENWASHING_RISK` verdict
- A deck with AI in healthcare and no transparency statement receives `CRITICAL` verdict
- Retrieved passages are from the EU regulatory KB (not SEC EDGAR)
- `red_flags`, `warnings`, and `verified` lists are populated in the returned assessment
- `action_items` list contains 1–3 specific questions to send to the founder

---

### Story 2.4 — Add EU Module to Analysis Config Card

**Context:** The existing card has Universal, Seed/Pre-Seed, Series A/B, and Late Stage groups. EU and CA modules are added as a new "Regulatory Compliance" group.

**Implementation:**  
Add to `index.html` inside the `analysis-config` grid:

```html
<div class="module-group">
  <div class="module-group-title">Regulatory Compliance</div>
  <label style="display:flex; align-items:center; gap:8px; margin-bottom: 4px;">
    <input type="checkbox" class="metric-cb regulatory" value="eu_sfdr_csrd" />
    EU SFDR / CSRD (ESG)
  </label>
  <label style="display:flex; align-items:center; gap:8px; margin-bottom: 4px;">
    <input type="checkbox" class="metric-cb regulatory" value="ca_sb54" />
    California SB 54 (Demographics)
  </label>
</div>
```

These checkboxes emit `eu_sfdr_csrd` and `ca_sb54` as module values to `/verify/stream`.

**Success Criteria:**
- EU SFDR/CSRD and California SB 54 checkboxes appear in the Analysis Configuration card
- Checking EU module sends `eu_sfdr_csrd` in the `modules` form field
- Unchecking both regulatory checkboxes and running analysis does not invoke either new analyzer
- Checkboxes are NOT auto-selected by stage (they are opt-in, not stage-gated)

---

### Story 2.5 — EU Module in `/verify/stream` Pipeline

**Context:** The stream endpoint needs to dispatch to `analyzer_sfdr.py` when `eu_sfdr_csrd` is in the selected modules list.

**Implementation:**  
In `agent.py`, extend `iter_compliance_report()`:

```python
if "eu_sfdr_csrd" in modules:
    eu_retriever = regulatory_kb.get_retriever("eu")
    # Analyze each claim against EU KB
    for claim in claims:
        hits = eu_retriever.search(claim.text, top_k=5)
        result = analyzer_sfdr.analyze_claim(client, claim.text, hits, deck_context, model)
        yield {"event": "claim_result", "data": {"entry": result, "jurisdiction": "eu_sfdr_csrd"}}
    # Completeness check (absence flags — not claim-driven)
    if deck_context.extraction.esg_metrics:
        for result in analyzer_sfdr.analyze_esg_completeness(client, deck_context.extraction.esg_metrics, model):
            yield {"event": "claim_result", "data": {"entry": result, "jurisdiction": "eu_sfdr_csrd"}}
```

Report structure: `results` array gains a `jurisdiction` key per entry. Frontend groups entries by `jurisdiction` and renders each group under a jurisdiction heading.

**Success Criteria:**
- Selecting EU module and running analysis produces EU-jurisdiction claim results in the SSE stream
- EU results appear alongside SEC results in the same stream (not a separate request)
- `analyze_esg_completeness()` fires even if the deck has zero ESG-related claims
- Final saved report JSON contains a `eu_sfdr_csrd` section with all verdicts
- EU module can be run without SEC module and vice versa

---

## Epic 3 — California SB 54 Agent

*Adds the California Fair Investment Practices Act compliance module. Mirrors Epic 2 structure.*

---

### Story 3.1 — Ingest California Regulatory Sources

**Sources to ingest:**

| Source | URL | Notes |
|--------|-----|-------|
| SB 54 — CA Fair Investment Practices Act | `https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202120220SB54` | Hash-check (no ETag) |
| SB 164 — VC firm annual reporting | `https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB164` | Hash-check |
| DFPI VCC Reporting Program page | `https://dfpi.ca.gov/regulated-industries/vcc-reporting-program/` | Hash-check; watches for new rulemaking/template publications |
| DFPI Draft Survey Template (if published) | DFPI rulemaking URL when available | Populated when DFPI finalizes format |

**Implementation:**
- Same `seed_regulatory_sources()` pattern as EU
- Module name: `"ca_sb54"`
- DFPI pages lack ETag/Last-Modified → use SHA-256 hash comparison
- Initial ingestion at startup

**Success Criteria:**
- All CA sources appear in `regulation_sources` table
- `regulatory_kb/ca/chunks.jsonl` contains > 50 chunks (CA statutes are shorter than EU directives)
- Search for "founder demographic" returns SB 54 relevant passages
- Hash-check correctly identifies when DFPI page content changes

---

### Story 3.2 — CA Extraction Schema Extensions

**Context:** SB 54 requires VC firms to *collect* founder demographic data. The extractor checks whether the deck contains it; its absence is the violation.

**Implementation:**  
Add optional fields to `DeckExtraction`:

```python
class FounderDemographics(BaseModel):
    """Self-reported demographic data per SB 54 requirements."""
    founders: list[str]                   # names of listed founders
    race_ethnicity_disclosed: bool | None = None
    gender_disclosed: bool | None = None
    lgbtq_disclosed: bool | None = None
    veteran_status_disclosed: bool | None = None
    disability_disclosed: bool | None = None
    self_reported_note: str | None = None  # any caveat the founders stated

class DeckExtraction(BaseModel):
    # ... existing fields ...
    founder_demographics: FounderDemographics | None = None  # populated when CA module selected
```

Extractor prompt addendum (injected when `"ca_sb54"` in `requested_metrics`):
```
Additionally extract California SB 54 demographic data:
- List all named founders
- For each, note whether any of these fields are disclosed: race/ethnicity,
  gender identity, LGBTQ+ status, veteran status, disability status
- Record any note that founders describe the data as "self-reported"
Do not infer demographics. Only record what is explicitly stated.
```

**Success Criteria:**
- `founder_demographics` populated when CA module selected
- A deck listing "Jane Smith, CEO (she/her, Latina, Veteran)" yields corresponding fields as `true`
- A deck with named founders but no demographic data yields all disclosure fields as `false`
- No demographic data invented for any field

---

### Story 3.3 — `analyzer_sb54.py`

**Context:** The CA compliance analyzer. Checks whether the deck satisfies SB 54 collection requirements.

**Implementation:**  
Create `analyzer_sb54.py`:

```python
SYSTEM_PROMPT_SB54 = """
You are a Venture Capital Compliance Auditor specializing in California law.
You audit pitch deck submissions against California SB 54 (Fair Investment Practices Act)
and SB 164 (VC annual reporting requirements).

Key rule: California-based VC firms are required to COLLECT (not publish) demographic
data for each founding team member: race/ethnicity, gender identity, LGBTQ+ status,
veteran status, and disability status.

As of April 2026, the DFPI has suspended enforcement pending final rulemaking, but
collection remains a proactive requirement. A deck that does not contain this data
is non-compliant for intake by a CA-based fund — flag it as CRITICAL.

ZERO HALLUCINATION. If data is absent, state DATA_ABSENT.
Output is an AI compliance audit, not formal legal advice.
"""

def analyze_demographic_completeness(client, demographics, model) -> list[ClaimAssessment]:
    """
    Check each required demographic field. Missing field → CRITICAL_ABSENT verdict.
    Returns one ClaimAssessment per required field category.
    """
    required_fields = [
        ("race_ethnicity_disclosed", "Race/Ethnicity", "SB 54 §2923(a)(1)"),
        ("gender_disclosed", "Gender Identity", "SB 54 §2923(a)(2)"),
        ("lgbtq_disclosed", "LGBTQ+ Status", "SB 54 §2923(a)(3)"),
        ("veteran_status_disclosed", "Veteran Status", "SB 54 §2923(a)(4)"),
        ("disability_disclosed", "Disability Status", "SB 54 §2923(a)(5)"),
    ]
    results = []
    for field, label, citation in required_fields:
        disclosed = getattr(demographics, field, None)
        if disclosed is False or disclosed is None:
            results.append(ClaimAssessment(
                verdict="CRITICAL_ABSENT",
                severity="HIGH",
                forward_looking=False,
                explanation=f"{label} data is absent. {citation} requires CA-based VC funds to collect this field before intake.",
                cited_passages=[],
                jurisdiction="ca_sb54",
                red_flags=[f"Missing {label} disclosure ({citation})"],
                action_items=[f"Request {label} self-identification from all listed founders before proceeding to term sheet."],
            ))
        else:
            results.append(ClaimAssessment(
                verdict="CONSISTENT",
                severity="NONE",
                forward_looking=False,
                explanation=f"{label} data is present in the deck.",
                cited_passages=[],
                jurisdiction="ca_sb54",
                verified=[f"{label} disclosed"],
            ))
    return results
```

**Note on enforcement status:** Every CA report must include a standard disclaimer: "As of April 2026, DFPI has suspended enforcement of SB 54 pending final rulemaking. Collection remains a proactive requirement. This status is checked daily and this report will notify you if enforcement resumes."

**Success Criteria:**
- A deck with all 5 demographic fields present returns 5 `CONSISTENT` assessments
- A deck missing race/ethnicity returns a `CRITICAL_ABSENT` assessment citing `SB 54 §2923(a)(1)`
- Every CA report includes the enforcement suspension disclaimer with the current date
- `action_items` contains specific, actionable questions per missing field
- No demographic data inferred or assumed

---

### Story 3.4 — CA Module in `/verify/stream` Pipeline

**Context:** Same dispatch pattern as Epic 2, Story 2.5, for the CA module.

**Implementation:**  
In `agent.py`:

```python
if "ca_sb54" in modules:
    ca_retriever = regulatory_kb.get_retriever("ca")
    # Demographic completeness check (primary SB 54 audit)
    if deck_context.extraction.founder_demographics:
        for result in analyzer_sb54.analyze_demographic_completeness(
            client, deck_context.extraction.founder_demographics, model
        ):
            yield {"event": "claim_result", "data": {"entry": result, "jurisdiction": "ca_sb54"}}
    else:
        # Deck didn't even list founders → maximum severity
        yield {"event": "claim_result", "data": {
            "entry": ClaimAssessment(
                verdict="CRITICAL_ABSENT",
                severity="HIGH",
                explanation="No founders identified in the deck. SB 54 demographic collection cannot be satisfied.",
                jurisdiction="ca_sb54",
                red_flags=["No founders listed"],
            ),
            "jurisdiction": "ca_sb54",
        }}
    # Also check any explicit CA-related claims against CA regulatory KB
    ca_claims = [c for c in claims if "california" in c.text.lower() or "sb 54" in c.text.lower()]
    for claim in ca_claims:
        hits = ca_retriever.search(claim.text, top_k=5)
        result = analyzer_sb54.analyze_claim(client, claim.text, hits, deck_context, model)
        yield {"event": "claim_result", "data": {"entry": result, "jurisdiction": "ca_sb54"}}
```

**Success Criteria:**
- CA module runs without SEC or EU modules selected
- A deck with no founder data emits a CRITICAL_ABSENT event in the stream
- A deck with partial demographic data emits a mix of CONSISTENT and CRITICAL_ABSENT events
- CA results are grouped under `ca_sb54` in the final report
- DFPI enforcement suspension disclaimer is present in the saved report JSON

---

## Epic 4 — SEC Knowledge Base Enhancement

*The SEC agent's core behavior (per-company EDGAR filing retrieval) is correct and stays unchanged. This epic adds a cached layer of SEC regulatory standard text so the analyzer has precise rule references — not just company filing passages.*

---

### Story 4.1 — Ingest SEC Regulatory Standard Sources

**Sources to ingest:**

| Source | URL | Notes |
|--------|-----|-------|
| Reg D — Rule 506 | `https://www.ecfr.gov/current/title-17/chapter-II/part-230/subject-group-ECFRc96992958e1cdca/section-230.506` | ETag |
| Regulation A (Tier 1/2) | `https://www.ecfr.gov/current/title-17/chapter-II/part-230/subject-group-ECFR5b5a7edfab7de5e` | ETag |
| Regulation CF | `https://www.ecfr.gov/current/title-17/chapter-II/part-227` | ETag |
| Form S-1 Instructions | `https://www.sec.gov/files/forms-1.pdf` | Hash |
| SEC Cybersecurity Disclosure Rules (2023) | `https://www.sec.gov/rules/final/2023/33-11216.pdf` | Hash |

**Success Criteria:**
- All 5 sources seeded and ingested at startup
- `get_retriever("sec")` returns a working retriever with regulatory text
- Retriever is used as supplemental context in `analyzer.py` (regulatory standards injected alongside EDGAR passages)
- Existing per-company EDGAR filing retrieval is unchanged

---

### Story 4.2 — Inject Regulatory Context into SEC Analyzer

**Context:** Currently `analyzer.py` only sees company-specific filing passages. Injecting the relevant SEC regulatory standard text makes the analyzer more precise (e.g., "under Reg D 506(b), accredited investor verification is required").

**Implementation:**
- In `analyzer.py`, after retrieving top-k EDGAR passages, also retrieve top-3 from the SEC regulatory KB
- Clearly label the two passage sources in the prompt: `[COMPANY FILINGS]` vs `[SEC REGULATIONS]`
- Do not change the verdict schema or output format

**Success Criteria:**
- SEC regulatory passages appear in the analyzer prompt labeled `[SEC REGULATIONS]`
- Company filing passages still appear labeled `[COMPANY FILINGS]`
- Verdicts reference regulatory citations when relevant (e.g., "Rule 506(b) requires...")
- No existing verdicts change in regression testing against previously saved reports

---

## Epic 5 — Testing & QA

---

### Story 5.1 — Unit Tests for Delta-Check Logic

**File:** `test_regulatory_kb.py`

**Tests:**
- `test_304_returns_no_change`: mock HTTP 304 → `fetch_if_changed` returns `(False, None)`
- `test_200_same_hash_returns_no_change`: mock 200 with same SHA-256 → `(False, None)`
- `test_200_new_hash_returns_changed`: mock 200 with different content → `(True, text)`
- `test_no_etag_uses_hash`: source with no stored ETag sends no conditional headers, falls back to hash
- `test_ingest_creates_chunks_and_embeddings`: after `ingest_source()`, files exist and row count matches

**Success Criteria:** All 5 tests pass. Coverage on `regulatory_kb.py` > 80%.

---

### Story 5.2 — Integration Tests for Each Analyzer

**File:** `test_analyzers.py`

**Tests (per analyzer):**
- `test_sfdr_greenwashing_flag`: deck with "eco-friendly" but no audit → `GREENWASHING_RISK`
- `test_sfdr_scope1_absent`: esg_metrics with `scope_1_emissions=null` → `CRITICAL_ABSENT`
- `test_sfdr_article9_without_data`: deck claiming Article 9 without PAI data → `CRITICAL`
- `test_sb54_missing_all_demographics`: all fields absent → 5 `CRITICAL_ABSENT` results
- `test_sb54_all_present`: all fields present → 5 `CONSISTENT` results
- `test_sb54_partial`: 3/5 present → 3 CONSISTENT + 2 CRITICAL_ABSENT
- `test_sec_regulatory_context_injected`: SEC analyzer prompt contains `[SEC REGULATIONS]` section

**Success Criteria:** All 7 tests pass. Tests run without live API calls (Anthropic client mocked).

---

### Story 5.3 — End-to-End Manual QA Checklist

**Procedure (run on staging Railway deployment):**

- [ ] Upload a deck with no ESG data; select EU module → report shows 🔴 for all ESG fields
- [ ] Upload a deck with "carbon neutral" claim; select EU module → report shows 🟡 greenwashing warning
- [ ] Upload a deck with Scope 1/2/3 + third-party audit; select EU module → report shows 🟢
- [ ] Upload a deck with no founder demographics; select CA module → report shows 🔴 for all 5 fields
- [ ] Upload a deck with full founder demographics; select CA module → report shows 🟢
- [ ] Run only SEC module → EU and CA sections absent from report
- [ ] Run all three modules → all three sections present in one combined report
- [ ] Trigger a manual regulatory update (set a test source URL to a modified page) → notification appears in dashboard
- [ ] Dismiss notification → does not reappear on refresh
- [ ] Verify CA enforcement disclaimer appears in every CA report

**Success Criteria:** All checklist items pass on Railway staging before merge to `main`.

---

## Environment Variables Added

| Variable | Purpose | Default |
|----------|---------|---------|
| `REGULATORY_CHECK_HOUR` | UTC hour for daily KB update job | `7` |
| `REGULATORY_KB_DIR` | Override directory for KB files | `{DATA_ROOT}/regulatory_kb` |

---

## New Files Created

| File | Purpose |
|------|---------|
| `analyzer_protocol.py` | Shared `ClaimAssessment` schema + `AnalyzerModule` protocol |
| `analyzer_sfdr.py` | EU SFDR/CSRD compliance analyzer |
| `analyzer_sb54.py` | California SB 54 compliance analyzer |
| `regulatory_kb.py` | Knowledge base ingestion, delta-check, retriever cache |
| `scheduler.py` | APScheduler daily update job |
| `test_regulatory_kb.py` | Unit tests for delta-check logic |
| `test_analyzers.py` | Integration tests for all three analyzers |

---

## Files Modified

| File | Change |
|------|--------|
| `models.py` | Add `RegulationSource`, `Notification` models |
| `db.py` | Import and create new models |
| `extractor.py` | Add `EsgMetrics`, `FounderDemographics` to `DeckExtraction` |
| `analyzer.py` | Add `jurisdiction` field; inject SEC regulatory KB context |
| `agent.py` | Dispatch to new analyzers by module name |
| `web.py` | Add notification endpoints; call `start_scheduler()`; remove `auth_old.py` import; use `_is_email_allowed()` |
| `requirements.txt` | Add `apscheduler>=3.10.0`; remove `chromadb` |
| `templates/index.html` | Add EU/CA checkboxes to Analysis Config card; add notification banner |

---

## Suggested Implementation Order

```
Week 1:  Epic 0 (cleanup)
Week 2:  Epic 1, Stories 1.1–1.3 (models, KB pipeline)
Week 3:  Epic 1, Stories 1.4–1.6 (scheduler, notifications UI)
Week 4:  Epic 2, Stories 2.1–2.3 (EU sources, extraction, analyzer)
Week 5:  Epic 2, Stories 2.4–2.5 + Epic 3, Stories 3.1–3.2 (EU UI; CA sources, extraction)
Week 6:  Epic 3, Stories 3.3–3.4 + Epic 4 (CA analyzer + SEC enhancement)
Week 7:  Epic 5 (testing, QA, merge to main)
```
