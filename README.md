# VC Compliance Pipeline

A multi-jurisdiction AI compliance platform for venture capital diligence. Three cooperating agents assess pitch deck claims against SEC filings, EU sustainable finance regulations (SFDR/CSRD), and California SB 54 founder-disclosure law — in parallel.

## Agents

### 1. Pitch Deck Extractor (`extractor.py`)
Parses an uploaded PDF and returns structured, *verbatim* claims, company identity, ESG metrics, and founder demographics. Extracts only — never analyzes or fabricates. Missing fields are `null`; every gap is recorded in `extraction_notes`.

### 2. SEC Compliance Agent (`analyzer.py`)
Resolves the company ticker/name to a CIK via EDGAR, pulls recent filings (10-K, 10-Q, S-1, 8-K — Form D available via `--forms D`), and uses dense vector retrieval to find the passages that verify or contradict each claim.

### 3. EU SFDR/CSRD Agent (`analyzer_sfdr.py`)
Checks claims against a self-hosted regulatory knowledge base built from five EU legal texts (SFDR, CSRD, EU AI Act, ESRS). Also runs an ESG completeness check — flags missing Scope 1/2/3 emissions, absent third-party audits, and undisclosed board diversity per SFDR Art. 4 and CSRD §5(1).

### 4. California SB 54 Agent (`analyzer_sb54.py`)
Audits founder demographic disclosures (gender, race/ethnicity, education, prior startup experience) against California SB 54 and SB 164 requirements. Returns CONSISTENT or CRITICAL_ABSENT for each required field.

### 5. Compliance Orchestrator (`orchestrator.py`)
Runs all three jurisdiction agents **concurrently** instead of sequentially, cutting average report time from ~13 minutes to ~3–4 minutes for a 10-claim deck. Includes a global rate-limit semaphore, `retry-after`-aware backoff, and Haiku routing for lightweight completeness checks.

---

## How it works

```
 PDF pitch deck
       │
       ▼
 [extractor.py]  verbatim extraction only; null when absent; no fabrication
       │
       ├── claims + company identity
       ├── esg_metrics (Scope 1/2/3, audit, board diversity, AI sector, SFDR claim)
       └── founder_demographics (gender, race/ethnicity, education, experience)
                 │
                 ▼
 [orchestrator.py]  fans out to all three jurisdictions in parallel
       │
       ├─── [analyzer.py]       SEC: CIK → filings → dense retrieval → verdict
       ├─── [analyzer_sfdr.py]  EU:  regulatory KB → SFDR/CSRD/AI Act → verdict
       │                            + ESG completeness check (Haiku)
       └─── [analyzer_sb54.py]  CA:  founder demographics → SB 54/164 → verdict
                 │
                 ▼
 [compliance_scorer.py]    per-jurisdiction score (0–100) + risk level
 [greenwashing_detector.py] pattern detection on ESG claims
 [regulatory_mapper.py]    maps violations to specific articles + remediations
 [compliance_trends.py]    velocity + trend direction from historical scores
 [predictive_risk.py]      30-day / 90-day risk forecast
                 │
                 ▼
 Final report (JSON)  ──►  web UI  /  logs/  /  saved_reports/
```

---

## Regulatory coverage

| Jurisdiction | Regulations | Source |
|---|---|---|
| **SEC** | Securities Act, Exchange Act, PSLRA | Live EDGAR filings (10-K, 10-Q, S-1, 8-K) |
| **EU SFDR/CSRD** | SFDR EU 2019/2088, CSRD EU 2022/2464, EU AI Act EU 2024/1689, ESRS EU 2023/2772 | 5 EU regulatory texts, seeded on first startup |
| **California** | SB 54, SB 164, DFEH Guidelines | 3 CA regulatory texts, seeded on first startup |

---

## Verdicts

Every claim receives one verdict per active jurisdiction:

| Verdict | Meaning |
|---|---|
| `CONSISTENT` | Filing / regulation explicitly corroborates the claim |
| `CONTRADICTS` | Filing / regulation materially conflicts with the claim |
| `UNSUPPORTED` | Claim is specific and checkable, but no evidence found |
| `INSUFFICIENT_EVIDENCE` | Not enough information to decide |
| `CRITICAL_ABSENT` | Required disclosure is entirely missing |
| `GREENWASHING_RISK` | Qualitative ESG claim without audited, quantified data |
| `DATA_QUALITY_ISSUE` | Data present but unreliable or unverifiable |

---

## Concurrency & rate limits

The orchestrator is tuned for **Anthropic Tier 2** (1k RPM / 80k ITPM / 16k OTPM):

| Mitigation | Detail |
|---|---|
| `claim_workers=2` per jurisdiction | Default; keeps sustained OTPM well within Tier 2 cap |
| `global_max_in_flight=6` semaphore | Hard cap across all concurrent jurisdictions combined |
| `max_tokens=4000` on all analyzers | `ClaimAssessment` fits in 4k; inflated ceiling was the primary OTPM drain |
| `retry-after` header-aware backoff | Sleeps exactly as long as Anthropic requests on 429, falls back to exponential (1s→2s→4s, max 30s) |
| Haiku for completeness checks | ESG and CA SB 54 field-presence checks use `claude-haiku-4-5` — separate rate pool, 20× cheaper than Sonnet |

All five values are tunable constructor parameters on `ComplianceOrchestrator`.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
```

First run downloads the embedding model (~90 MB) to your HuggingFace cache and seeds the EU and CA regulatory knowledge bases automatically.

---

## Usage

### Web UI (recommended)

```bash
uvicorn web:app --reload
# open http://localhost:8000
```

Upload a pitch deck → review extracted claims, ESG metrics, and founder demographics → select modules (SEC / EU / CA) → run compliance check → read the report with citations to filings and regulatory articles.

### CLI

```bash
# Single claim against a public company (SEC only)
python agent.py --company AAPL \
  --claim "We expect Services revenue to grow 40% YoY through 2027"

# Full deck context across all three jurisdictions
python agent.py --deck deck_contexts/deck_abc123.json \
  --modules sec,eu_sfdr_csrd,ca_sb54

# Private company with Form D filings
python agent.py --cik 0001318605 --forms D,S-1 --claims my_claims.json
```

### Programmatic (orchestrator)

```python
from orchestrator import ComplianceOrchestrator
from deck_context import DeckContext

orch = ComplianceOrchestrator(
    deck=DeckContext.load("deck_contexts/deck_abc123.json"),
    modules=["sec", "eu_sfdr_csrd", "ca_sb54"],
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

# Blocking — returns full report dict
report = orch.run()

# Streaming — events arrive as each claim finishes
for event in orch.stream():
    if event["event"] == "claim_result":
        print(event["data"]["entry"]["verdict"])
```

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--company` | — | Ticker or company name (resolves to CIK via EDGAR) |
| `--cik` | — | Direct 10-digit SEC CIK |
| `--claim` | — | Single claim string |
| `--claims` | — | Path to JSON file with `claims: [...]` |
| `--forms` | `10-K,10-Q,S-1,8-K` | Comma-separated filing types to index |
| `--filings` | `3` | Max filings to pull per form type |
| `--top-k` | `5` | Top-k passages retrieved per claim |
| `--modules` | `sec` | Comma-separated: `sec`, `eu_sfdr_csrd`, `ca_sb54` |

---

## Output

The JSON report includes:

- **Per-claim results** — verdict, severity, `forward_looking` flag, explanation, cited passage excerpts with filing URLs, action items for founders
- **Compliance scores** — 0–100 score and risk level (NONE / LOW / MEDIUM / HIGH / CRITICAL) per jurisdiction
- **Greenwashing assessment** — detected patterns, missing evidence, risk score
- **Regulatory map** — violations mapped to specific articles (SFDR Art. 4, CSRD §5(1), SB 54, etc.) with required remediations
- **Trend analysis** — velocity and direction from historical scores
- **Predictive risk** — 30-day and 90-day risk forecast with early warnings
- **Timing** — per-jurisdiction wall-clock and `under_10_minute_budget` flag

Exit code is non-zero if any contradictory forward-looking statements are flagged — suitable for CI or automated diligence workflows.

---

## Transparency & no-fabrication guarantees

- Extractor leaves unknown fields `null`; `extraction_notes` records every gap.
- Analyzer returns `INSUFFICIENT_EVIDENCE` when retrieved passages don't support a decision, with a `missing_information` field naming exactly what was needed.
- Compliance agent returns `INSUFFICIENT_EVIDENCE` for every claim when no SEC CIK can be resolved, rather than silently producing an empty report.
- Deck context is labeled *"clarifying metadata only, NOT a source of truth"* in every analyzer prompt — the deck cannot be used to verify itself.
- EU and CA agents return `CRITICAL_ABSENT` (not a fabricated response) when required regulatory disclosures are entirely missing.

---

## Regulatory knowledge base

EU and CA regulatory texts are fetched, chunked, and embedded automatically on first startup. The background scheduler (`scheduler.py`) checks for updates daily. Retrieval uses `sentence-transformers/all-MiniLM-L6-v2` with cosine similarity over 1,200-character overlapping chunks.

**EU sources (seeded at startup):**
- SFDR Level 1 (EU 2019/2088)
- SFDR RTS Delegated Regulation (EU 2022/1288)
- CSRD Directive (EU 2022/2464)
- EU AI Act High-Risk Annex III (EU 2024/1689)
- ESRS Set 1 — CSRD Technical Standards (EU 2023/2772)

**California sources (seeded at startup):**
- California SB 54 — Nonprofit Integrity Act
- California SB 164 — Board Diversity Requirements
- DFEH Fair Employment Guidelines

---

## Notes on private startups

Most early-stage startups have limited SEC exposure. For companies without a CIK, the SEC agent falls back to web search (`analyze_industry_claim`) to verify market-size and industry claims using live authoritative sources. For companies with a Form D on file, add `--forms D` to include private placement disclosures in the retrieval index.
