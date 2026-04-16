# VC Compliance Pipeline

Two cooperating agents for venture-capital diligence:

1. **Pitch Deck Extractor** — parses an uploaded PDF deck and returns
   structured, *verbatim* claims + company identity. Extracts only; never
   analyzes or fabricates. If a field is absent from the deck it is null, and
   the extractor records the gap in its transparency notes.
2. **Compliance Agent** — cross-references those claims against the
   company's SEC filings using **dense vector retrieval** and flags
   contradictory **forward-looking statements**. When evidence is missing, it
   returns `INSUFFICIENT_EVIDENCE` with an explicit `missing_information`
   field rather than guessing. The deck context is used only as *clarifying*
   metadata — SEC filings are the sole source of truth for verification.

A FastAPI web UI wires both agents together: upload deck → review extraction
→ run compliance check → read report.

## How it works

```
 PDF pitch deck
       │
       ▼
 [extractor.py]  verbatim extraction only; null when absent; no fabrication
       │
       ▼
 DeckExtraction JSON  ──► claims + company identity + transparency notes
       │
       ▼
 [agent.py]  company → SEC CIK → fetch 10-K/10-Q/S-1/8-K
       │
       ▼
 [retriever.py]  sentence-transformers + cosine over filing chunks
       │
       ▼
 [analyzer.py]  Claude Opus 4.6 w/ adaptive thinking
       │           verdict ∈ {CONTRADICTS, UNSUPPORTED, CONSISTENT, INSUFFICIENT_EVIDENCE}
       ▼
 logs/discrepancies_<CIK>_<timestamp>.json   +   web UI report
```

- **Pitch deck extraction** — `extractor.py` sends the PDF to Claude Opus 4.6
  via native PDF support, extracts a Pydantic `DeckExtraction` with verbatim
  claims, company identity, and explicit transparency notes about anything
  it could not extract. No analysis, no guessing.
- **SEC ingestion** — `sec.py` resolves tickers/names to CIK via EDGAR,
  pulls recent filings, strips HTML, and chunks with overlap.
- **Dense retrieval** — `retriever.py` embeds chunks with
  `all-MiniLM-L6-v2` and runs cosine similarity in NumPy. Each passage keeps
  a pointer back to its source filing (accession, form, date, URL).
- **Claim analysis** — `analyzer.py` sends the claim + deck context (clarifying
  only) + top-k retrieved passages to Claude Opus 4.6 with adaptive thinking.
  Returns a structured `ClaimAssessment` including `missing_information` when
  it cannot complete the assessment.
- **Discrepancy log** — contradictory forward-looking claims are flagged and
  written to `logs/` alongside the evidence trail. The report is always
  transparent about what it could and could not verify.

### Transparency & no-fabrication guarantees

- Extractor leaves unknown fields null; `extraction_notes` records every gap.
- Analyzer returns `INSUFFICIENT_EVIDENCE` when retrieved passages don't
  support a decision, with a `missing_information` field naming what was
  needed.
- Compliance agent returns `INSUFFICIENT_EVIDENCE` for every claim when no
  SEC CIK can be resolved, rather than silently producing an empty report.
- Deck context is labeled "clarifying metadata only, NOT a source of truth"
  in the analyzer prompt — the deck can't be used to verify itself.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
```

First run downloads the embedding model (~90 MB) to your HuggingFace cache.

## Usage

### Web UI (recommended — runs both agents)

```bash
uvicorn web:app --reload
# open http://localhost:8000
```

Upload a pitch deck → review the extracted claims and transparency notes →
click "Run compliance check" → read the report with citations to SEC filings.

### CLI

```bash
# Single claim against a public company
python agent.py --company AAPL \
  --claim "We expect Services revenue to grow 40% YoY through 2027"

# Batch of claims from a JSON file
python agent.py --company AAPL --claims examples/sample_claims.json

# From a previously extracted deck context (auto-derives company + claims)
python agent.py --deck deck_contexts/deck_abc123.json

# Using a CIK directly (private companies with Form D filings)
python agent.py --cik 0001318605 --forms D,S-1 --claims my_claims.json
```

### Flags

| Flag         | Default                 | Purpose                                       |
|--------------|-------------------------|-----------------------------------------------|
| `--company`  | —                       | Ticker or company name (resolves to CIK)      |
| `--cik`      | —                       | Direct 10-digit SEC CIK                       |
| `--claim`    | —                       | Single claim (string)                         |
| `--claims`   | —                       | Path to JSON file with `claims: [...]`        |
| `--forms`    | `10-K,10-Q,S-1,8-K`     | Comma-separated filing types to index         |
| `--filings`  | `3`                     | Max number of filings to pull                 |
| `--top-k`    | `5`                     | Top-k passages retrieved per claim            |

## Output

Exit code is non-zero if any contradictory forward-looking statements are
flagged — wire it into CI or a diligence workflow. The JSON log in `logs/`
contains every claim, verdict (`CONTRADICTS` / `UNSUPPORTED` / `CONSISTENT` /
`INSUFFICIENT_EVIDENCE`), severity, explanation, and cited passage excerpts
with filing URLs for audit.

## Notes on private startups

Most early-stage startups have limited SEC exposure — typically Form D
(private placement notices) and not much else. Point `--forms D` at the
relevant CIK, or skip SEC filings and adapt `sec.py` to ingest a data room
(the retrieval + analysis pipeline is agnostic to source).
