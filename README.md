# Compliance Agent

A venture-capital diligence tool that cross-references claims about a startup
against its SEC filings using **dense vector retrieval**, then uses Claude to
flag contradictory **forward-looking statements** (projections, expectations,
plans) and logs discrepancies with citations back to the source regulatory
filing.

## How it works

```
claims ──► [Dense Retriever: sentence-transformers + cosine]
                         │
                         ▼
              top-k SEC filing passages
                         │
                         ▼
         [Claude Opus 4.6: forward-looking detection
           + contradiction analysis, structured output]
                         │
                         ▼
           logs/discrepancies_<CIK>_<timestamp>.json
```

- **SEC ingestion** — `sec.py` pulls filings from EDGAR (10-K, 10-Q, S-1, 8-K
  by default; configurable for Form D, S-3, etc.), strips HTML, and chunks
  with overlap.
- **Dense retrieval** — `retriever.py` embeds every chunk with
  `all-MiniLM-L6-v2` and runs cosine similarity in NumPy. Each passage keeps a
  pointer back to its source filing (accession number, form, date, URL).
- **Claim analysis** — `analyzer.py` sends the claim + top-k retrieved
  passages to Claude Opus 4.6 with adaptive thinking. Returns a structured
  `ClaimAssessment` (Pydantic) with verdict, forward-looking flag, severity,
  explanation, and cited passage numbers.
- **Discrepancy log** — contradictory forward-looking claims are flagged and
  written to `logs/` alongside the evidence trail (filing URL + excerpt).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
```

First run downloads the embedding model (~90 MB) to your HuggingFace cache.

## Usage

```bash
# Single claim against a public company
python agent.py --company AAPL \
  --claim "We expect Services revenue to grow 40% YoY through 2027"

# Batch of claims from a JSON file
python agent.py --company AAPL --claims examples/sample_claims.json

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
