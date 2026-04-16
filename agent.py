"""Compliance Agent — entry point.

Cross-references claims about a startup against its SEC filings using dense
vector retrieval. Flags contradictory forward-looking statements and writes a
discrepancy log to ./logs/.

Usage:
    python agent.py --company AAPL --claims examples/sample_claims.json
    python agent.py --cik 0000320193 --claim "We expect 50% YoY revenue growth in 2026"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from analyzer import analyze_claim
from retriever import DenseRetriever
from sec import Filing, chunk_text, fetch_text, list_filings, lookup_cik


LOG_DIR = Path(__file__).parent / "logs"


def load_claims(args: argparse.Namespace) -> list[str]:
    if args.claim:
        return [args.claim]
    if args.claims:
        data = json.loads(Path(args.claims).read_text())
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and "claims" in data:
            return [str(x) for x in data["claims"]]
        raise ValueError("--claims file must be a JSON list or object with 'claims' key")
    raise SystemExit("Provide --claim or --claims")


def resolve_cik(args: argparse.Namespace) -> str:
    if args.cik:
        return str(int(args.cik)).zfill(10)
    if args.company:
        cik = lookup_cik(args.company)
        if cik is None:
            raise SystemExit(f"Could not resolve '{args.company}' to a CIK")
        return cik
    raise SystemExit("Provide --company or --cik")


def build_index(
    cik: str, forms: list[str], limit: int, verbose: bool = True
) -> DenseRetriever:
    filings: list[Filing] = list_filings(cik, forms=forms, limit=limit)
    if not filings:
        raise SystemExit(
            f"No filings found for CIK {cik} among forms {forms}. "
            "Private startups often have no filings; try forms like ['D']."
        )
    if verbose:
        print(f"[+] Found {len(filings)} filings; indexing...")
    retriever = DenseRetriever()
    for f in filings:
        if verbose:
            print(f"    - {f.form} {f.filing_date} ({f.accession})")
        try:
            text = fetch_text(f)
        except Exception as e:
            print(f"      ! fetch failed: {e}", file=sys.stderr)
            continue
        chunks = chunk_text(text)
        retriever.add(f, chunks)
    retriever.build()
    if verbose:
        print(f"[+] Indexed {retriever.size} passages")
    return retriever


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    cik = resolve_cik(args)
    claims = load_claims(args)
    forms = [f.strip() for f in args.forms.split(",") if f.strip()]

    print(f"[+] Target CIK: {cik}")
    print(f"[+] {len(claims)} claim(s) to verify against forms {forms}")

    retriever = build_index(cik, forms=forms, limit=args.filings)
    client = anthropic.Anthropic()

    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"discrepancies_{cik}_{ts}.json"

    results = []
    for i, claim in enumerate(claims, start=1):
        print(f"\n[{i}/{len(claims)}] Analyzing: {claim[:100]}...")
        hits = retriever.search(claim, top_k=args.top_k)
        assessment = analyze_claim(client, claim, hits)
        entry = {
            "claim": claim,
            "verdict": assessment.verdict,
            "forward_looking": assessment.forward_looking,
            "severity": assessment.severity,
            "explanation": assessment.explanation,
            "cited_passages": [
                {
                    "passage_num": p,
                    "form": hits[p - 1].passage.filing.form,
                    "accession": hits[p - 1].passage.filing.accession,
                    "filing_date": hits[p - 1].passage.filing.filing_date,
                    "url": hits[p - 1].passage.filing.url,
                    "score": hits[p - 1].score,
                    "excerpt": hits[p - 1].passage.text[:400],
                }
                for p in assessment.cited_passages
                if 1 <= p <= len(hits)
            ],
        }
        results.append(entry)
        flag = "⚠️  FLAG" if (
            assessment.verdict == "CONTRADICTS" and assessment.forward_looking
        ) else f"    {assessment.verdict}"
        print(f"    {flag} [{assessment.severity}] forward_looking={assessment.forward_looking}")
        print(f"    {assessment.explanation}")

    log_path.write_text(
        json.dumps(
            {"cik": cik, "generated_at": ts, "forms": forms, "results": results},
            indent=2,
        )
    )
    print(f"\n[+] Discrepancy log written to {log_path}")

    flagged = sum(
        1 for r in results if r["verdict"] == "CONTRADICTS" and r["forward_looking"]
    )
    print(f"[+] Flagged contradictory forward-looking statements: {flagged}")
    return 1 if flagged else 0


def main() -> int:
    p = argparse.ArgumentParser(description="VC compliance agent: verify startup claims against SEC filings.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--company", help="Ticker or company name (e.g., AAPL, 'Apple Inc.')")
    src.add_argument("--cik", help="10-digit SEC CIK (e.g., 0000320193)")

    claim_src = p.add_mutually_exclusive_group(required=True)
    claim_src.add_argument("--claim", help="A single claim to verify (string)")
    claim_src.add_argument("--claims", help="Path to JSON file with a list of claims")

    p.add_argument(
        "--forms", default="10-K,10-Q,S-1,8-K",
        help="Comma-separated SEC form types to index (default: 10-K,10-Q,S-1,8-K)",
    )
    p.add_argument("--filings", type=int, default=3, help="Max number of filings to index")
    p.add_argument("--top-k", type=int, default=5, help="Top-k passages per claim")

    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
