"""Compliance Agent — entry point.

Cross-references claims about a startup against its SEC filings using dense
vector retrieval. Flags contradictory forward-looking statements and writes a
discrepancy log to ./logs/.

Usage:
    python agent.py --company AAPL --claims examples/sample_claims.json
    python agent.py --cik 0000320193 --claim "We expect 50% YoY revenue growth in 2026"
    python agent.py --deck deck_contexts/pitch.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from analyzer import analyze_claim, analyze_industry_claim
from deck_context import DeckContext
from retriever import DenseRetriever
from sec import Filing, chunk_text, fetch_text, list_filings, lookup_cik
from version import ANALYZER_VERSION, EXTRACTOR_VERSION


LOG_DIR = Path(__file__).parent / "logs"

# Give the Anthropic client an explicit timeout so a hung request can't stall
# the whole run. Analysis calls with adaptive thinking can take 30-60s; web
# search tool loops can take longer. 300s is a generous upper bound.
ANTHROPIC_TIMEOUT = 300.0


def _ts() -> str:
    """HH:MM:SS timestamp for CLI log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------- helpers

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
    return []


def resolve_cik(args: argparse.Namespace, deck: DeckContext | None) -> str | None:
    """Resolve a 10-digit CIK from (in order): --cik, --company, deck company name/ticker."""
    if args.cik:
        return str(int(args.cik)).zfill(10)
    if args.company:
        return lookup_cik(args.company)
    if deck:
        key = deck.company_lookup_key()
        if key:
            if key.isdigit() and len(key) <= 10:
                return key.zfill(10)
            return lookup_cik(key)
    return None


# ---------------------------------------------------------- orchestration

def build_index(
    cik: str, forms: list[str], limit: int, verbose: bool = True
) -> tuple[DenseRetriever | None, list[Filing]]:
    """Index recent filings. Returns (retriever, filings). retriever is None if no filings."""
    filings: list[Filing] = list_filings(cik, forms=forms, limit=limit)
    if not filings:
        return None, []
    if verbose:
        _log(f"[+] Found {len(filings)} filings; indexing...")
    retriever = DenseRetriever()
    for f in filings:
        if verbose:
            _log(f"    - fetching {f.form} {f.filing_date} ({f.accession})")
        try:
            text = fetch_text(f)
        except Exception as e:
            _log(f"      ! fetch failed: {e}")
            continue
        retriever.add(f, chunk_text(text))
    if retriever.size == 0:
        return None, filings
    if verbose:
        _log(f"[+] Building dense index over {retriever.size} passages...")
    retriever.build()
    if verbose:
        _log(f"[+] Indexed {retriever.size} passages")
    return retriever, filings


def _resolve_workers(model: str | None, requested: int | None) -> int:
    """How many parallel threads to use for claim analysis.

    Local Ollama models are typically single-threaded (one inference at a time),
    so cap at 1. Cloud models can safely run 4 concurrent calls before hitting
    Anthropic's default rate limits — callers can raise this if they have
    higher-tier limits.
    """
    from llm_local import is_local_model
    if is_local_model(model):
        return 1
    if requested is not None:
        return max(1, requested)
    return 4  # safe default for standard Anthropic API tier


def iter_compliance_report(
    *,
    claims: list[str],
    cik: str | None = None,
    deck: DeckContext | None = None,
    forms: list[str] | None = None,
    filings_limit: int = 3,
    top_k: int = 5,
    verbose: bool = True,
    analyzer_model: str | None = None,
    extractor_model: str | None = None,
    max_workers: int | None = None,
):
    """Generator that yields events as each claim is analyzed.

    Callers get findings immediately — no waiting for the full run.

    Event shapes
    ────────────
    {"event": "start",        "data": {cik, company_name, assumed_industry,
                                        total_claims, forms, generated_at}}
    {"event": "warning",      "data": {"message": str}}
    {"event": "status",       "data": {"message": str}}
    {"event": "claim_result", "data": {"index": int, "total": int,
                                        "entry": {claim, verdict, ...}}}
    {"event": "done",         "data": {"report": {full aggregate dict}}}
    """
    load_dotenv()
    forms = forms or ["10-K", "10-Q", "S-1", "8-K"]

    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cik_label = cik or "unknown"

    assumed_industry = deck.extraction.company.industry if deck else None
    company_name = deck.extraction.company.name if deck else None

    report: dict = {
        "generated_at": ts,
        "cik": cik,
        "forms": forms,
        "deck_context_used": deck is not None,
        "assumed_industry": assumed_industry,
        "company_name": company_name,
        "extractor_model": extractor_model or os.environ.get("EXTRACTOR_MODEL", "claude-haiku-4-5"),
        "extractor_version": EXTRACTOR_VERSION,
        "analyzer_model": analyzer_model or os.environ.get("ANALYZER_MODEL", "claude-sonnet-4-6"),
        "analyzer_version": ANALYZER_VERSION,
        "claims_analyzed": 0,
        "flagged_forward_looking_contradictions": 0,
        "results": [],
        "warnings": [],
    }

    yield {
        "event": "start",
        "data": {
            "cik": cik,
            "company_name": company_name,
            "assumed_industry": assumed_industry,
            "total_claims": len(claims),
            "forms": forms,
            "generated_at": ts,
        },
    }

    if not claims:
        w = "No claims provided. Extract claims from a pitch deck or supply them via --claim / --claims."
        report["warnings"].append(w)
        yield {"event": "warning", "data": {"message": w}}
        yield {"event": "done", "data": {"report": report}}
        return

    # ── No CIK: web-search for market claims, INSUFFICIENT_EVIDENCE for rest ──
    if cik is None:
        industry_note = (
            f"Assumed industry: {assumed_industry}"
            if assumed_industry
            else "No industry inferred from deck — web-search analyses will scope from the claim itself"
        )
        w = (
            "Could not resolve company to an SEC CIK (common for private / early-stage "
            f"startups). {industry_note}. Market/industry claims will be assessed via "
            "web search; company-specific claims remain INSUFFICIENT_EVIDENCE."
        )
        report["warnings"].append(w)
        yield {"event": "warning", "data": {"message": w}}

        claim_meta = {c.text: c for c in deck.extraction.claims} if deck else {}
        client = anthropic.Anthropic(timeout=ANTHROPIC_TIMEOUT)
        workers = _resolve_workers(analyzer_model, max_workers)

        def _analyze_one(args):
            i, claim = args
            meta = claim_meta.get(claim)
            category = meta.category if meta else None
            forward_looking_hint = meta.likely_forward_looking if meta else None
            if category == "market":
                try:
                    assessment, web_sources = analyze_industry_claim(
                        client, claim,
                        company_name=company_name or "unknown",
                        industry=assumed_industry,
                        model=analyzer_model,
                    )
                    return i, {
                        "claim": claim, "verdict": assessment.verdict,
                        "forward_looking": assessment.forward_looking,
                        "severity": assessment.severity,
                        "explanation": assessment.explanation,
                        "missing_information": assessment.missing_information,
                        "cited_passages": [], "web_sources": web_sources,
                        "analysis_method": "web_search",
                        "assumed_industry": assumed_industry,
                    }
                except Exception as exc:
                    return i, {
                        "claim": claim, "verdict": "INSUFFICIENT_EVIDENCE",
                        "forward_looking": forward_looking_hint, "severity": "NONE",
                        "explanation": f"Web-search analysis failed: {exc}",
                        "missing_information": "Retry web search, or provide authoritative industry reports.",
                        "cited_passages": [], "web_sources": [],
                        "analysis_method": "web_search_failed",
                        "assumed_industry": assumed_industry,
                    }
            else:
                reason_category = category or "unknown category"
                return i, {
                    "claim": claim, "verdict": "INSUFFICIENT_EVIDENCE",
                    "forward_looking": forward_looking_hint, "severity": "NONE",
                    "explanation": (
                        f"This is a company-specific claim ({reason_category}) and cannot be "
                        "verified without SEC filings for the target company. No CIK was resolved from the deck."
                    ),
                    "missing_information": (
                        "SEC filings for this company (10-K, 10-Q, S-1, Form D), or direct "
                        "company disclosures (data room, audited financials)."
                    ),
                    "cited_passages": [], "web_sources": [],
                    "analysis_method": "none",
                    "assumed_industry": assumed_industry,
                }

        indexed = list(enumerate(claims, start=1))
        results_map: dict[int, dict] = {}
        if verbose:
            _log(f"[+] Analyzing {len(claims)} claims with {workers} parallel workers...")
        yield {"event": "status", "data": {"message": f"Analyzing {len(claims)} claims ({workers} parallel)…"}}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_analyze_one, item): item for item in indexed}
            for future in as_completed(futures):
                i, entry = future.result()
                results_map[i] = entry
                if verbose:
                    _log(f"    [{i}/{len(claims)}] {entry['verdict']} [{entry['severity']}] {entry['claim'][:80]}")
                yield {"event": "claim_result", "data": {"index": i, "total": len(claims), "entry": entry}}

        # Restore order for the final report
        for i in range(1, len(claims) + 1):
            report["results"].append(results_map[i])

        report["claims_analyzed"] = len(claims)
        _write_log(report, cik_label, ts)
        yield {"event": "done", "data": {"report": report}}
        return

    # ── CIK available: build dense index then analyze against SEC filings ──
    yield {"event": "status", "data": {"message": "Fetching SEC filings and building search index…"}}

    retriever, filings = build_index(cik, forms=forms, limit=filings_limit, verbose=verbose)
    if retriever is None:
        msg = (
            f"No SEC filings of types {forms} found for CIK {cik}. "
            "This is common for private / early-stage startups. "
            "Consider rerunning with --forms D,S-1 or treat claims as unverified."
        )
        report["warnings"].append(msg)
        yield {"event": "warning", "data": {"message": msg}}
        for i, claim in enumerate(claims, start=1):
            entry = {
                "claim": claim,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "forward_looking": None,
                "severity": "NONE",
                "explanation": msg,
                "missing_information": "Applicable SEC filings for this company.",
                "cited_passages": [],
                "web_sources": [],
            }
            report["results"].append(entry)
            yield {"event": "claim_result", "data": {"index": i, "total": len(claims), "entry": entry}}
        report["claims_analyzed"] = len(claims)
        _write_log(report, cik_label, ts)
        yield {"event": "done", "data": {"report": report}}
        return

    client = anthropic.Anthropic(timeout=ANTHROPIC_TIMEOUT)
    deck_ctx_str = deck.clarifying_context() if deck else None
    workers = _resolve_workers(analyzer_model, max_workers)

    if verbose:
        _log(f"[+] Analyzing {len(claims)} claims with {workers} parallel workers...")
    yield {"event": "status", "data": {"message": f"Analyzing {len(claims)} claims against SEC filings ({workers} parallel)…"}}

    def _analyze_sec(args):
        i, claim = args
        hits = retriever.search(claim, top_k=top_k)
        assessment = analyze_claim(client, claim, hits, deck_context=deck_ctx_str, model=analyzer_model)
        entry = {
            "claim": claim,
            "verdict": assessment.verdict,
            "forward_looking": assessment.forward_looking,
            "severity": assessment.severity,
            "explanation": assessment.explanation,
            "missing_information": assessment.missing_information,
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
            "web_sources": [],
        }
        return i, entry

    results_map: dict[int, dict] = {}
    flagged = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_sec, (i, c)): i for i, c in enumerate(claims, start=1)}
        for future in as_completed(futures):
            i, entry = future.result()
            results_map[i] = entry
            if entry["verdict"] == "CONTRADICTS" and entry.get("forward_looking"):
                flagged += 1
            if verbose:
                flag = "⚠️  FLAG" if (entry["verdict"] == "CONTRADICTS" and entry.get("forward_looking")) else f"    {entry['verdict']}"
                _log(f"    [{i}/{len(claims)}] {flag} [{entry['severity']}] {entry['claim'][:80]}")
            yield {"event": "claim_result", "data": {"index": i, "total": len(claims), "entry": entry}}

    report["claims_analyzed"] = len(claims)
    report["flagged_forward_looking_contradictions"] = flagged
    _write_log(report, cik_label, ts)
    yield {"event": "done", "data": {"report": report}}


def run_compliance_report(
    *,
    claims: list[str],
    cik: str | None = None,
    deck: DeckContext | None = None,
    forms: list[str] | None = None,
    filings_limit: int = 3,
    top_k: int = 5,
    verbose: bool = True,
    analyzer_model: str | None = None,
    max_workers: int | None = None,
) -> dict:
    """Thin wrapper around iter_compliance_report — returns the final report dict.

    Used by the CLI and any caller that doesn't need streaming.
    """
    report: dict = {}
    for event in iter_compliance_report(
        claims=claims,
        cik=cik,
        deck=deck,
        forms=forms,
        filings_limit=filings_limit,
        top_k=top_k,
        verbose=verbose,
        analyzer_model=analyzer_model,
        max_workers=max_workers,
    ):
        if event["event"] == "done":
            report = event["data"]["report"]
    return report


def _write_log(report: dict, cik_label: str, ts: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"discrepancies_{cik_label}_{ts}.json"
    log_path.write_text(json.dumps(report, indent=2))
    report["log_path"] = str(log_path)


# ------------------------------------------------------------------- CLI

def main() -> int:
    p = argparse.ArgumentParser(
        description="VC compliance agent: verify startup claims against SEC filings."
    )
    p.add_argument("--company", help="Ticker or company name (e.g., AAPL, 'Apple Inc.')")
    p.add_argument("--cik", help="10-digit SEC CIK (e.g., 0000320193)")
    p.add_argument("--claim", help="A single claim to verify")
    p.add_argument("--claims", help="Path to JSON file with a list of claims")
    p.add_argument(
        "--deck",
        help="Path to deck context JSON (produced by the extractor). "
        "Supplies claims + company identity if not provided via other flags.",
    )
    p.add_argument(
        "--forms", default="10-K,10-Q,S-1,8-K",
        help="Comma-separated SEC form types to index",
    )
    p.add_argument("--filings", type=int, default=3, help="Max filings to index")
    p.add_argument("--top-k", type=int, default=5, help="Top-k passages per claim")
    args = p.parse_args()

    deck: DeckContext | None = None
    if args.deck:
        deck = DeckContext.load(args.deck)

    claims = load_claims(args)
    if not claims and deck:
        claims = deck.claims_for_verification()

    if not claims:
        print("error: provide claims via --claim, --claims, or --deck", file=sys.stderr)
        return 2

    cik = resolve_cik(args, deck)
    forms = [f.strip() for f in args.forms.split(",") if f.strip()]

    # Stream findings to the terminal as each claim finishes.
    report: dict = {}
    for event in iter_compliance_report(
        claims=claims,
        cik=cik,
        deck=deck,
        forms=forms,
        filings_limit=args.filings,
        top_k=args.top_k,
        verbose=True,
    ):
        if event["event"] == "claim_result":
            e = event["data"]["entry"]
            idx = event["data"]["index"]
            total = event["data"]["total"]
            flag = "⚠️  FLAG" if (e["verdict"] == "CONTRADICTS" and e.get("forward_looking")) else ""
            _log(f"\n── Claim {idx}/{total} {flag}")
            _log(f"   {e['claim']}")
            _log(f"   Verdict : {e['verdict']}  Severity: {e['severity']}")
            _log(f"   {e['explanation']}")
            if e.get("missing_information"):
                _log(f"   Missing : {e['missing_information']}")
            for src in e.get("web_sources", []):
                _log(f"   Source  : {src['title']}  {src['url']}")
        elif event["event"] == "done":
            report = event["data"]["report"]

    _log(f"\n[+] Discrepancy log written to {report.get('log_path')}")
    _log(
        "[+] Flagged contradictory forward-looking statements: "
        f"{report.get('flagged_forward_looking_contradictions', 0)}"
    )
    if report.get("warnings"):
        _log("[!] Warnings:")
        for w in report["warnings"]:
            _log(f"    - {w}")
    return 1 if report.get("flagged_forward_looking_contradictions") else 0


if __name__ == "__main__":
    sys.exit(main())
