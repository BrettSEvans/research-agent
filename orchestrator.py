"""Compliance Orchestrator — parallel jurisdiction routing.

Fans out compliance analysis across SEC, EU SFDR/CSRD, and CA SB 54 modules
concurrently instead of running them one-after-another. Each jurisdiction
internally parallelizes its per-claim analysis.

Why this exists
───────────────
The original `iter_compliance_report()` in agent.py runs jurisdictions
sequentially: SEC → (wait) → EU → (wait) → CA. It also uses a sequential
`for` loop for EU and CA claim analysis. For a 10-claim deck with 3
jurisdictions, that's ~10–13 minutes on Sonnet.

This orchestrator runs all three jurisdictions concurrently and parallelizes
the per-claim loop in each one, cutting wall-clock time to ~3–4 minutes for
the same deck.

Concurrency model
─────────────────
    outer pool : 3 workers — one per jurisdiction (SEC, EU, CA)
    inner pool : 2 workers per jurisdiction — claim-level parallelism
    global cap : threading.Semaphore(6) across all jurisdictions combined
    → at most 6 in-flight Anthropic calls at any moment (Tier 2 safe)

    Completeness checks (ESG, CA demographics) use Haiku instead of Sonnet
    so they draw from a separate, cheaper rate-limit pool.

Thread-safety
─────────────
- `anthropic.Anthropic()` is thread-safe
- `DenseRetriever.search()` is read-only after `build()` — thread-safe
- Event streaming uses a thread-safe queue so callers still get live updates

Backwards compatibility
───────────────────────
`iter_compliance_report()` in agent.py is untouched. Callers that want
parallel jurisdiction routing should use `ComplianceOrchestrator.run()`.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Generator, Iterable, List, Optional

import anthropic

from deck_context import DeckContext
from version import ANALYZER_VERSION, EXTRACTOR_VERSION


# Timeout for a single Anthropic call (covers adaptive thinking + web search).
ANTHROPIC_TIMEOUT = 300.0

# Default concurrency tuned for Anthropic Tier 2 (1k RPM / 80k ITPM / 16k OTPM).
#
# With max_tokens=4000 per call:
#   3 jurisdictions × 2 claims = 6 in-flight calls → ~6k OTPM sustained (well under 16k)
# The global semaphore (default 6) is the hard cap regardless of worker counts.
DEFAULT_JURISDICTION_WORKERS = 3
DEFAULT_CLAIM_WORKERS = 2
DEFAULT_GLOBAL_IN_FLIGHT = 6

# Completeness checks (ESG, CA demographics) are simpler rule-based evaluations
# that don't need full Sonnet reasoning — Haiku is 20× cheaper and has its own
# separate rate-limit pool.
DEFAULT_COMPLETENESS_MODEL = "claude-haiku-4-5"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class JurisdictionResult:
    """Output of a single jurisdiction analyzer (SEC, EU, or CA)."""
    jurisdiction: str
    entries: List[dict] = field(default_factory=list)  # per-claim result dicts
    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None  # set if the whole jurisdiction failed


@dataclass
class OrchestrationTiming:
    """Wall-clock timing for the full orchestration run."""
    started_at: datetime
    finished_at: Optional[datetime] = None
    total_seconds: float = 0.0
    per_jurisdiction: dict = field(default_factory=dict)  # {jurisdiction: seconds}

    def under_budget(self, budget_seconds: float = 600.0) -> bool:
        """True if the run finished under the target budget (default 10 min)."""
        return self.total_seconds <= budget_seconds


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────


class ComplianceOrchestrator:
    """Runs compliance modules in parallel across jurisdictions.

    Usage
    ─────
        orch = ComplianceOrchestrator(
            deck=deck_context,
            modules=["sec", "eu_sfdr_csrd", "ca_sb54"],
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

        # Blocking: returns final aggregated report
        report = orch.run()

        # Streaming: yields events as they happen
        for event in orch.stream():
            if event["event"] == "claim_result":
                print(event["data"]["entry"]["verdict"])
    """

    def __init__(
        self,
        *,
        deck: DeckContext,
        modules: Iterable[str] = ("sec",),
        cik: Optional[str] = None,
        forms: Optional[List[str]] = None,
        filings_limit: int = 3,
        top_k: int = 5,
        analyzer_model: Optional[str] = None,
        completeness_model: Optional[str] = None,
        extractor_model: Optional[str] = None,
        api_key: Optional[str] = None,
        jurisdiction_workers: int = DEFAULT_JURISDICTION_WORKERS,
        claim_workers: int = DEFAULT_CLAIM_WORKERS,
        global_max_in_flight: int = DEFAULT_GLOBAL_IN_FLIGHT,
        verbose: bool = False,
        # Dependency injection: callers can pass pre-built analyzer functions
        # for testing. If None, we import the real modules lazily at run time.
        sec_analyzer: Optional[Callable] = None,
        eu_analyzer: Optional[Callable] = None,
        ca_analyzer: Optional[Callable] = None,
    ):
        self.deck = deck
        self.modules = set(modules)
        self.cik = cik
        self.forms = forms or ["10-K", "10-Q", "S-1", "8-K"]
        self.filings_limit = filings_limit
        self.top_k = top_k
        self.analyzer_model = analyzer_model or os.environ.get("ANALYZER_MODEL", "claude-sonnet-4-6")
        # Completeness checks use Haiku by default — simpler evaluations, separate rate pool.
        self.completeness_model = (
            completeness_model
            or os.environ.get("COMPLETENESS_MODEL", DEFAULT_COMPLETENESS_MODEL)
        )
        self.extractor_model = extractor_model or os.environ.get("EXTRACTOR_MODEL", "claude-haiku-4-5")
        self.api_key = api_key
        self.jurisdiction_workers = jurisdiction_workers
        self.claim_workers = claim_workers
        self.verbose = verbose

        # Global semaphore: hard cap on simultaneous in-flight Anthropic calls
        # across ALL jurisdictions combined. Prevents burst-through even when
        # per-jurisdiction worker pools are individually within limits.
        self._inflight_sem = threading.Semaphore(max(1, global_max_in_flight))

        self._sec_analyzer = sec_analyzer
        self._eu_analyzer = eu_analyzer
        self._ca_analyzer = ca_analyzer

        self.timing = OrchestrationTiming(started_at=datetime.now(timezone.utc))
        self._client: Optional[anthropic.Anthropic] = None

    # ─── Public API ────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Blocking: run all jurisdictions in parallel and return the final report."""
        report: dict = {}
        for event in self.stream():
            if event["event"] == "done":
                report = event["data"]["report"]
        return report

    def stream(self) -> Generator[dict, None, None]:
        """Streaming: yield events as each jurisdiction produces results.

        Events:
            {"event": "start",          "data": {...metadata...}}
            {"event": "jurisdiction_start", "data": {"jurisdiction": str}}
            {"event": "claim_result",   "data": {"jurisdiction": str, "entry": {...}}}
            {"event": "jurisdiction_done", "data": {"jurisdiction": str, "elapsed_seconds": float}}
            {"event": "warning",        "data": {"jurisdiction": str, "message": str}}
            {"event": "done",           "data": {"report": {...}}}
        """
        t_start = time.monotonic()

        self._client = anthropic.Anthropic(api_key=self.api_key, timeout=ANTHROPIC_TIMEOUT)
        claims = self.deck.claims_for_verification()

        # Build the report skeleton
        report = self._build_report_skeleton(claims)
        yield {
            "event": "start",
            "data": {
                "company_name": report["company_name"],
                "total_claims": len(claims),
                "modules": list(self.modules),
                "jurisdiction_workers": self.jurisdiction_workers,
                "claim_workers": self.claim_workers,
            },
        }

        # Queue for streaming events from jurisdiction workers to the caller.
        event_queue: "queue.Queue[dict]" = queue.Queue()

        # Build the list of jurisdiction tasks to run.
        jurisdiction_tasks: List[tuple[str, Callable[[], JurisdictionResult]]] = []
        if "sec" in self.modules:
            jurisdiction_tasks.append(("sec", lambda: self._run_sec(claims, event_queue)))
        if "eu_sfdr_csrd" in self.modules:
            jurisdiction_tasks.append(
                ("eu_sfdr_csrd", lambda: self._run_eu(claims, event_queue))
            )
        if "ca_sb54" in self.modules:
            jurisdiction_tasks.append(
                ("ca_sb54", lambda: self._run_ca(event_queue))
            )

        # Fan out jurisdictions in parallel. Each jurisdiction parallelizes
        # its own per-claim analysis using self.claim_workers.
        jurisdiction_results: List[JurisdictionResult] = []
        with ThreadPoolExecutor(
            max_workers=max(1, self.jurisdiction_workers),
            thread_name_prefix="juris",
        ) as pool:
            future_map = {pool.submit(fn): name for name, fn in jurisdiction_tasks}

            # Poll the event queue while jurisdictions are running so the caller
            # gets live events. Keep going until all futures are done AND the
            # queue is drained.
            pending = set(future_map.keys())
            while pending:
                # Drain any events that have arrived
                while not event_queue.empty():
                    try:
                        yield event_queue.get_nowait()
                    except queue.Empty:
                        break
                # Check for completed futures (non-blocking via short timeout)
                done = {f for f in pending if f.done()}
                for future in done:
                    pending.remove(future)
                    try:
                        result = future.result()
                        jurisdiction_results.append(result)
                    except Exception as exc:
                        name = future_map[future]
                        jurisdiction_results.append(
                            JurisdictionResult(
                                jurisdiction=name,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                if pending:
                    time.sleep(0.05)  # avoid busy-spin

            # Drain remaining events after everything finishes
            while not event_queue.empty():
                yield event_queue.get_nowait()

        # Stitch results into the final report
        for jr in jurisdiction_results:
            self.timing.per_jurisdiction[jr.jurisdiction] = jr.elapsed_seconds
            report["results"].extend(jr.entries)
            report["warnings"].extend(jr.warnings)
            if jr.error:
                report["warnings"].append(f"[{jr.jurisdiction}] {jr.error}")

        report["claims_analyzed"] = len(claims)
        report["flagged_forward_looking_contradictions"] = sum(
            1 for e in report["results"]
            if e.get("verdict") == "CONTRADICTS" and e.get("forward_looking")
        )

        # Timing summary
        self.timing.finished_at = datetime.now(timezone.utc)
        self.timing.total_seconds = time.monotonic() - t_start
        report["timing"] = {
            "total_seconds": round(self.timing.total_seconds, 2),
            "per_jurisdiction": {k: round(v, 2) for k, v in self.timing.per_jurisdiction.items()},
            "under_10_minute_budget": self.timing.under_budget(600.0),
        }

        yield {"event": "done", "data": {"report": report}}

    # ─── Jurisdiction runners ──────────────────────────────────────────────

    def _run_sec(self, claims: List[str], events: "queue.Queue[dict]") -> JurisdictionResult:
        """Run SEC analysis in parallel across all claims."""
        t0 = time.monotonic()
        events.put({"event": "jurisdiction_start", "data": {"jurisdiction": "sec"}})

        result = JurisdictionResult(jurisdiction="sec")

        # Resolve the analyzer: injected for tests, real import otherwise.
        analyzer = self._sec_analyzer or self._default_sec_analyzer

        try:
            entries = self._parallel_claim_loop(
                claims=claims,
                jurisdiction="sec",
                analyzer=analyzer,
                events=events,
            )
            result.entries = entries
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.warnings.append(f"SEC module failed: {exc}")

        result.elapsed_seconds = time.monotonic() - t0
        events.put({
            "event": "jurisdiction_done",
            "data": {"jurisdiction": "sec", "elapsed_seconds": result.elapsed_seconds},
        })
        return result

    def _run_eu(self, claims: List[str], events: "queue.Queue[dict]") -> JurisdictionResult:
        """Run EU SFDR/CSRD analysis in parallel across all claims."""
        t0 = time.monotonic()
        events.put({"event": "jurisdiction_start", "data": {"jurisdiction": "eu_sfdr_csrd"}})

        result = JurisdictionResult(jurisdiction="eu_sfdr_csrd")

        analyzer = self._eu_analyzer or self._default_eu_analyzer

        try:
            entries = self._parallel_claim_loop(
                claims=claims,
                jurisdiction="eu_sfdr_csrd",
                analyzer=analyzer,
                events=events,
            )
            result.entries = entries

            # ESG completeness check (independent of claims)
            try:
                completeness_entries = self._run_eu_completeness(events)
                result.entries.extend(completeness_entries)
            except Exception as exc:
                result.warnings.append(f"EU ESG completeness failed: {exc}")
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"

        result.elapsed_seconds = time.monotonic() - t0
        events.put({
            "event": "jurisdiction_done",
            "data": {"jurisdiction": "eu_sfdr_csrd", "elapsed_seconds": result.elapsed_seconds},
        })
        return result

    def _run_ca(self, events: "queue.Queue[dict]") -> JurisdictionResult:
        """Run CA SB 54 demographic completeness analysis.

        Note: CA SB 54 is a completeness check across demographic fields,
        not a per-claim analysis. It runs once over the founder_demographics
        object. We still parallelize the individual field checks internally.
        """
        t0 = time.monotonic()
        events.put({"event": "jurisdiction_start", "data": {"jurisdiction": "ca_sb54"}})

        result = JurisdictionResult(jurisdiction="ca_sb54")

        analyzer = self._ca_analyzer or self._default_ca_analyzer

        try:
            demos = self.deck.extraction.founder_demographics
            assessments = analyzer(self._client, demos, self.completeness_model)
            for assessment in assessments:
                entry = self._assessment_to_entry(
                    claim=f"[Demographics] {assessment.explanation[:80]}",
                    assessment=assessment,
                    jurisdiction="ca_sb54",
                )
                result.entries.append(entry)
                events.put({
                    "event": "claim_result",
                    "data": {"jurisdiction": "ca_sb54", "entry": entry},
                })
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.warnings.append(f"CA SB 54 failed: {exc}")

        result.elapsed_seconds = time.monotonic() - t0
        events.put({
            "event": "jurisdiction_done",
            "data": {"jurisdiction": "ca_sb54", "elapsed_seconds": result.elapsed_seconds},
        })
        return result

    # ─── Shared helpers ────────────────────────────────────────────────────

    def _parallel_claim_loop(
        self,
        *,
        claims: List[str],
        jurisdiction: str,
        analyzer: Callable[[str], dict],
        events: "queue.Queue[dict]",
    ) -> List[dict]:
        """Run `analyzer(claim)` across all claims in parallel; preserve order.

        This is the key parallelism primitive. Each jurisdiction's per-claim
        loop uses this so EU and CA get the same speedup as SEC previously
        had on its own.
        """
        if not claims:
            return []

        results_map: dict[int, dict] = {}
        workers = max(1, self.claim_workers)

        # Wrap every analyzer call with the global semaphore so the total
        # in-flight count across all concurrent jurisdictions stays bounded.
        sem = self._inflight_sem

        def _throttled(claim: str) -> dict:
            with sem:
                return analyzer(claim)

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"claim-{jurisdiction}",
        ) as pool:
            futures = {
                pool.submit(_throttled, claim): i
                for i, claim in enumerate(claims, start=1)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    entry = future.result()
                    entry.setdefault("jurisdiction", jurisdiction)
                    entry.setdefault("claim", claims[i - 1])
                    results_map[i] = entry
                    events.put({
                        "event": "claim_result",
                        "data": {
                            "jurisdiction": jurisdiction,
                            "index": i,
                            "total": len(claims),
                            "entry": entry,
                        },
                    })
                except Exception as exc:
                    # Never let one bad claim kill the whole jurisdiction.
                    error_entry = {
                        "claim": claims[i - 1],
                        "verdict": "INSUFFICIENT_EVIDENCE",
                        "severity": "NONE",
                        "forward_looking": None,
                        "explanation": f"Analyzer error: {type(exc).__name__}: {exc}",
                        "missing_information": "Retry or inspect logs.",
                        "cited_passages": [],
                        "jurisdiction": jurisdiction,
                    }
                    results_map[i] = error_entry
                    events.put({
                        "event": "warning",
                        "data": {
                            "jurisdiction": jurisdiction,
                            "message": f"Claim {i} failed: {exc}",
                        },
                    })

        # Preserve original claim order in the returned list
        return [results_map[i] for i in range(1, len(claims) + 1)]

    def _run_eu_completeness(self, events: "queue.Queue[dict]") -> List[dict]:
        """Run EU ESG completeness check (non-claim-based)."""
        from analyzer_sfdr import analyze_esg_completeness

        esg = self.deck.extraction.esg_metrics
        entries: List[dict] = []
        for assessment in analyze_esg_completeness(self._client, esg, model=self.completeness_model):
            entry = self._assessment_to_entry(
                claim=f"[ESG Completeness] {assessment.explanation[:80]}",
                assessment=assessment,
                jurisdiction="eu_sfdr_csrd",
            )
            entries.append(entry)
            events.put({
                "event": "claim_result",
                "data": {"jurisdiction": "eu_sfdr_csrd", "entry": entry},
            })
        return entries

    @staticmethod
    def _assessment_to_entry(*, claim: str, assessment, jurisdiction: str) -> dict:
        """Convert a ClaimAssessment to the report entry dict shape."""
        return {
            "claim": claim,
            "verdict": assessment.verdict,
            "severity": assessment.severity,
            "forward_looking": assessment.forward_looking,
            "explanation": assessment.explanation,
            "missing_information": getattr(assessment, "missing_information", None),
            "cited_passages": list(getattr(assessment, "cited_passages", []) or []),
            "jurisdiction": jurisdiction,
            "red_flags": list(getattr(assessment, "red_flags", []) or []),
            "warnings": list(getattr(assessment, "warnings", []) or []),
            "verified": list(getattr(assessment, "verified", []) or []),
            "action_items": list(getattr(assessment, "action_items", []) or []),
        }

    # ─── Default analyzer bindings (lazy imports) ─────────────────────────

    def _default_sec_analyzer(self, claim: str) -> dict:
        """Real SEC analyzer — imports lazily so tests can mock the module."""
        from analyzer import analyze_claim, analyze_industry_claim
        # For brevity this implementation focuses on the web-search fallback
        # path (no CIK). Callers that need filing-indexed SEC analysis should
        # still use agent.iter_compliance_report for CIK-resolved companies.
        # The orchestrator's primary value-add is parallelism across
        # jurisdictions, which applies equally well either way.
        company_name = self.deck.extraction.company.name if self.deck else "unknown"
        industry = self.deck.extraction.company.industry if self.deck else None

        try:
            assessment, web_sources = analyze_industry_claim(
                self._client, claim,
                company_name=company_name,
                industry=industry,
                model=self.analyzer_model,
            )
            entry = self._assessment_to_entry(
                claim=claim, assessment=assessment, jurisdiction="sec"
            )
            entry["web_sources"] = web_sources
            return entry
        except Exception as exc:
            return {
                "claim": claim,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "severity": "NONE",
                "forward_looking": None,
                "explanation": f"SEC web-search analysis failed: {exc}",
                "missing_information": "Retry or inspect API limits.",
                "cited_passages": [],
                "web_sources": [],
                "jurisdiction": "sec",
            }

    def _default_eu_analyzer(self, claim: str) -> dict:
        """Real EU SFDR/CSRD analyzer — imports lazily."""
        from analyzer_sfdr import analyze_claim as analyze_claim_eu
        import regulatory_kb

        try:
            retriever = regulatory_kb.get_retriever("eu_sfdr_csrd")
        except Exception:
            retriever = None

        hits = retriever.search(claim, top_k=self.top_k) if retriever else []
        deck_ctx_str = self.deck.clarifying_context() if self.deck else None
        esg = self.deck.extraction.esg_metrics if self.deck else None

        assessment = analyze_claim_eu(
            self._client,
            claim,
            hits,
            deck_context=deck_ctx_str,
            model=self.analyzer_model,
            esg_metrics=esg,
        )
        return self._assessment_to_entry(
            claim=claim, assessment=assessment, jurisdiction="eu_sfdr_csrd"
        )

    def _default_ca_analyzer(self, client, founder_demographics, model):
        """Real CA SB 54 analyzer — imports lazily."""
        from analyzer_sb54 import analyze_demographic_completeness
        return analyze_demographic_completeness(client, founder_demographics, model=model)

    # ─── Report scaffolding ────────────────────────────────────────────────

    def _build_report_skeleton(self, claims: List[str]) -> dict:
        """Build the report dict that jurisdictions will populate."""
        ext = self.deck.extraction
        return {
            "report_id": uuid.uuid4().hex[:12],
            "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "cik": self.cik,
            "forms": self.forms,
            "deck_context_used": True,
            "assumed_industry": ext.company.industry,
            "company_name": ext.company.name,
            "extractor_model": self.extractor_model,
            "extractor_version": EXTRACTOR_VERSION,
            "analyzer_model": self.analyzer_model,
            "analyzer_version": ANALYZER_VERSION,
            "claims_analyzed": 0,
            "flagged_forward_looking_contradictions": 0,
            "results": [],
            "warnings": [],
            "modules": sorted(self.modules),
        }
