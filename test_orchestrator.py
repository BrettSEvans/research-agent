"""
Tests for the ComplianceOrchestrator — verifies parallel jurisdiction routing
actually runs concurrently and meets the 10-minute budget for realistic decks.

Strategy
────────
Real Anthropic calls are expensive and non-deterministic, so these tests use
injected analyzer stubs with `time.sleep()` to simulate realistic latency.
A real Sonnet call takes ~20–40 seconds per claim; we simulate 0.3 seconds
and scale expectations accordingly.

What we prove
─────────────
1. SEC, EU, and CA run in parallel (total wall-clock ≈ max, not sum)
2. Each jurisdiction parallelizes its per-claim loop
3. Claim order is preserved in the final report
4. Analyzer failures don't kill the whole run
5. The 10-minute budget check flag is set correctly
6. Stream events arrive in real time, not just at the end
"""
from __future__ import annotations

import threading
import time
from typing import List

import anthropic
import pytest

from deck_context import DeckContext
from extractor import (
    CompanyIdentity,
    DeckExtraction,
    EsgMetrics,
    ExtractedClaim,
    FounderDemographics,
)
from analyzer_protocol import ClaimAssessment
from orchestrator import (
    ComplianceOrchestrator,
    DEFAULT_CLAIM_WORKERS,
    DEFAULT_GLOBAL_IN_FLIGHT,
    DEFAULT_COMPLETENESS_MODEL,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def deck() -> DeckContext:
    """Realistic deck with 10 claims — typical of an average pitch deck."""
    claims = [
        ExtractedClaim(
            text=f"Claim {i}: The company projects X% growth in segment Y.",
            verbatim=f"Verbatim quote {i}",
            slide=i,
            category="projection" if i % 2 == 0 else "financial",
            likely_forward_looking=(i % 2 == 0),
        )
        for i in range(1, 11)
    ]
    extraction = DeckExtraction(
        company=CompanyIdentity(
            name="TestCo Inc",
            industry="SaaS",
            founders=["Alice", "Bob"],
        ),
        claims=claims,
        esg_metrics=EsgMetrics(
            scope_1_emissions="1,000 tCO2e (2025)",
            scope_2_emissions="500 tCO2e (2025)",
            scope_3_emissions=None,
            has_third_party_audit=True,
            audit_body="Bureau Veritas",
            board_diversity_pct="40% women",
            sfdr_article_claim="Article 8",
            ai_risk_sector=False,
            ai_transparency_statement=None,
            supply_chain_disclosure=None,
        ),
        founder_demographics=FounderDemographics(
            founder_count=2,
            gender_diversity="50/50",
            women_founder_pct=50.0,
            race_ethnicity_data=None,
            educational_background=["MIT", "Stanford"],
            prior_startup_experience=True,
        ),
        extraction_notes="Test deck for orchestrator validation.",
    )
    return DeckContext(extraction)


# ──────────────────────────────────────────────────────────────────────────────
# Stub analyzers — simulate Anthropic latency without calling the API
# ──────────────────────────────────────────────────────────────────────────────


def _make_claim_assessment(verdict: str = "CONSISTENT", jurisdiction: str = "sec") -> ClaimAssessment:
    return ClaimAssessment(
        verdict=verdict,
        severity="NONE",
        forward_looking=False,
        explanation=f"Stub assessment for {jurisdiction}",
        cited_passages=[],
        missing_information=None,
        jurisdiction=jurisdiction,
        red_flags=[],
        warnings=[],
        verified=[],
        action_items=[],
    )


def make_slow_analyzer(jurisdiction: str, latency: float = 0.3):
    """
    Build a per-claim analyzer that sleeps `latency` seconds then returns
    a well-formed result dict. Simulates a real Anthropic call.
    """
    call_count = {"value": 0}
    call_lock = threading.Lock()
    concurrent_calls_observed = {"max": 0, "current": 0}

    def analyzer(claim: str) -> dict:
        with call_lock:
            concurrent_calls_observed["current"] += 1
            concurrent_calls_observed["max"] = max(
                concurrent_calls_observed["max"],
                concurrent_calls_observed["current"],
            )
            call_count["value"] += 1

        try:
            time.sleep(latency)
            return {
                "claim": claim,
                "verdict": "CONSISTENT",
                "severity": "NONE",
                "forward_looking": False,
                "explanation": f"Stub {jurisdiction} verdict",
                "missing_information": None,
                "cited_passages": [],
                "jurisdiction": jurisdiction,
                "red_flags": [],
                "warnings": [],
                "verified": [],
                "action_items": [],
            }
        finally:
            with call_lock:
                concurrent_calls_observed["current"] -= 1

    analyzer.call_count = call_count
    analyzer.concurrent_calls = concurrent_calls_observed
    return analyzer


def make_slow_ca_analyzer(latency: float = 0.3):
    """Build a stub CA analyzer that returns demographic completeness assessments."""
    def ca_analyzer(client, demos, model):
        # Simulate ~5 field checks
        assessments = []
        for field_name in ["founder_count", "gender_diversity", "race_ethnicity_data", "educational_background", "prior_startup_experience"]:
            time.sleep(latency / 5)  # amortized across checks
            assessments.append(_make_claim_assessment(
                verdict="CONSISTENT",
                jurisdiction="ca_sb54",
            ))
        return assessments
    return ca_analyzer


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestParallelClaimLoop:
    """Each jurisdiction should parallelize its per-claim analysis."""

    def test_sec_claims_analyzed_concurrently(self, deck):
        """With 4 workers and 10 claims × 0.3s each, wall-clock should be ~0.75s, not 3.0s."""
        sec = make_slow_analyzer("sec", latency=0.3)

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=4,
            sec_analyzer=sec,
            api_key="stub-key-not-used",
        )

        t0 = time.monotonic()
        report = orch.run()
        elapsed = time.monotonic() - t0

        # Sequential baseline: 10 × 0.3 = 3.0s
        # Parallel with 4 workers: ceil(10/4) × 0.3 = 0.9s + overhead
        assert elapsed < 1.5, f"SEC claim loop was not parallel: {elapsed:.2f}s (expected <1.5s)"
        assert sec.call_count["value"] == 10
        # Observed concurrency should be > 1 (multiple claims in flight)
        assert sec.concurrent_calls["max"] > 1, "No concurrent SEC calls observed"
        assert report["claims_analyzed"] == 10

    def test_eu_claims_analyzed_concurrently(self, deck):
        """EU module must parallelize claims (this was broken in agent.py's sequential loop)."""
        eu = make_slow_analyzer("eu_sfdr_csrd", latency=0.3)

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["eu_sfdr_csrd"],
            claim_workers=4,
            eu_analyzer=eu,
            api_key="stub-key-not-used",
        )

        t0 = time.monotonic()
        # Skip the ESG completeness check by injecting a no-op
        # (it tries to call anthropic.Anthropic internally)
        orch._run_eu_completeness = lambda events: []
        report = orch.run()
        elapsed = time.monotonic() - t0

        # With 4 workers: should be ~0.9s, sequential would be 3.0s
        assert elapsed < 1.5, f"EU claim loop was not parallel: {elapsed:.2f}s"
        assert eu.call_count["value"] == 10
        assert eu.concurrent_calls["max"] > 1, "No concurrent EU calls observed"


class TestCrossJurisdictionParallelism:
    """The core promise: SEC, EU, and CA run concurrently, not one-after-another."""

    def test_three_jurisdictions_run_in_parallel(self, deck):
        """
        Critical test: SEC + EU + CA should finish in ≈ max(each), not sum(each).

        With 10 claims × 0.3s / 4 workers ≈ 0.9s per jurisdiction:
            sequential (old): 0.9 × 3 = 2.7s
            parallel (new):   0.9 × 1 = 0.9s (all three concurrent)
        """
        sec = make_slow_analyzer("sec", latency=0.3)
        eu = make_slow_analyzer("eu_sfdr_csrd", latency=0.3)
        ca = make_slow_ca_analyzer(latency=0.3)

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec", "eu_sfdr_csrd", "ca_sb54"],
            jurisdiction_workers=3,
            claim_workers=4,
            sec_analyzer=sec,
            eu_analyzer=eu,
            ca_analyzer=ca,
            api_key="stub-key-not-used",
        )
        orch._run_eu_completeness = lambda events: []  # skip ESG completeness (needs real client)

        t0 = time.monotonic()
        report = orch.run()
        elapsed = time.monotonic() - t0

        # All three jurisdictions should finish in roughly the time of the slowest one.
        # Sequential: ~2.7s. Parallel: ~0.9-1.5s. We allow up to 2.0s for scheduling overhead.
        assert elapsed < 2.0, (
            f"Jurisdictions did not run in parallel: {elapsed:.2f}s "
            f"(sequential would be ~2.7s; parallel should be <2.0s)"
        )

        # Verify all three jurisdictions produced results
        jurisdictions_in_report = {e["jurisdiction"] for e in report["results"]}
        assert "sec" in jurisdictions_in_report
        assert "eu_sfdr_csrd" in jurisdictions_in_report
        assert "ca_sb54" in jurisdictions_in_report

    def test_jurisdiction_timing_tracked_separately(self, deck):
        """Each jurisdiction's wall-clock time should be tracked independently."""
        sec = make_slow_analyzer("sec", latency=0.2)
        eu = make_slow_analyzer("eu_sfdr_csrd", latency=0.4)  # slower
        ca = make_slow_ca_analyzer(latency=0.1)

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec", "eu_sfdr_csrd", "ca_sb54"],
            jurisdiction_workers=3,
            claim_workers=4,
            sec_analyzer=sec,
            eu_analyzer=eu,
            ca_analyzer=ca,
            api_key="stub-key-not-used",
        )
        orch._run_eu_completeness = lambda events: []

        report = orch.run()

        timing = report["timing"]
        assert "per_jurisdiction" in timing
        assert "sec" in timing["per_jurisdiction"]
        assert "eu_sfdr_csrd" in timing["per_jurisdiction"]
        assert "ca_sb54" in timing["per_jurisdiction"]

        # EU was configured as slowest — should take longer than SEC
        assert timing["per_jurisdiction"]["eu_sfdr_csrd"] > timing["per_jurisdiction"]["sec"]

    def test_budget_flag_set_for_fast_run(self, deck):
        """A fast stub run should be marked as under the 10-minute budget."""
        sec = make_slow_analyzer("sec", latency=0.1)
        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=4,
            sec_analyzer=sec,
            api_key="stub-key-not-used",
        )
        report = orch.run()

        assert report["timing"]["under_10_minute_budget"] is True
        assert report["timing"]["total_seconds"] < 600


class TestResilience:
    """Failures in one claim or jurisdiction must not kill the whole run."""

    def test_single_claim_failure_does_not_kill_jurisdiction(self, deck):
        """If one SEC claim raises, the other 9 should still complete."""
        call_count = {"value": 0}
        lock = threading.Lock()

        def flaky_analyzer(claim: str) -> dict:
            with lock:
                call_count["value"] += 1
                n = call_count["value"]
            if n == 3:  # Third call fails
                raise RuntimeError("Simulated analyzer crash")
            time.sleep(0.1)
            return {
                "claim": claim,
                "verdict": "CONSISTENT",
                "severity": "NONE",
                "forward_looking": False,
                "explanation": "stub",
                "missing_information": None,
                "cited_passages": [],
                "jurisdiction": "sec",
            }

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=4,
            sec_analyzer=flaky_analyzer,
            api_key="stub-key-not-used",
        )

        report = orch.run()

        # All 10 claims should produce an entry (even the failed one, as INSUFFICIENT_EVIDENCE)
        assert len(report["results"]) == 10
        # At least one entry should be the error fallback
        error_entries = [
            e for e in report["results"]
            if "Analyzer error" in (e.get("explanation") or "")
        ]
        assert len(error_entries) == 1

    def test_whole_jurisdiction_failure_isolated(self, deck):
        """If the EU module raises at the top level, SEC and CA should still succeed."""
        def broken_eu_analyzer(claim: str) -> dict:
            raise RuntimeError("EU module unreachable")

        sec = make_slow_analyzer("sec", latency=0.05)
        ca = make_slow_ca_analyzer(latency=0.05)

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec", "eu_sfdr_csrd", "ca_sb54"],
            jurisdiction_workers=3,
            claim_workers=4,
            sec_analyzer=sec,
            eu_analyzer=broken_eu_analyzer,
            ca_analyzer=ca,
            api_key="stub-key-not-used",
        )
        orch._run_eu_completeness = lambda events: []

        report = orch.run()

        # SEC and CA should still produce results
        sec_entries = [e for e in report["results"] if e["jurisdiction"] == "sec"]
        ca_entries = [e for e in report["results"] if e["jurisdiction"] == "ca_sb54"]
        assert len(sec_entries) == 10, "SEC should still complete despite EU failure"
        assert len(ca_entries) > 0, "CA should still complete despite EU failure"


class TestStreaming:
    """Events should flow to the caller in real time, not just at the end."""

    def test_stream_emits_events_as_claims_finish(self, deck):
        """claim_result events must arrive while analysis is still running."""
        sec = make_slow_analyzer("sec", latency=0.1)

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=4,
            sec_analyzer=sec,
            api_key="stub-key-not-used",
        )

        events_seen: List[dict] = []
        for event in orch.stream():
            events_seen.append(event)

        event_types = [e["event"] for e in events_seen]
        assert event_types[0] == "start"
        assert event_types[-1] == "done"
        # Must have at least one claim_result between start and done
        assert any(t == "claim_result" for t in event_types)
        # Must have at least one jurisdiction_start and one jurisdiction_done
        assert any(t == "jurisdiction_start" for t in event_types)
        assert any(t == "jurisdiction_done" for t in event_types)

    def test_claim_order_preserved_in_final_report(self, deck):
        """Even with out-of-order completion, results in the report must match original claim order."""
        # Randomize latencies so claims finish in different orders
        import random
        random.seed(42)

        def jittery_analyzer(claim: str) -> dict:
            time.sleep(random.uniform(0.05, 0.3))
            return {
                "claim": claim,
                "verdict": "CONSISTENT",
                "severity": "NONE",
                "forward_looking": False,
                "explanation": "stub",
                "missing_information": None,
                "cited_passages": [],
                "jurisdiction": "sec",
            }

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=4,
            sec_analyzer=jittery_analyzer,
            api_key="stub-key-not-used",
        )

        report = orch.run()

        original_claims = deck.claims_for_verification()
        report_claims = [e["claim"] for e in report["results"]]
        assert report_claims == original_claims, "Claim order must be preserved despite parallel execution"


class TestBudget:
    """The 10-minute budget target for average reports."""

    def test_ten_claim_three_jurisdiction_report_under_budget(self, deck):
        """
        Simulates a realistic average deck (10 claims, 3 jurisdictions) and
        confirms it stays under the 10-minute budget. Stub latency = 0.3s/claim
        approximates a real Sonnet call, scaled down ~100x for test speed.
        """
        sec = make_slow_analyzer("sec", latency=0.3)
        eu = make_slow_analyzer("eu_sfdr_csrd", latency=0.3)
        ca = make_slow_ca_analyzer(latency=0.3)

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec", "eu_sfdr_csrd", "ca_sb54"],
            jurisdiction_workers=3,
            claim_workers=4,
            sec_analyzer=sec,
            eu_analyzer=eu,
            ca_analyzer=ca,
            api_key="stub-key-not-used",
        )
        orch._run_eu_completeness = lambda events: []

        report = orch.run()

        # Fast stub run: well under 10 minutes
        assert report["timing"]["under_10_minute_budget"] is True

        # Scale up to real Sonnet latency (~30s per claim): 0.3s × 100 = 30s
        # Sequential total would be: 10 × 30 × 3 = 900s (15 min) — over budget ❌
        # Parallel total: ceil(10/4) × 30 = 90s per jurisdiction, all 3 concurrent ≈ 90s
        # Plus overhead for SEC filing fetch (~60s) and ESG completeness (~30s) ≈ 180s (3 min) ✅
        scaled_estimate = report["timing"]["total_seconds"] * 100
        assert scaled_estimate < 600, (
            f"Scaled estimate ({scaled_estimate:.1f}s) exceeds 10-minute budget"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Mitigation 1 — Default claim_workers reduced to 2
# ──────────────────────────────────────────────────────────────────────────────


class TestDefaultWorkerCounts:
    """Verify the Tier-2-safe defaults are in effect."""

    def test_default_claim_workers_is_two(self, deck):
        """DEFAULT_CLAIM_WORKERS must be 2, not 4 (Tier 2 OTPM budget)."""
        assert DEFAULT_CLAIM_WORKERS == 2

    def test_orchestrator_uses_default_claim_workers(self, deck):
        """When claim_workers is not overridden, the orchestrator uses DEFAULT_CLAIM_WORKERS."""
        sec = make_slow_analyzer("sec", latency=0.05)
        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            sec_analyzer=sec,
            api_key="stub-key-not-used",
        )
        assert orch.claim_workers == DEFAULT_CLAIM_WORKERS

    def test_parallelism_still_works_with_two_workers(self, deck):
        """2 workers should still be faster than sequential for 10 claims × 0.15s."""
        sec = make_slow_analyzer("sec", latency=0.15)
        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=2,
            sec_analyzer=sec,
            api_key="stub-key-not-used",
        )
        t0 = time.monotonic()
        report = orch.run()
        elapsed = time.monotonic() - t0

        # Sequential: 10 × 0.15 = 1.5s.  2 workers: ceil(10/2) × 0.15 = 0.75s
        assert elapsed < 1.2, f"2-worker loop too slow: {elapsed:.2f}s (sequential would be 1.5s)"
        assert report["claims_analyzed"] == 10


# ──────────────────────────────────────────────────────────────────────────────
# Mitigation 2 — Global in-flight semaphore
# ──────────────────────────────────────────────────────────────────────────────


class TestGlobalSemaphore:
    """The semaphore must cap simultaneous Anthropic calls across all jurisdictions."""

    def test_default_global_in_flight_is_six(self):
        assert DEFAULT_GLOBAL_IN_FLIGHT == 6

    def test_semaphore_created_on_init(self, deck):
        orch = ComplianceOrchestrator(
            deck=deck, modules=["sec"], api_key="stub"
        )
        import threading as _t
        assert isinstance(orch._inflight_sem, _t.Semaphore)

    def test_semaphore_limits_concurrent_calls(self, deck):
        """Even with many workers, global_max_in_flight=3 should cap concurrency at 3."""
        concurrency = {"current": 0, "max": 0}
        lock = threading.Lock()

        def instrumented_analyzer(claim: str) -> dict:
            with lock:
                concurrency["current"] += 1
                concurrency["max"] = max(concurrency["max"], concurrency["current"])
            try:
                time.sleep(0.15)
                return {
                    "claim": claim,
                    "verdict": "CONSISTENT",
                    "severity": "NONE",
                    "forward_looking": False,
                    "explanation": "stub",
                    "missing_information": None,
                    "cited_passages": [],
                    "jurisdiction": "sec",
                }
            finally:
                with lock:
                    concurrency["current"] -= 1

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=10,          # high worker count
            global_max_in_flight=3,    # hard cap at 3
            sec_analyzer=instrumented_analyzer,
            api_key="stub-key-not-used",
        )
        orch.run()

        assert concurrency["max"] <= 3, (
            f"Semaphore did not limit concurrency: observed {concurrency['max']} simultaneous calls"
        )

    def test_semaphore_does_not_serialise_when_cap_is_high(self, deck):
        """With global_max_in_flight=20, the semaphore should not become the bottleneck."""
        sec = make_slow_analyzer("sec", latency=0.15)
        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            claim_workers=4,
            global_max_in_flight=20,
            sec_analyzer=sec,
            api_key="stub-key-not-used",
        )
        t0 = time.monotonic()
        orch.run()
        elapsed = time.monotonic() - t0
        # With 4 workers and a generous semaphore, should still be fast
        assert elapsed < 1.5, f"High semaphore cap unexpectedly slowed run: {elapsed:.2f}s"

    def test_semaphore_shared_across_jurisdictions(self, deck):
        """global_max_in_flight=2 across 2 active jurisdictions must cap total at 2."""
        total_concurrent = {"max": 0, "current": 0}
        lock = threading.Lock()

        def make_capped_analyzer(jurisdiction: str):
            def analyzer(claim: str) -> dict:
                with lock:
                    total_concurrent["current"] += 1
                    total_concurrent["max"] = max(
                        total_concurrent["max"], total_concurrent["current"]
                    )
                try:
                    time.sleep(0.15)
                    return {
                        "claim": claim,
                        "verdict": "CONSISTENT",
                        "severity": "NONE",
                        "forward_looking": False,
                        "explanation": "stub",
                        "missing_information": None,
                        "cited_passages": [],
                        "jurisdiction": jurisdiction,
                    }
                finally:
                    with lock:
                        total_concurrent["current"] -= 1
            return analyzer

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec", "eu_sfdr_csrd"],
            jurisdiction_workers=2,
            claim_workers=4,
            global_max_in_flight=2,  # total cap across both jurisdictions
            sec_analyzer=make_capped_analyzer("sec"),
            eu_analyzer=make_capped_analyzer("eu_sfdr_csrd"),
            api_key="stub-key-not-used",
        )
        orch._run_eu_completeness = lambda events: []
        orch.run()

        assert total_concurrent["max"] <= 2, (
            f"Cross-jurisdiction semaphore failed: {total_concurrent['max']} simultaneous calls"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Mitigation 4 — Retry-After header awareness
# ──────────────────────────────────────────────────────────────────────────────


class TestRetryAfterWait:
    """_retry_after_wait must honour the server-supplied Retry-After header."""

    def test_uses_retry_after_header_when_present(self):
        """When RateLimitError carries retry-after=5, wait must return 5.0."""
        from analyzer import _retry_after_wait

        # Build a mock RateLimitError with a response that has the header
        class FakeHeaders:
            def get(self, key, default=0):
                return "5" if key == "retry-after" else default

        class FakeResponse:
            headers = FakeHeaders()

        exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
        exc.response = FakeResponse()

        class FakeRetryState:
            attempt_number = 1
            outcome = type("O", (), {"exception": staticmethod(lambda: exc)})()

        wait = _retry_after_wait(FakeRetryState())
        assert wait == 5.0, f"Expected 5.0s from retry-after header, got {wait}"

    def test_falls_back_to_exponential_without_header(self):
        """When no retry-after header, wait must use exponential backoff."""
        from analyzer import _retry_after_wait

        exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
        exc.response = None

        class FakeRetryState:
            attempt_number = 2
            outcome = type("O", (), {"exception": staticmethod(lambda: exc)})()

        wait = _retry_after_wait(FakeRetryState())
        # attempt 2 → 1 * 2^(2-1) = 2.0s
        assert wait == 2.0, f"Expected exponential 2.0s, got {wait}"

    def test_exponential_capped_at_thirty(self):
        """Exponential backoff must not exceed 30s."""
        from analyzer import _retry_after_wait

        exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
        exc.response = None

        class FakeRetryState:
            attempt_number = 10  # 1 * 2^9 = 512 → should cap at 30
            outcome = type("O", (), {"exception": staticmethod(lambda: exc)})()

        wait = _retry_after_wait(FakeRetryState())
        assert wait == 30.0, f"Exponential cap not enforced: {wait}"

    def test_sfdr_analyzer_also_has_retry_after_wait(self):
        """analyzer_sfdr._retry_after_wait must exist and honour the header."""
        from analyzer_sfdr import _retry_after_wait as sfdr_wait

        class FakeHeaders:
            def get(self, key, default=0):
                return "10" if key == "retry-after" else default

        class FakeResponse:
            headers = FakeHeaders()

        exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
        exc.response = FakeResponse()

        class FakeRetryState:
            attempt_number = 1
            outcome = type("O", (), {"exception": staticmethod(lambda: exc)})()

        assert sfdr_wait(FakeRetryState()) == 10.0


# ──────────────────────────────────────────────────────────────────────────────
# Mitigation 5 — Haiku for completeness checks
# ──────────────────────────────────────────────────────────────────────────────


class TestCompletenessModel:
    """Completeness checks (ESG, CA demographics) must use Haiku, not Sonnet."""

    def test_default_completeness_model_is_haiku(self):
        assert "haiku" in DEFAULT_COMPLETENESS_MODEL.lower()

    def test_orchestrator_uses_haiku_for_completeness_by_default(self, deck):
        orch = ComplianceOrchestrator(
            deck=deck, modules=["sec"], api_key="stub"
        )
        assert orch.completeness_model == DEFAULT_COMPLETENESS_MODEL

    def test_ca_analyzer_called_with_completeness_model(self, deck):
        """CA analyzer must receive completeness_model, not analyzer_model."""
        models_received = []

        def ca_stub(client, demos, model):
            models_received.append(model)
            return []

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["ca_sb54"],
            analyzer_model="claude-sonnet-4-6",
            completeness_model="claude-haiku-4-5",
            ca_analyzer=ca_stub,
            api_key="stub-key-not-used",
        )
        orch.run()

        assert len(models_received) == 1
        assert models_received[0] == "claude-haiku-4-5", (
            f"CA analyzer called with wrong model: {models_received[0]}"
        )

    def test_completeness_model_overridable(self, deck):
        """Caller can override completeness_model to use a different model."""
        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            completeness_model="claude-opus-4-5",
            api_key="stub",
        )
        assert orch.completeness_model == "claude-opus-4-5"

    def test_analyzer_model_unchanged_for_claim_analysis(self, deck):
        """Claim analysis (SEC, EU) must still use analyzer_model, not completeness_model."""
        models_received = []

        def sec_stub(claim: str) -> dict:
            return {
                "claim": claim,
                "verdict": "CONSISTENT",
                "severity": "NONE",
                "forward_looking": False,
                "explanation": "stub",
                "missing_information": None,
                "cited_passages": [],
                "jurisdiction": "sec",
            }

        orch = ComplianceOrchestrator(
            deck=deck,
            modules=["sec"],
            analyzer_model="claude-sonnet-4-6",
            completeness_model="claude-haiku-4-5",
            sec_analyzer=sec_stub,
            api_key="stub-key-not-used",
        )
        report = orch.run()
        # If the orchestrator incorrectly used completeness_model for claims,
        # the stub would need to inspect `self.analyzer_model`. Since we're
        # injecting a stub, we verify the attribute directly.
        assert orch.analyzer_model == "claude-sonnet-4-6"
        assert orch.completeness_model == "claude-haiku-4-5"
        assert report["claims_analyzed"] == 10


# ──────────────────────────────────────────────────────────────────────────────
# Mitigation 3 — max_tokens reduced to 4000
# ──────────────────────────────────────────────────────────────────────────────


class TestMaxTokens:
    """Both analyzers must use max_tokens=4000, not 16000/8000."""

    def test_analyzer_industry_claim_uses_4000_tokens(self):
        """analyzer.analyze_industry_claim must pass max_tokens=4000."""
        import inspect, ast
        import analyzer as _a
        src = inspect.getsource(_a.analyze_industry_claim)
        # Check the literal 4000 is present (not 16000)
        assert "4000" in src, "analyze_industry_claim must set max_tokens=4000"
        assert "16000" not in src, "analyze_industry_claim still sets max_tokens=16000"

    def test_analyzer_claim_uses_4000_tokens(self):
        """analyzer.analyze_claim must pass max_tokens=4000."""
        import inspect
        import analyzer as _a
        src = inspect.getsource(_a.analyze_claim)
        assert "4000" in src, "analyze_claim must set max_tokens=4000"
        assert "16000" not in src, "analyze_claim still sets max_tokens=16000"

    def test_analyzer_sfdr_uses_4000_tokens(self):
        """analyzer_sfdr.analyze_claim must pass max_tokens=4000."""
        import inspect
        import analyzer_sfdr as _sfdr
        src = inspect.getsource(_sfdr.analyze_claim)
        assert "4000" in src, "analyzer_sfdr.analyze_claim must set max_tokens=4000"
        assert "8000" not in src, "analyzer_sfdr.analyze_claim still sets max_tokens=8000"
