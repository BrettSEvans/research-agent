"""Industry Benchmarking and Peer Comparison

Compares company compliance against industry peers and identifies outliers.
Provides percentile ranking and best practice recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class IndustrySegment(Enum):
    """Industry classification for benchmarking."""
    SAAS = "saas"
    FINTECH = "fintech"
    CLEANTECH = "cleantech"
    BIOTECH = "biotech"
    HEALTHTECH = "healthtech"
    OTHER = "other"


@dataclass
class PeerBenchmark:
    """Peer company benchmark data."""
    company_id: str
    industry: IndustrySegment
    jurisdiction: str
    compliance_score: float
    flagged_issues: int
    critical_issues: int
    time_in_current_stage: int  # days


@dataclass
class BenchmarkResult:
    """Benchmarking analysis result."""
    jurisdiction: str
    company_score: float
    industry: IndustrySegment
    percentile_rank: float  # 0-100
    percentile_label: str  # "Top 10%", "Median", etc.
    peers_better: int
    peers_worse: int
    median_peer_score: float
    best_peer_score: float
    worst_peer_score: float
    outlier_status: str  # "outlier_high", "outlier_low", "average"
    recommendation: str


def benchmark_compliance(
    company_score: float,
    jurisdiction: str,
    industry: IndustrySegment,
    peer_data: List[PeerBenchmark],
) -> BenchmarkResult:
    """Compare company compliance against industry peers.

    Args:
        company_score: Company's compliance score (0-100)
        jurisdiction: Target jurisdiction
        industry: Company's industry segment
        peer_data: List of peer benchmark data

    Returns:
        BenchmarkResult with percentile ranking and recommendations
    """
    # Filter peers in same jurisdiction and industry
    relevant_peers = [
        p for p in peer_data
        if p.jurisdiction == jurisdiction and p.industry == industry
    ]

    if not relevant_peers:
        # No peer data - return neutral benchmark
        return BenchmarkResult(
            jurisdiction=jurisdiction,
            company_score=company_score,
            industry=industry,
            percentile_rank=50.0,
            percentile_label="Insufficient peer data",
            peers_better=0,
            peers_worse=0,
            median_peer_score=company_score,
            best_peer_score=company_score,
            worst_peer_score=company_score,
            outlier_status="unknown",
            recommendation="Monitor compliance against future peer benchmarks.",
        )

    # Calculate peer statistics
    peer_scores = [p.compliance_score for p in relevant_peers]
    peer_scores.sort()

    peers_better = sum(1 for s in peer_scores if s > company_score)
    peers_worse = sum(1 for s in peer_scores if s < company_score)
    total_peers = len(relevant_peers)

    percentile_rank = (peers_worse / total_peers * 100) if total_peers > 0 else 50.0

    # Determine percentile label
    if percentile_rank >= 90:
        percentile_label = "Top 10% (Exceptional)"
    elif percentile_rank >= 75:
        percentile_label = "Top Quartile (Strong)"
    elif percentile_rank >= 50:
        percentile_label = "Above Median"
    elif percentile_rank >= 25:
        percentile_label = "Below Median"
    else:
        percentile_label = "Bottom Quartile (At Risk)"

    # Detect outliers (>1.5 std dev from mean)
    mean_score = sum(peer_scores) / len(peer_scores)
    variance = sum((s - mean_score) ** 2 for s in peer_scores) / len(peer_scores)
    std_dev = variance ** 0.5

    if company_score > mean_score + 1.5 * std_dev:
        outlier_status = "outlier_high"
    elif company_score < mean_score - 1.5 * std_dev:
        outlier_status = "outlier_low"
    else:
        outlier_status = "average"

    # Generate recommendation
    median_peer_score = peer_scores[len(peer_scores) // 2]
    recommendation = _generate_benchmark_recommendation(
        company_score, median_peer_score, outlier_status, percentile_rank
    )

    return BenchmarkResult(
        jurisdiction=jurisdiction,
        company_score=company_score,
        industry=industry,
        percentile_rank=round(percentile_rank, 1),
        percentile_label=percentile_label,
        peers_better=peers_better,
        peers_worse=peers_worse,
        median_peer_score=round(median_peer_score, 1),
        best_peer_score=round(peer_scores[-1], 1),
        worst_peer_score=round(peer_scores[0], 1),
        outlier_status=outlier_status,
        recommendation=recommendation,
    )


def _generate_benchmark_recommendation(
    company_score: float,
    median_score: float,
    outlier_status: str,
    percentile: float,
) -> str:
    """Generate benchmark-based recommendation."""
    if outlier_status == "outlier_high":
        return f"✓ EXEMPLARY: Score of {company_score:.0f} far exceeds peers ({median_score:.0f} median). Share best practices."

    if outlier_status == "outlier_low":
        return f"⚠️  LAGGING: Score of {company_score:.0f} significantly below median ({median_score:.0f}). Urgent remediation needed."

    score_gap = median_score - company_score

    if score_gap > 15:
        return f"⚠️  BELOW PEER MEDIAN: Gap of {score_gap:.0f} points. Review peer best practices for {percentile:.0f}th percentile improvement."

    if score_gap < -5:
        return f"✓ ABOVE PEER MEDIAN: Score of {company_score:.0f} outperforms {percentile:.0f}% of peers. Maintain current practices."

    return f"ℹ️  IN LINE WITH PEERS: Comparable to median score of {median_score:.0f}. Benchmark against top quartile for further improvement."


def identify_best_practices(
    benchmark: BenchmarkResult,
    peer_data: List[PeerBenchmark],
) -> Dict:
    """Identify best practices from top-performing peers.

    Args:
        benchmark: Company's benchmark result
        peer_data: All peer data

    Returns:
        Best practices identified from top performers
    """
    # Find top 25% performers in same jurisdiction/industry
    relevant_peers = [
        p for p in peer_data
        if p.jurisdiction == benchmark.jurisdiction
        and p.industry == benchmark.industry
    ]

    if not relevant_peers:
        return {
            "best_practice_count": 0,
            "practices": [],
            "recommendation": "No peer data available.",
        }

    relevant_peers.sort(key=lambda p: p.compliance_score, reverse=True)
    top_peers = relevant_peers[:max(1, len(relevant_peers) // 4)]

    practices = []
    for peer in top_peers:
        practices.append({
            "peer": peer.company_id,
            "score": peer.compliance_score,
            "critical_issues": peer.critical_issues,
            "time_in_stage": peer.time_in_current_stage,
        })

    return {
        "best_practice_count": len(practices),
        "top_performers": practices,
        "average_top_score": round(sum(p.compliance_score for p in top_peers) / len(top_peers), 1),
        "recommendation": f"Study practices from top {len(practices)} performers averaging {sum(p.compliance_score for p in top_peers) / len(top_peers):.0f}.",
    }


def stage_adjusted_benchmark(
    company_score: float,
    funding_stage: str,
    jurisdiction: str,
    peer_data: List[PeerBenchmark],
) -> Dict:
    """Benchmark adjusted for funding stage (earlier stages have lower expectations).

    Args:
        company_score: Company's compliance score
        funding_stage: Funding stage (pre_seed, seed, series_a, series_b, series_c_plus)
        jurisdiction: Target jurisdiction
        peer_data: All peer benchmark data

    Returns:
        Stage-adjusted benchmark with adjusted expectations
    """
    # Stage-adjusted score expectations (what's "good" at each stage)
    stage_expectations = {
        "pre_seed": 40,
        "seed": 50,
        "series_a": 65,
        "series_b": 75,
        "series_c_plus": 85,
    }

    expected_score = stage_expectations.get(funding_stage, 60)

    # Filter peers at same stage
    # (In real implementation, peer_data would include stage info)
    relevant_peers = [p for p in peer_data if p.jurisdiction == jurisdiction]

    if relevant_peers:
        peer_scores = [p.compliance_score for p in relevant_peers]
        median_at_stage = sum(peer_scores) / len(peer_scores)
    else:
        median_at_stage = expected_score

    # Calculate adjusted percentile (relative to stage expectations)
    stage_gap = company_score - expected_score
    peer_gap = median_at_stage - expected_score

    adjusted_performance = "on-track" if stage_gap >= -5 else "behind"

    return {
        "funding_stage": funding_stage,
        "expected_score": expected_score,
        "actual_score": company_score,
        "stage_gap": round(stage_gap, 1),
        "peer_median": round(median_at_stage, 1),
        "peer_gap": round(peer_gap, 1),
        "adjusted_performance": adjusted_performance,
        "recommendation": (
            f"✓ On track for {funding_stage} stage (target {expected_score})."
            if adjusted_performance == "on-track"
            else f"⚠️  Behind expectations for {funding_stage}. Accelerate remediation before next funding round."
        ),
    }
