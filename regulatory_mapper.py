"""Regulatory Citation Tracker and Mapper

Maps violations to specific regulatory articles, builds citation index,
and identifies most critical regulatory gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set
from analyzer_protocol import ClaimAssessment


@dataclass
class RegulatoryViolation:
    """Individual regulatory violation."""
    jurisdiction: str
    article_or_section: str
    description: str
    severity: str
    claim_count: int
    recommended_action: str


@dataclass
class RegulatoryMap:
    """Complete regulatory compliance map."""
    jurisdiction: str
    total_violations: int
    critical_violations: int
    articles_cited: List[str]
    violations_by_severity: Dict[str, int]
    most_critical_articles: List[str]
    required_remediations: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "jurisdiction": self.jurisdiction,
            "total_violations": self.total_violations,
            "critical_violations": self.critical_violations,
            "articles_cited": self.articles_cited,
            "violations_by_severity": self.violations_by_severity,
            "most_critical_articles": self.most_critical_articles,
            "required_remediations": self.required_remediations,
        }


def build_regulatory_map(
    jurisdiction: str,
    assessments: List[ClaimAssessment],
) -> RegulatoryMap:
    """Build comprehensive regulatory compliance map from assessments.

    Args:
        jurisdiction: Jurisdiction identifier
        assessments: List of claim assessments

    Returns:
        RegulatoryMap with violation tracking and remediation guidance
    """
    violations_by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    articles_cited: Set[str] = set()
    critical_violations = 0
    total_violations = 0

    for assessment in assessments:
        # Count violations by severity
        severity = assessment.severity or "NONE"
        if severity in violations_by_severity:
            violations_by_severity[severity] += 1

        # Track critical violations
        if assessment.verdict in ["CONTRADICTS", "CRITICAL_ABSENT"] and severity == "HIGH":
            critical_violations += 1
            total_violations += 1

        # Extract cited articles/sections from explanation
        articles = _extract_articles(assessment.explanation)
        articles_cited.update(articles)

    # Determine most critical articles
    most_critical = _rank_articles(jurisdiction, list(articles_cited))

    # Generate required remediations
    required_remediations = _generate_remediations(jurisdiction, violations_by_severity)

    return RegulatoryMap(
        jurisdiction=jurisdiction,
        total_violations=total_violations,
        critical_violations=critical_violations,
        articles_cited=sorted(list(articles_cited)),
        violations_by_severity=violations_by_severity,
        most_critical_articles=most_critical[:5],  # Top 5
        required_remediations=required_remediations,
    )


def _extract_articles(text: str) -> Set[str]:
    """Extract regulatory article references from text.

    Looks for patterns like:
    - "SFDR Art. 4"
    - "CSRD Article 5"
    - "EU 2019/2088"
    - "SB 54"
    """
    articles = set()

    # EU patterns
    eu_patterns = [
        r"SFDR\s+(?:Art|Article)\.?\s*\d+",
        r"CSRD\s+(?:Art|Article)\.?\s*\d+",
        r"EU\s+\d{4}/\d{4}",
        r"ESRS\s+\w+[-\d]+",
    ]

    # CA patterns
    ca_patterns = [
        r"(?:CA\s+)?SB\s+\d+",
        r"(?:CA\s+)?AB\s+\d+",
        r"DFEH\s+\w+",
    ]

    import re
    for pattern in eu_patterns + ca_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        articles.update(matches)

    return articles


def _rank_articles(jurisdiction: str, articles: List[str]) -> List[str]:
    """Rank articles by criticality within jurisdiction.

    Args:
        jurisdiction: Jurisdiction identifier
        articles: List of article citations

    Returns:
        Ranked list of most critical articles
    """
    # Define criticality weights per jurisdiction
    criticality_weights = {
        "eu_sfdr_csrd": {
            "SFDR Art. 4": 10,  # Principal Adverse Impacts disclosure
            "SFDR Art. 8": 8,   # Article 8 fund requirements
            "CSRD Art. 5": 9,   # Sustainability reporting
            "EU 2022/2464": 9,  # CSRD Directive
            "ESRS": 7,          # Technical standards
        },
        "ca_sb54": {
            "SB 54": 10,        # Nonprofit integrity
            "SB 164": 8,        # Board diversity
            "DFEH": 9,          # Fair employment
        },
        "sec": {
            "Item 1A": 9,       # Risk factors
            "Item 7": 8,        # MD&A
            "Item 8": 7,        # Financial statements
        }
    }

    weights = criticality_weights.get(jurisdiction, {})
    ranked = sorted(
        articles,
        key=lambda a: weights.get(a, 5),
        reverse=True
    )
    return ranked


def _generate_remediations(jurisdiction: str, violations_by_severity: Dict[str, int]) -> List[str]:
    """Generate required remediation actions.

    Args:
        jurisdiction: Jurisdiction identifier
        violations_by_severity: Count of violations by severity

    Returns:
        List of required remediation actions
    """
    remediations = []

    high_violations = violations_by_severity.get("HIGH", 0)
    medium_violations = violations_by_severity.get("MEDIUM", 0)

    if jurisdiction == "eu_sfdr_csrd":
        if high_violations > 0:
            remediations.append("Obtain third-party ESG audit per CSRD §5(1)")
            remediations.append("Document Scope 1, 2, 3 emissions with methodology")
            remediations.append("Disclose board diversity metrics per ESRS")
        if medium_violations > 0:
            remediations.append("Review SFDR Article 8/9 classification; ensure PAI data available")
            remediations.append("Conduct EU AI Act risk assessment if AI in regulated sectors")

    elif jurisdiction == "ca_sb54":
        if high_violations > 0:
            remediations.append("Compile founder demographic disclosure (gender, race/ethnicity)")
            remediations.append("Document educational background and prior startup experience")
        if medium_violations > 0:
            remediations.append("Review board composition against SB 164 requirements")

    elif jurisdiction == "sec":
        if high_violations > 0:
            remediations.append("File updated 10-Q or 8-K with corrected information")
            remediations.append("Engage SEC counsel to determine disclosure obligations")
        if medium_violations > 0:
            remediations.append("Update MD&A in next quarterly or annual filing")

    if not remediations:
        remediations.append("Continue monitoring regulatory updates and maintain compliance posture.")

    return remediations


def aggregate_regulatory_maps(maps: List[RegulatoryMap]) -> dict:
    """Aggregate regulatory maps across all jurisdictions.

    Args:
        maps: List of RegulatoryMap objects

    Returns:
        Aggregated regulatory compliance overview
    """
    if not maps:
        return {
            "total_violations_all_jurisdictions": 0,
            "jurisdictions": [],
            "all_cited_articles": [],
            "critical_action_items": [],
        }

    all_articles = set()
    total_violations = 0
    all_remediations = []

    for rmap in maps:
        all_articles.update(rmap.articles_cited)
        total_violations += rmap.total_violations
        all_remediations.extend(rmap.required_remediations)

    # Deduplicate and prioritize remediations
    unique_remediations = list(set(all_remediations))
    # Sort by critical terms in first position
    critical_terms = ["critical", "must", "required", "immediately"]
    unique_remediations.sort(
        key=lambda x: any(term in x.lower() for term in critical_terms),
        reverse=True
    )

    return {
        "total_violations_all_jurisdictions": total_violations,
        "jurisdictions": [m.to_dict() for m in maps],
        "all_cited_articles": sorted(list(all_articles)),
        "critical_action_items": unique_remediations[:10],  # Top 10
    }
