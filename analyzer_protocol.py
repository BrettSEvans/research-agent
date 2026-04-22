"""
Standard protocol for compliance analyzer modules.

All analyzer modules (SEC, SFDR/CSRD, SB54) implement this interface to ensure
consistent claim assessment, reporting, and integration with the verification pipeline.
"""

from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field


class ClaimAssessment(BaseModel):
    """
    Standardized compliance assessment for a single claim extracted from a pitch deck.

    Used by all analyzer modules (SEC, EU SFDR/CSRD, CA SB54) to provide:
    - A verdict (Compliant / Warning / Non-Compliant)
    - Severity level (Info / Medium / Critical)
    - Explanation with cited passages
    - Actionable next steps
    """

    verdict: str = Field(
        ..., description="Compliant | Warning | Non-Compliant"
    )
    severity: str = Field(
        ..., description="Info | Medium | Critical"
    )
    forward_looking: bool = Field(
        ..., description="True if the claim is forward-looking (e.g., a projection or target)"
    )
    explanation: str = Field(
        ..., description="Clear explanation of the assessment"
    )
    cited_passages: list[int] = Field(
        default_factory=list,
        description="Indices of KB passages used to justify the assessment",
    )
    missing_information: str | None = Field(
        default=None,
        description="What additional data is needed for a complete assessment",
    )
    jurisdiction: str = Field(
        ..., description="sec | eu_sfdr_csrd | ca_sb54"
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Critical compliance issues requiring immediate attention",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-critical issues requiring clarification",
    )
    verified: list[str] = Field(
        default_factory=list,
        description="Confirmed compliant data points",
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Specific questions or remediation steps for compliance officer",
    )


@runtime_checkable
class AnalyzerModule(Protocol):
    """
    Protocol for compliance analyzer modules.

    All analyzer implementations must provide:
    - An `assess()` method for evaluating a single claim
    - Optional initialization with a KB retriever
    """

    def assess(
        self, claim: str, deck_context: dict, retriever=None
    ) -> ClaimAssessment:
        """
        Assess a single claim from a pitch deck against compliance regulations.

        Args:
            claim: The claim text to assess
            deck_context: Contextual data from the deck extraction (founder info, metrics, etc.)
            retriever: Optional KB retriever for regulatory passages

        Returns:
            ClaimAssessment with verdict, severity, explanation, and action items
        """
        ...
