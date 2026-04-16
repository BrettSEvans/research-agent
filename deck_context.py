"""Deck context — the handoff artifact between the extractor and the compliance agent.

The compliance agent uses this strictly as a *clarifying* information source:
- to identify the company (name → CIK resolution),
- to look up fiscal context (period, currency),
- to read the extractor's transparency notes.

It does NOT use the deck to verify claims. SEC filings are the sole source of
truth for verification. This separation is enforced in the analyzer prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

from extractor import DeckExtraction


class DeckContext:
    def __init__(self, extraction: DeckExtraction):
        self.extraction = extraction

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, path: str | Path) -> "DeckContext":
        data = json.loads(Path(path).read_text())
        return cls(DeckExtraction.model_validate(data))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.extraction.model_dump_json(indent=2))

    # --------------------------------------------------------- consumption

    def claims_for_verification(self) -> list[str]:
        """The list of claims the compliance agent should verify."""
        return [c.text for c in self.extraction.claims]

    def company_lookup_key(self) -> str | None:
        """Best-effort identifier for resolving the company to an SEC CIK.

        Preference order: explicit CIK → ticker → name.
        """
        c = self.extraction.company
        return c.cik or c.ticker or c.name or None

    def clarifying_context(self) -> str:
        """Formatted block the analyzer may consult for clarifying details.

        This is explicitly NOT the source of truth. The analyzer verifies against
        SEC filings. The deck context helps disambiguate fiscal periods, company
        identity, and similar metadata.
        """
        c = self.extraction.company
        lines = [
            "[DECK CONTEXT — clarifying metadata only, NOT a source of truth]",
            f"Company: {c.name}",
        ]
        if c.ticker:
            lines.append(f"Ticker: {c.ticker}")
        if c.cik:
            lines.append(f"CIK: {c.cik}")
        if c.description:
            lines.append(f"Description: {c.description}")
        if self.extraction.fiscal_year_end:
            lines.append(f"Fiscal year end: {self.extraction.fiscal_year_end}")
        if self.extraction.currency:
            lines.append(f"Currency: {self.extraction.currency}")
        lines.append("")
        lines.append("Extractor's transparency notes:")
        lines.append(self.extraction.extraction_notes)
        return "\n".join(lines)
