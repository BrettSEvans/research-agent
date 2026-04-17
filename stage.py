import anthropic
import os
from enum import Enum
from pydantic import BaseModel, Field

class FundingStage(str, Enum):
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C_PLUS = "series_c_plus"

class StageAssessment(BaseModel):
    stage: FundingStage
    confidence: float = Field(description="Confidence from 0.0 to 1.0")
    signals: list[str] = Field(description="Key indicators found in the text, e.g. 'ARR $2M mentioned', 'Raising $8M Series A'")
    user_should_confirm: bool = Field(description="True if confidence < 0.7")

SYSTEM_PROMPT = """You are an expert venture capital analyst. Your task is to determine the funding stage of a startup based on the raw text of their pitch deck.
Look for these signals:
- ARR/revenue amount: seed <$1M, A $1-10M, B $10-30M, C+ $30M+
- Raise amount: seed <$3M, A $5-20M, B $20-60M
- Prior rounds mentioned
- Keywords: "MVP", "beta", "launching" -> pre_seed/seed
- Keywords: "repeatable GTM", "sales team" -> A/B
- Keywords: "path to profitability", "EBITDA" -> late

Provide your assessment, confidence score (0.0 to 1.0), the exact signals you found, and whether the user should manually confirm (set to True if confidence < 0.7).
"""

def infer_stage_from_text(text: str, client: anthropic.Anthropic | None = None, model: str | None = None) -> StageAssessment:
    client = client or anthropic.Anthropic()
    model = model or os.environ.get("EXTRACTOR_MODEL", "claude-haiku-4-5")
    
    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Here is the pitch deck text:\n\n{text[:40000]}\n\nAnalyze the stage."
            }
        ],
        output_format=StageAssessment,
    )
    return response.parsed_output
