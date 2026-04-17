import sys
import os
from dotenv import load_dotenv

load_dotenv("/Users/brettevanssf/Code/Saasless/VC/compliance-agent/.env")
sys.path.append("/Users/brettevanssf/Code/Saasless/VC/compliance-agent")
from extractor import _resolve_model, SYSTEM_PROMPT, DeckExtraction
import anthropic

client = anthropic.Anthropic()
model = _resolve_model()

print(f"Testing {model}...")
response = client.messages.parse(
    model=model,
    max_tokens=16000,
    system=SYSTEM_PROMPT,
    messages=[
        {
            "role": "user",
            "content": "Extract the structured pitch deck information. Here is the deck text: We are a Series A startup with $5M ARR. Our name is TestCo. We are raising $10M. Our CAC is $100."
        }
    ],
    output_format=DeckExtraction,
)
print("Success!")
print(response.parsed_output)
