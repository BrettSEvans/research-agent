import anyio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, SystemMessage


async def research(topic: str) -> str:
    """Run a research query using the Claude Agent SDK with web search."""
    session_id = None
    result = None

    async for message in query(
        prompt=f"Research the following topic thoroughly and provide a comprehensive summary with key findings, recent developments, and important sources: {topic}",
        options=ClaudeAgentOptions(
            allowed_tools=["WebSearch", "WebFetch"],
            system_prompt=(
                "You are an expert research assistant. When given a topic, you search "
                "the web for current and relevant information, synthesize the findings, "
                "and produce a clear, well-structured summary. Always cite your sources."
            ),
            max_turns=20,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            session_id = message.data.get("session_id")
            print(f"Session: {session_id}")
        elif isinstance(message, ResultMessage):
            result = message.result

    return result or "No result returned."


async def main():
    import sys

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI agents in 2025"
    print(f"\nResearching: {topic}\n{'=' * 60}\n")
    result = await research(topic)
    print(result)


if __name__ == "__main__":
    anyio.run(main)
