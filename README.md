# Research Agent

An AI-powered research agent built with the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python). Give it a topic and it searches the web, synthesizes findings, and returns a cited summary.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Usage

```bash
python agent.py "venture capital trends 2025"
python agent.py "latest AI safety research"
```

## How it works

Uses `claude-agent-sdk` with `WebSearch` and `WebFetch` tools so Claude can autonomously search and retrieve current information before synthesizing a response.
