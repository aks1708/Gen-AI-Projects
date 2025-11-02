# Browsing Agent

## Overview

The goal of this project is to build an agent that can autonomously perform actions over the web such as visiting pages, clicking links etc. 

We leverage the Playwright MCP server to help us connect our agent to tools that allow it to perform actions over the web.
MCP (Model Context Protocol) is a standardised way of connecting LLMs to resources, prompts and tools without having to configure each LLM seperately.

We aren't using any agentic frameworks typically seen in development such as crewAI, AutoGen and LangGraph as we keep in mind that agents are just models that call tools to perform actions in a loop until they reach a terminal state. Thus this agent is implemented from scratch in Python.

## Setup

Create your virtual environment and install the dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a .env file and set up your Gemini api key.

```bash
GEMINI_API_KEY=your_key
```

## Usage

Then run the following command to start the agent:

```bash
python3 agent.py
```