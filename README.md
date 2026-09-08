# Jen's AI Agent Engineering Portfolio

I'm building a collection of AI agent projects, starting with a small chatbot and working toward a production-grade coding agent. This repository documents my hands-on experience with language models, tool calling, and the engineering decisions behind useful AI applications.

Each project builds on the previous one as I learn new concepts and put them into practice. The portfolio is a work in progress: the first chatbot is implemented, with retrieval-augmented generation (RAG) and coding agents on the roadmap.

## Project guide

For a quick overview, scan the table below. To review the current implementation, start with [the chatbot with tool calling](chatbot/chatbot_v2.py).

| Project | Status | Focus | Explore |
| --- | --- | --- | --- |
| Chatbot | Implemented | Conversation history, model API integration, and tool calling | [Overview](#1-chatbot) · [Code](chatbot/) |
| RAG Chatbot | Planned | Retrieving relevant documents to answer questions with supporting sources | [Roadmap](#2-rag-chatbot) |
| Simple Coding Agent | Planned | Connecting a model to development tools through an iterative task loop | [Roadmap](#3-simple-coding-agent) |
| Production-Grade Coding Agent | Planned | Building toward reliable, observable, and testable agent workflows | [Roadmap](#4-production-grade-coding-agent) |

## 1. Chatbot

My first small project explores how a language model maintains a conversation and calls Python functions to answer a user's request.

- **[Basic chatbot](chatbot/chatbot.py):** A command-line conversation loop that keeps message history during the session.
- **[Chatbot with tool calling](chatbot/chatbot_v2.py):** Adds tools for looking up sample weather data and extracting text from a supplied webpage. The model selects a tool, Python executes it, and the result is returned to the model to generate a response.

**Skills demonstrated:** Python, model API integration, conversation state, JSON tool definitions, tool dispatch, and HTML text extraction.

**Built with:** Python, the OpenAI Python SDK connected to DeepSeek, Beautiful Soup, and python-dotenv.

The weather tool uses hardcoded sample data. The webpage tool reads server-returned HTML and returns up to 5,000 characters; it does not execute JavaScript.

## 2. RAG Chatbot

**Status: Planned.** Extend the chatbot to answer questions using a collection of documents.

Planned learning areas include document ingestion, text chunking, embeddings, retrieval, source citations, and evaluating whether answers are supported by the retrieved content.

## 3. Simple Coding Agent

**Status: Planned.** Build an agent that can inspect code, propose edits, and run checks on small programming tasks.

Planned learning areas include agent control loops, file and command tools, tool feedback, and clear stopping conditions.

## 4. Production-Grade Coding Agent

**Status: Planned.** Develop the simple coding agent toward production requirements, with an emphasis on reliability and measurable behavior.

Planned learning areas include controlled code execution, permissions, error recovery, automated evaluations, observability, and cost and latency tracking.
