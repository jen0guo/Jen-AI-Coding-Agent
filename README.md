# Jen's AI Agent Engineering Portfolio

I'm building a collection of AI agent projects, starting with a small chatbot and working toward a production-grade coding agent. This repository documents my hands-on experience with language models, tool calling, and the engineering decisions behind useful AI applications.

Each project builds on the previous one as I learn new concepts and put them into practice. The chatbot and retrieval-augmented generation (RAG) chatbot are implemented, with coding agents next on the roadmap.

## Project guide

For a quick overview, scan the table below. Start with the [RAG chatbot](rag-qa-bot/rag_chatbot.py) to see document retrieval and answer generation working together, or the [chatbot with tool calling](chatbot/chatbot_v2.py) for the foundations.

| Project | Status | Focus | Explore |
| --- | --- | --- | --- |
| Chatbot | Implemented | Conversation history, model API integration, and tool calling | [Overview](#1-chatbot) · [Code](chatbot/) |
| RAG Chatbot | Implemented | Document Q&A using Qwen embeddings, Chroma vector search, and DeepSeek | [Overview](#2-rag-chatbot) · [Code](rag-qa-bot/) |
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

**Status: Implemented.** A command-line customer support chatbot that answers product questions using eight sample documentation excerpts. It retrieves the three closest matches by meaning, then asks DeepSeek to answer from those excerpts, link to the sources, and acknowledge when the documentation lacks an answer.

**Workflow:** Question → embedding → Chroma retrieval → excerpts added to the prompt → DeepSeek response.

| Component | Technology |
| --- | --- |
| Application | Python |
| Text embeddings | Qwen3-Embedding-0.6B via Sentence Transformers; model weights downloaded through ModelScope |
| Vector storage and retrieval | Persistent Chroma database with cosine distance |
| Answer generation | DeepSeek (`deepseek-v4-flash`) through the OpenAI Python SDK |
| Initial search implementation | NumPy embedding storage and similarity ranking |
| Configuration | Environment variables loaded with python-dotenv |

**Explore the implementation:**

- [RAG chatbot](rag-qa-bot/rag_chatbot.py): Retrieves context, builds the prompt, and generates answers.
- [Chroma index builder](rag-qa-bot/build_index_chroma.py) and [search](rag-qa-bot/search_chroma.py): Store document vectors and metadata, then retrieve the top three matches.
- [NumPy index builder](rag-qa-bot/build_index.py) and [search](rag-qa-bot/search.py): Show the initial embedding and ranking workflow before adding a vector database.

**Skills demonstrated:** RAG pipeline integration, semantic search, vector databases, metadata handling, and prompting for answers with source links.

The demo uses [sample documents](rag-qa-bot/documents.py) with placeholder source URLs and handles each question independently.

## 3. Simple Coding Agent

**Status: Planned.** Build an agent that can inspect code, propose edits, and run checks on small programming tasks.

Planned learning areas include agent control loops, file and command tools, tool feedback, and clear stopping conditions.

## 4. Production-Grade Coding Agent

**Status: Planned.** Develop the simple coding agent toward production requirements, with an emphasis on reliability and measurable behavior.

Planned learning areas include controlled code execution, permissions, error recovery, automated evaluations, observability, and cost and latency tracking.
