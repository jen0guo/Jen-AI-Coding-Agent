# Jen's AI Agent Engineering Portfolio

I'm building a collection of AI agent projects, starting with a small chatbot and working toward a production-grade coding agent. This repository documents my hands-on experience with language models, tool calling, and the engineering decisions behind useful AI applications.

Each project builds on the previous one as I learn new concepts and put them into practice. The chatbot, retrieval-augmented generation (RAG) chatbot, and simple coding agent are implemented. The production-grade coding agent is in progress; work so far includes a PydanticAI prototype and four SDK examples.

## Project guide

For a quick overview, scan the table below. Compare the [simple coding agent](coding-agent/agent.py) with the [PydanticAI coding agent](coding-agent-pro/agent.py) to see the progression from a manual tool-calling loop to SDK orchestration. Explore the [RAG chatbot](rag-qa-bot/rag_chatbot.py) for document retrieval and answer generation, or the [chatbot with tool calling](chatbot/chatbot_v2.py) for the foundations.

| Project | Status | Focus | Explore |
| --- | --- | --- | --- |
| Chatbot | Implemented | Conversation history, model API integration, and tool calling | [Overview](#1-chatbot) · [Code](chatbot/) |
| RAG Chatbot | Implemented | Document Q&A using Qwen embeddings, Chroma vector search, and DeepSeek | [Overview](#2-rag-chatbot) · [Code](rag-qa-bot/) |
| Simple Coding Agent | Implemented | File editing and command execution through an iterative DeepSeek tool-calling loop | [Overview](#3-simple-coding-agent) · [Code](coding-agent/) |
| Production-Grade Coding Agent | In progress | PydanticAI orchestration, typed tools, and examples of structured output, retries, dependencies, and hooks | [Overview](#4-production-grade-coding-agent) · [Code](coding-agent-pro/) |

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

**Status: Implemented.** A command-line programming assistant that uses DeepSeek to read and write local files and run shell commands. It returns tool output to the model so it can inspect results, revise code, and continue working on a task. Conversation history is retained during the session.

**Workflow:** Programming task → model selects tools → Python executes tools → results returned to the model → repeat until the model responds without tool calls.

| Component | Technology |
| --- | --- |
| Application and agent loop | Python with conversation history and sequential tool execution |
| Language model | DeepSeek (`deepseek-v4-flash`) through the OpenAI Python SDK |
| Tool interface | JSON function schemas and a Python function registry |
| File access and command execution | Python file I/O and `subprocess.run` with a 10-second command timeout |
| Webpage text extraction | `urllib.request` and Beautiful Soup |
| Configuration | Environment variables loaded with python-dotenv |

**Explore the implementation:**

- [Agent loop](coding-agent/agent.py): Sends conversation history to the model, dispatches tool calls, and returns results for the next iteration.
- [Tool implementations](coding-agent/tools/tools.py): Read and write files, execute shell commands, extract webpage text, and look up sample weather data.
- [Tool definitions](coding-agent/tools/tools_config.py) and [function registry](coding-agent/tools/tools_mapping.py): Describe available tools and map model requests to Python functions.

**Skills demonstrated:** Agent control loops, function calling, tool dispatch, conversation state, and using execution feedback to guide code changes.

Commands run directly on the local machine. The agent prints tool activity and output previews; its loop has no fixed iteration limit.

## 4. Production-Grade Coding Agent

**Status: In progress.** Building a coding assistant toward production requirements using PydanticAI and DeepSeek. The current command-line prototype reads and writes local files, runs commands, and retains conversation history. PydanticAI manages the model/tool loop, while four standalone examples explore SDK features useful for building more reliable applications.

**Workflow:** Programming task → PydanticAI agent → typed Python tools → results returned to the model → final response and updated conversation history.

| Component | Technology |
| --- | --- |
| Application and orchestration | Python and PydanticAI (`Agent`, `run_sync`, and message history) |
| Language model | DeepSeek (`deepseek-v4-flash`) through PydanticAI's `OpenAIChatModel` and `DeepSeekProvider` |
| Tool registration | `@agent.tool_plain`, Python type annotations, and docstrings |
| File access and command execution | Python file I/O and `subprocess.run` with a 10-second command timeout |
| SDK examples | Pydantic `BaseModel`, `ModelRetry`, `RunContext`, dataclasses, and `Hooks` |
| Configuration | API key loaded from environment variables with python-dotenv |

**Explore the implementation:**

- [Coding agent](coding-agent-pro/agent.py): Registers file and command tools and delegates the execution loop to PydanticAI.
- [Structured output](coding-agent-pro/sdk_usage_cases/getCityLocation.py): Requests a typed `CityLocation` result with city and country fields.
- [Retry feedback](coding-agent-pro/sdk_usage_cases/alarm-assistant.py): Checks a reminder's date/time format and uses `ModelRetry` to request corrected tool arguments, with up to two retries.
- [Dependency injection](coding-agent-pro/sdk_usage_cases/note-assistant.py): Passes a user ID and an in-memory notes store through `RunContext` so the tool searches that user's notes.
- [Request hooks](coding-agent-pro/sdk_usage_cases/useHooks.py): Logs context message counts before model calls and input/output token usage afterward.

**Skills demonstrated:** SDK-based agent orchestration, typed tool interfaces, structured model output, tool validation and retries, dependency injection, and request instrumentation.

The SDK examples are separate from the coding agent. The reminder example validates arguments and returns a confirmation without scheduling a real notification; the notes example uses sample in-memory data. Commands run directly on the local machine. Sandboxing, permissions, automated evaluations, and deployment remain future work.
