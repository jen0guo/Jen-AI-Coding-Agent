# OpenAI Agents SDK

import asyncio
import os
import sys

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

API_KEY = os.environ["API_KEY"]

# Disable the default OpenAI tracing.
set_tracing_disabled(True)

# Use the DeepSeek model for all agents through LiteLLM.
model = LitellmModel(
    model="deepseek/deepseek-chat",
    api_key=API_KEY,
)

# Billing specialist
billing_agent = Agent(
    name="Billing",
    handoff_description=(
        "Handles questions related to charges, refunds, and subscriptions."
    ),
    instructions=(
        "You are a billing specialist. Only answer questions related to "
        "payments, refunds, and subscriptions. Keep your answers brief "
        "and specific."
    ),
    model=model,
)

# Technical support specialist
technical_agent = Agent(
    name="Technical",
    handoff_description=(
        "Handles app malfunctions, login issues, and bug reports."
    ),
    instructions=(
        "You are a technical support specialist. Only answer questions "
        "about using the app and troubleshooting bugs. Provide no more "
        "than three troubleshooting suggestions, using one sentence for each."
    ),
    model=model,
)

# Triage entry point
triage_agent = Agent(
    name="Triage",
    instructions=(
        "You are a customer-support coordinator. Route each user request "
        "to the appropriate specialist based on the type of question. "
        "Send billing-related questions to Billing and technical questions "
        "to Technical."
    ),
    handoffs=[
        billing_agent,
        technical_agent,
    ],
    model=model,
)


async def main():
    # Read the user's question from the command-line arguments.
    question = sys.argv[1]
    print(f"User: {question}")

    result = await Runner.run(
        triage_agent,
        question,
    )

    # last_agent identifies the agent that ultimately handled the conversation.
    print(f"Handled by: {result.last_agent.name}")
    print(f"Response: {result.final_output}")


asyncio.run(main())