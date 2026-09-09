# SDK Usage: Hooks
import os
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

# Read the API key from an environment variable to avoid hard-coding
# sensitive credentials in the source code.
load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    raise RuntimeError("Please set the API_KEY environment variable first.")


# --- Model Configuration ---
# Explicitly configure the model and provider. To switch to another model,
# replace the model name and its corresponding provider class.
model = OpenAIChatModel(
    "deepseek-v4-flash",
    provider=DeepSeekProvider(api_key=api_key),
)

hooks = Hooks()

@hooks.on.before_model_request
async def log_before(ctx, request_context):
    print(
        f"[before] About to call the model. "
        f"Context message count: {len(request_context.messages)}"
    )

    # If necessary, you can modify the context by updating the messages
    # array directly before returning the request context.
    return request_context


@hooks.on.after_model_request
async def log_after(ctx, request_context, response):
    usage = response.usage

    print(
        f"[after] input_tokens={usage.input_tokens} "
        f"output_tokens={usage.output_tokens}"
    )

    return response


agent = Agent(
    model,
    capabilities=[hooks],
)

result = agent.run_sync(
    "Describe the Great Wall of China in one sentence."
)

print(result)

# [before] About to call the model. Context message count: 1
# [after] input_tokens=9 output_tokens=28