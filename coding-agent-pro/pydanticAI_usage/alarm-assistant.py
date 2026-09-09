# SDK Usage: ModelRetry()
import os
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from datetime import datetime

from pydantic_ai import Agent, ModelRetry

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

agent = Agent(
    model,
    instructions=(
        "You are a scheduling assistant. Convert the time provided by the "
        "user into an absolute time, then call schedule_reminder."
    ),
)


@agent.tool_plain(retries=2)  # Retry up to two times after the initial attempt.
def schedule_reminder(content: str, when: str) -> str:
    """Schedule a reminder."""
    print(f"[tool] called with when={when!r}")

    try:
        datetime.strptime(when, "%Y-%m-%d %H:%M")
    except ValueError:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        print(f"[error] {when!r} is invalid.")
        # Include both the required format and the current time in the retry
        # message so the model can convert the user's relative time into the
        # correct absolute date and time.
        raise ModelRetry(
            f"Invalid value when={when!r}. The required format is "
            f"'YYYY-MM-DD HH:MM'. The current time is {now}. Convert the "
            "relative time provided by the user into an absolute time based "
            "on the current time, then try again."
        )

    return f"Reminder scheduled: {when} {content}"

result = agent.run_sync(
    "Set a reminder for me to go running tomorrow at 8:00 PM."
)

print(result)