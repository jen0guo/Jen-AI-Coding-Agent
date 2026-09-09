# SDK Usage: deps_type
import os
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

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


# Use a simple in-memory dictionary to simulate a notes database.
# Replace it with a real database connection in a production application.
class NotesDB:
    _store = {
        "alice": [{"title": "Introduction to Redis"}],
        "bob": [{"title": "How MySQL Indexes Work"}],
    }

    def search(self, user_id: str, keyword: str) -> list[dict]:
        return [
            note
            for note in self._store.get(user_id, [])
            if keyword in note["title"]
        ]


@dataclass
class AppContext:
    user_id: str  # The user associated with the current request
    db: NotesDB  # A reusable database connection pool


agent = Agent(
    model,
    deps_type=AppContext,
    instructions=(
        "You are the user's personal notes assistant. Before answering, "
        "use search_notes to search the user's notes."
    ),
)


# Change the decorator from @agent.tool_plain to @agent.tool and declare
# RunContext as the first parameter.
@agent.tool
def search_notes(
    ctx: RunContext[AppContext],
    keyword: str,
) -> list[dict]:
    """Search for a keyword in the current user's notes."""
    # The tool automatically obtains the user_id for access isolation,
    # equivalent to applying WHERE user_id = ? in SQL.
    return ctx.deps.db.search(ctx.deps.user_id, keyword)


# The same Agent instance serves two different users. The supplied dependencies
# determine which user's data the agent can access.
db = NotesDB()

result_alice = agent.run_sync(
    "Have I previously written any notes about Redis?",
    deps=AppContext(user_id="alice", db=db),
)

result_bob = agent.run_sync(
    "Have I previously written any notes about Redis?",
    deps=AppContext(user_id="bob", db=db),
)

print(result_alice)
print(result_bob)