import os
import subprocess
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

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

# OpenAI Model:
# from pydantic_ai.providers.openai import OpenAIProvider
# model = OpenAIChatModel('gpt-5.5',
#                        provider=OpenAIProvider(api_key=API_KEY))

# Anthropic Model:
# from pydantic_ai.models.anthropic import AnthropicModel
# from pydantic_ai.providers.anthropic import AnthropicProvider
# model = AnthropicModel('claude-sonnet-4-6',
#                       provider=AnthropicProvider(api_key=API_KEY))


agent = Agent(
    model,
    # These instructions serve as the system prompt from the manual
    # implementation. The SDK automatically places them at the beginning
    # of the model context.
    instructions=(
        "You are a programming assistant. You can read and write files "
        "and run commands to help the user complete programming tasks.\n"
        "Workflow: First understand the requirements, then write the code "
        "and run it to verify the result. If you find an error, fix it and "
        "run the code again. Continue until you have confirmed that the "
        "solution works correctly."
    )
)


# --- Tool definitions ---
@agent.tool_plain
def read_file(path: str) -> str:
    """Read the contents of a specified file.

    Args:
        path: The path of the file to read.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File {path} does not exist."


@agent.tool_plain
def write_file(path: str, content: str) -> str:
    """Write content to a specified file.

    Args:
        path: The path of the file to write.
        content: The content to write to the file.
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

    return f"Successfully wrote to {path}."


@agent.tool_plain
def run_command(command: str) -> str:
    """Run a shell command and return its output.

    Args:
        command: The shell command to run.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )

        output = result.stdout

        if result.returncode != 0:
            output += f"\n[Error] {result.stderr}"

        return output or "(No output)"

    except subprocess.TimeoutExpired:
        return "[Error] Command timed out after 10 seconds."


# --- Start the agent ---
# Preserve message history across turns so the agent remembers its
# previous actions and conversations.
history = []

print("Programming agent started. Enter a task to begin or enter q to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.strip() == "q":
        break

    # run_sync manages the internal agent loop: it calls the model,
    # executes requested tools, returns the results to the model, and
    # continues until the model produces a final response.
    result = agent.run_sync(
        user_input,
        message_history=history,
    )
    
    # Save the complete message history, including messages added during
    # this turn, so they remain available to the model on the next turn.
    history = result.all_messages()

    print(f"AI: {result.output}\n")