import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not base_url:
    raise RuntimeError(
        "Please set the API_KEY and BASE_URL environment variables first."
    )

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# --- Tool definitions ---
from tools.tools_config import tools_config

# --- Tool implementations ---
from tools.tools_mapping import tools_mapping

# --- Main agent loop ---
SYSTEM_PROMPT = """You are a programming assistant. You can read and write files
and run commands to help the user complete programming tasks.

Workflow: First understand the requirements, then write the code and run it to
verify the result. If you find an error, fix it and run the code again. Continue
until you have confirmed that the solution works correctly."""

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

print("Programming agent started. Enter a task to begin or enter q to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.strip() == "q":
        break

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Agent loop
    while True:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tools_config,
            # Enable high reasoning effort so the model can reason
            # before making each decision.
            extra_body={"reasoning_effort": "high"},
        )

        assistant_message = response.choices[0].message

        # Always add the assistant message to the conversation before
        # processing any tool calls it may contain.
        messages.append(assistant_message)

        # An assistant message may contain both text and tool calls.
        # Display any text before executing the requested tools.
        if assistant_message.content:
            print(f"AI: {assistant_message.content}\n")

        # If the model does not request any more tool calls, it is ready
        # to provide its final response, so exit the agent loop.
        if not assistant_message.tool_calls:
            break

        # The model may request multiple tools. Run them sequentially.
        for tool_call in assistant_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            function = tools_mapping[tool_call.function.name]
            result = function(**args)

            # Display tool calls so the agent's behavior can be observed.
            print(f"  [Tool] {tool_call.function.name}({args})")

            # Display only the first 200 characters to prevent large files
            # or command outputs from flooding the terminal.
            print(f"  [Result] {result[:200]}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )