import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not base_url:
    raise RuntimeError("Please set up API_KEY and BASE_URL.")

client = OpenAI(api_key=api_key, base_url=base_url)

SYSTEM_PROMPT = "Your answer should be concise. Please keep your answer no longer than 3 sentences."

# Context, all the messages are added to this list
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print(f"[Role Description] {SYSTEM_PROMPT}")
print("Type in a message to start, type in q to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip() == "q":
        break

    # Add user's messages to context
    messages.append({"role": "user", "content": user_input})

    # Send the complete context to model
    # Thinking:disabled, not display reasoning text
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}}
    )

    # Extract answer from model and add it to context
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}\n")