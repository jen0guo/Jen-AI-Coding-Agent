import json
import os
import urllib.request

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not base_url:
    raise RuntimeError("Please set up API_KEY and BASE_URL.")

client = OpenAI(api_key=api_key, base_url=base_url)

# Tools in JSON format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Check the current weather for a specified city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "city name"
                    }
                },
                "required": ["city"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Look up content on the page of the given page url",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "page url"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

# Tool implementation (call weather API in production)
def get_weather(city):
    weather_data = {
        "Beijing": {"temperature": 8, "condition": "Cloudy"},
        "Shanghai": {"temperature": 15, "condition": "Sunny"},
        "Seattle": {"temperature": 22, "condition": "Rainy"},
    }
    data = weather_data.get(city, {"temperature": "unknown", "condition": "unknown"})
    return json.dumps(data, ensure_ascii=False)

def get_page_content(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        encoding = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(encoding, errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(
        ["script", "style", "nav", "header", "footer", "iframe", "noscript"]
    ):
        element.decompose()

    return soup.get_text(separator="\n", strip=True)[:5000]   


# Mapping from tool names to functions
tool_functions = {"get_weather": get_weather, "get_page_content": get_page_content}

SYSTEM_PROMPT = "You are excellent assistant for looking up weather. Please remember what information user told you."
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print(f"[Role Description] {SYSTEM_PROMPT}")
print("Type in a message to start, type in q to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip() == "q":
        break

    messages.append({"role": "user", "content": user_input})

    # Pass the list of tools to the API, and the model will determine for itself whether a call is necessary.
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=tools,
        extra_body={"thinking": {"type": "disabled"}}
    )
    assistant_message = response.choices[0].message

    # If the model decides to call a tool
    if assistant_message.tool_calls:
        messages.append(assistant_message)
        for tool_call in assistant_message.tool_calls:
            # `arguments` is a JSON string that needs to be parsed into a dictionary.
            args = json.loads(tool_call.function.arguments)
            # **args unpacks a dictionary into keyword arguments; it is equivalent to func(city="Beijing").
            func = tool_functions[tool_call.function.name]
            result = func(**args)
            print(f"  [Tool calling] {tool_call.function.name}({args}) => {result}")
            # The role "tool" indicates that this is the result returned by a tool.
            # tool_call_id is used to associate this result with the corresponding tool call.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
            
        # After obtaining the tool results, run the model again to generate a natural language response.
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}}
        )
        assistant_message = response.choices[0].message

    messages.append({"role": "assistant", "content": assistant_message.content})
    print(f"AI: {assistant_message.content}\n")
