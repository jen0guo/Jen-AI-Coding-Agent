# SDK Usage: output_type
import os
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic import BaseModel

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

class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent(model, output_type=CityLocation)
result = agent.run_sync("Which city hosted the 2022 Olympics?")

# result.output is already a CityLocation instance, so the IDE can
# provide autocompletion for its fields.
print(result.output)       # city='Beijing' country='China'
print(result.output.city)  # Beijing