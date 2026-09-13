import os
from datetime import date

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key or not base_url:
    raise RuntimeError(
        "Please set the API_KEY and BASE_URL environment variables first."
    )

# --- Step 1: Fetch the GitHub Trending page ---
def fetch_trending():
    url = "https://github.com/trending?since=daily"

    # Simulate a browser request to reduce the chance of GitHub rejecting it.
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    # Raise an exception immediately if the request fails, such as a
    # 404 or 500 response.
    response.raise_for_status()

    return response.text


# --- Step 2: Parse the HTML and extract repository information ---
# The selectors below, such as article.Box-row, come from the HTML structure
# of GitHub's page. Their purpose is to locate the corresponding page elements.
def parse_repos(html):
    soup = BeautifulSoup(html, "html.parser")
    repos = []

    for article in soup.select("article.Box-row"):
        # Repository name in owner/repository format
        link = article.select_one("h2 a")
        name = link["href"].strip("/") if link else "unknown"

        # Repository description
        description_tag = article.select_one("p.col-9")
        description = (
            description_tag.text.strip()
            if description_tag
            else "No description"
        )

        # Programming language
        language_tag = article.select_one(
            'span[itemprop="programmingLanguage"]'
        )
        language = language_tag.text.strip() if language_tag else ""

        # Stars gained today
        stars_tag = article.select_one(
            "span.d-inline-block.float-sm-right"
        )
        stars = stars_tag.text.strip() if stars_tag else ""

        repos.append(
            f"- {name} ({language}) {stars}\n"
            f"  {description}"
        )

    return repos


# --- Step 3: Ask the LLM to generate an English summary ---
def summarize(repos):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Use only the first 15 repositories to keep the input concise
    # and avoid consuming too many tokens.
    repo_text = "\n".join(repos[:15])

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "user",
                "content": (
                    "Below is today's list of popular repositories from "
                    "GitHub Trending. Write a concise daily report in English. "
                    "Select the five repositories most worth following and "
                    "briefly explain what each project does and why it is "
                    "noteworthy."
                    f"\n\n{repo_text}"
                ),
            }
        ],
        # Disable thinking mode because this summarization task does not
        # require a visible reasoning process.
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )

    return response.choices[0].message.content


# --- Step 4: Save the report to a local file ---
def save_report(summary):
    filename = f"trending_{date.today()}.md"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# GitHub Trending Daily Report ({date.today()})\n\n")
        file.write(summary)

    print(f"Daily report saved to {filename}")


# --- Run the workflow ---
html = fetch_trending()
repos = parse_repos(html)
summary = summarize(repos)
save_report(summary)