import os
from datetime import date

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from prefect import flow, task

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# The @task decorator turns this function into a task managed by Prefect.
# retries=3 means Prefect will automatically retry the task up to three times
# if the request fails.
@task(retries=3)
def fetch_trending():
    url = "https://github.com/trending?since=daily"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.text


@task
def parse_repos(html):
    soup = BeautifulSoup(html, "html.parser")
    repos = []

    for article in soup.select("article.Box-row"):
        link = article.select_one("h2 a")
        name = link["href"].strip("/") if link else "unknown"

        description_tag = article.select_one("p.col-9")
        description = (
            description_tag.text.strip()
            if description_tag
            else "No description"
        )

        language_tag = article.select_one(
            'span[itemprop="programmingLanguage"]'
        )
        language = language_tag.text.strip() if language_tag else ""

        stars_tag = article.select_one(
            "span.d-inline-block.float-sm-right"
        )
        stars = stars_tag.text.strip() if stars_tag else ""

        repos.append(
            f"- {name} ({language}) {stars}\n"
            f"  {description}"
        )

    return repos


@task
def summarize(repos):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

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
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )

    return response.choices[0].message.content


@task
def save_report(summary):
    filename = f"trending_{date.today()}.md"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# GitHub Trending Daily Report ({date.today()})\n\n")
        file.write(summary)

    print(f"Daily report saved to {filename}")


# The @flow decorator marks the entry-point function. Prefect automatically
# tracks the execution status of every @task called inside it.
@flow
def daily_report():
    html = fetch_trending()
    repos = parse_repos(html)
    summary = summarize(repos)
    save_report(summary)


# Run the workflow once to test it.
daily_report()

# To deploy it as a scheduled workflow, replace daily_report() above
# with the following line:
# daily_report.serve(name="github-trending", cron="0 9 * * *")