import json
import subprocess
import urllib.request
from bs4 import BeautifulSoup

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


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File {path} does not exist."


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

    return f"Successfully wrote to {path}."


def run_command(command):
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

        # By shell convention, a return code of 0 indicates success,
        # while a nonzero return code indicates an error.
        if result.returncode != 0:
            output += f"\n[Error] {result.stderr}"

        return output or "(No output)"

    except subprocess.TimeoutExpired:
        return "[Error] Command timed out after 10 seconds."
