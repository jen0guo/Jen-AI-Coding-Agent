# Crew AI

import os

from crewai import LLM, Agent, Crew, Process, Task

llm = LLM(model="deepseek/deepseek-chat", api_key=os.environ["API_KEY"])

# Role 1: Researcher
researcher = Agent(
    role="Technical Researcher",
    goal="Gather key facts about a technical topic and organize them into concise bullet points",
    backstory="You are an experienced technical researcher who excels at capturing the essence of a topic without being verbose.",
    llm=llm,
)

# Role 2: Writer
writer = Agent(
    role="Technical Writer",
    goal="Rewrite the researcher's bullet points into an easy-to-understand short article",
    backstory="You excel at explaining technical concepts in plain language. Your writing is concise and engaging, avoiding jargon overload.",
    llm=llm,
)

# Task 1: Research
research_task = Task(
    description="Research the major improvements HTTP/2 brings over HTTP/1.1 and summarize them into 3-5 bullet points.",
    expected_output="A brief bullet-point list, one sentence per point.",
    agent=researcher,
)

# Task 2: Write a short article based on the research
writing_task = Task(
    description="Based on the researcher's bullet points, write a popular introduction within 150 words. No lists — use one flowing paragraph.",
    expected_output="A single paragraph within 150 words.",
    agent=writer,
    # Depends on the output of task 1
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
)

crew.kickoff()
print("=== Researcher's Bullet Points ===")
print(research_task.output.raw)
print("\n=== Writer's Final Draft ===")
print(writing_task.output.raw)