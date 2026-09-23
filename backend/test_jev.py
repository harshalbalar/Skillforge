import os
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

load_dotenv()

client = TypeSafeClient(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api",
)

result = client.system_one(
    model="jev-1.13",
    state="Compare MongoDB vs PostgreSQL for analytics",
    questions={
        "skill": {
            "type": "choice",
            "instructions": "Which skill should handle this task?",
            "criteria": {
                "research_topic": "Researches a topic thoroughly",
                "compare_tools": "Compares multiple tools side by side",
                "summarize_content": "Summarizes content from URLs or text",
                "find_alternatives": "Finds alternatives to a tool",
                "none": "No matching skill exists",
            },
        },
    },
)

print("Chosen skill:", result.answers["skill"])