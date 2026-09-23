"""
SkillForge — Training Data Generator for Laya Fine-Tuning.

Generates synthetic training examples from the skill library.
For each skill, Gemini creates task variations that should route to it.
Also generates "none" examples — tasks that shouldn't match any skill.

Output: A JSON file ready for the Laya fine-tuning notebook.

Usage:
  python -m app.generate_training_data

  Or from the API:
  POST /api/generate-training-data
"""

import json
import asyncio
import os
from pathlib import Path

import google.generativeai as genai


OUTPUT_DIR = Path(__file__).parent.parent / "training_data"


TASK_GENERATION_PROMPT = """You are generating training data for a skill routing model.

Given this skill definition, generate {num_tasks} diverse task strings that a user might type
and that THIS skill should handle. The tasks should:
- Vary in phrasing, specificity, and complexity
- Cover different topics the skill could handle
- Include both short and detailed requests
- Use natural, human-like language (not robotic)
- NOT overlap with other skills (listed below for reference)

## Target Skill
Name: {skill_name}
Description: {skill_description}
Parameters: {parameters}
Examples from skill definition: {examples}

## Other Skills (DO NOT generate tasks for these)
{other_skills}

Return ONLY a JSON array of task strings, nothing else:
["task 1", "task 2", "task 3", ...]
"""

NONE_TASKS_PROMPT = """Generate {num_tasks} diverse task strings that should NOT match any of
these skills. These are tasks that require explore mode — no existing skill can handle them.

## Existing Skills
{all_skills}

The tasks should:
- Be legitimate AI agent requests (not garbage)
- Cover topics/patterns not represented by any skill above
- Include creative, analytical, and informational requests
- NOT be answerable by any of the skills listed

Return ONLY a JSON array of task strings:
["task 1", "task 2", "task 3", ...]
"""


async def generate_tasks_for_skill(skill, all_skills, num_tasks=25):
    """Generate training task variations for one skill."""
    model = genai.GenerativeModel("gemini-3.6-flash")

    other_skills_text = "\n".join(
        f"- {s['name']}: {s['description']}"
        for s in all_skills if s['name'] != skill['name']
    )

    prompt = TASK_GENERATION_PROMPT.format(
        num_tasks=num_tasks,
        skill_name=skill['name'],
        skill_description=skill['description'],
        parameters=json.dumps(list(skill.get('parameters', {}).keys())),
        examples=json.dumps([ex.get('input', '') for ex in skill.get('examples', [])]),
        other_skills=other_skills_text,
    )

    response = await asyncio.to_thread(model.generate_content, prompt)
    text = response.text.strip()

    # Clean markdown fencing
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        tasks = json.loads(text)
        return tasks
    except json.JSONDecodeError:
        print(f"  [WARN] Failed to parse tasks for {skill['name']}")
        return []


async def generate_none_tasks(all_skills, num_tasks=30):
    """Generate tasks that shouldn't match any skill."""
    model = genai.GenerativeModel("gemini-3.6-flash")

    skills_text = "\n".join(
        f"- {s['name']}: {s['description']}"
        for s in all_skills
    )

    prompt = NONE_TASKS_PROMPT.format(
        num_tasks=num_tasks,
        all_skills=skills_text,
    )

    response = await asyncio.to_thread(model.generate_content, prompt)
    text = response.text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("  [WARN] Failed to parse none tasks")
        return []


def build_laya_training_example(task: str, correct_skill: str, all_skills: list) -> dict:
    """
    Build one Laya training example in the format the fine-tuning notebook expects.

    Each example has:
      - state: the task text
      - questions: the routing questions
      - answers: the correct routing decision
    """
    criteria = {}
    for s in all_skills:
        criteria[s['name']] = s['description']
    criteria["none"] = "No existing skill matches this task."

    return {
        "state": {"task": task},
        "questions": {
            "skill_route": {
                "type": "choice",
                "instructions": (
                    "Which skill should handle this task? "
                    "Pick the best match, or 'none' if nothing fits."
                ),
                "criteria": criteria,
            },
            "is_good_match": {
                "type": "noul",
                "instructions": "Is there a skill that genuinely matches this task?",
            },
        },
        "answers": {
            "skill_route": correct_skill,
            "is_good_match": 0.0 if correct_skill == "none" else 1.0,
        },
    }


async def generate_training_data(skills_endpoint: str = "http://localhost:8000/api/skills"):
    """
    Full pipeline: fetch skills → generate tasks → build training data → save.

    Can be run standalone or called from the API.
    """
    import aiohttp

    # Fetch current skills from the API
    print("[1/4] Fetching skills from API...")
    async with aiohttp.ClientSession() as session:
        async with session.get(skills_endpoint) as resp:
            all_skills = await resp.json()

    print(f"  Found {len(all_skills)} skills")

    # Generate task variations for each skill
    print("[2/4] Generating task variations with Gemini...")
    training_examples = []
    tasks_per_skill = max(15, 60 // len(all_skills))  # ~60 total positive examples

    for skill in all_skills:
        print(f"  Generating {tasks_per_skill} tasks for: {skill['name']}")
        tasks = await generate_tasks_for_skill(skill, all_skills, tasks_per_skill)
        print(f"    Got {len(tasks)} tasks")

        for task in tasks:
            example = build_laya_training_example(task, skill['name'], all_skills)
            training_examples.append(example)

    # Generate "none" examples
    print(f"[3/4] Generating 'none' (explore mode) tasks...")
    none_tasks = await generate_none_tasks(all_skills, num_tasks=30)
    print(f"  Got {len(none_tasks)} none tasks")

    for task in none_tasks:
        example = build_laya_training_example(task, "none", all_skills)
        training_examples.append(example)

    # Save
    print(f"[4/4] Saving {len(training_examples)} training examples...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "laya_routing_data.json"

    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "total_examples": len(training_examples),
                "skills": [s['name'] for s in all_skills],
                "skill_count": len(all_skills),
                "tasks_per_skill": tasks_per_skill,
                "none_tasks": len(none_tasks),
            },
            "examples": training_examples,
        }, f, indent=2)

    print(f"  Saved to: {output_path}")
    print(f"\n  Total: {len(training_examples)} examples")
    print(f"  Skills: {len(all_skills)} + none")
    print(f"\n  Next: Upload this file to the Colab/Kaggle fine-tuning notebook")

    return {
        "path": str(output_path),
        "total_examples": len(training_examples),
        "skills": [s['name'] for s in all_skills],
    }


# Standalone execution
if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        exit(1)

    asyncio.run(generate_training_data())
