"""
SkillForge — Skill Refiner (Phase 4).

When a user gives negative feedback on an execution, this module:
  1. Analyzes what went wrong (the task, skill used, output produced)
  2. Asks Gemini to diagnose the failure and improve the skill's prompts
  3. Updates the skill with refined prompt templates
  4. Increments the skill version

The skill gets better with each refinement cycle. Refinement history
is tracked so you can see how a skill evolved over time.
"""

import json
import asyncio
from datetime import datetime

from app.models import (
    Skill, SkillWorkflow, SkillWorkflowStep, SkillParameter,
)
from app.agent_pipeline import call_gemini


REFINEMENT_PROMPT = """You are a skill refinement engine for an AI agent system called SkillForge.

A user ran a task through a stored skill and rated the output as BAD. Your job is to analyze what went wrong and produce IMPROVED prompt templates that will fix the problem.

## The Failed Execution

**User's task:**
{task}

**Skill that was used:**
Name: {skill_name}
Description: {skill_description}
Parameters: {parameters}

**Extracted parameters:**
{extracted_params}

**Current prompt templates and their outputs:**
{steps_detail}

**Final output the user rejected:**
{final_output_preview}

{user_reason}

## Your Analysis

Think step by step:
1. What did the user likely expect from this task?
2. What did the output actually deliver?
3. Where is the gap? (wrong focus, too shallow, missing specifics, wrong format, etc.)
4. Which prompt template(s) need to change and how?

## Your Job

Return ONLY a valid JSON object with the improved skill:

{{
    "diagnosis": "One paragraph explaining what went wrong and why",
    "changes_made": ["List of specific changes to the prompt templates"],
    "improved_description": "Updated skill description if needed (or null to keep current)",
    "improved_steps": [
        {{
            "agent": "researcher|formatter|planner|analyzer",
            "action": "short_action_name",
            "description": "What this step does",
            "prompt_template": "The IMPROVED prompt template. Keep {{parameter}} placeholders. Fix the specific issue identified in the diagnosis."
        }}
    ]
}}

RULES:
- Keep the same parameter names — don't change the parameter schema
- Keep the same number of steps unless adding/removing one genuinely fixes the issue
- The improved prompts should be MORE specific, MORE structured, and produce BETTER output
- Focus on the actual failure — don't rewrite everything, fix what's broken
- Prompt templates must still use {{param_name}} placeholders for parameters"""


async def refine_skill(
    skill: Skill,
    execution_data: dict,
    user_reason: str = "",
) -> dict:
    """
    Analyze a bad execution and produce improved skill prompts.

    Args:
        skill: The skill that produced the bad output
        execution_data: The full execution trace dict
        user_reason: Optional user explanation of what was wrong

    Returns:
        dict with "success", "diagnosis", "changes", and "improved_skill" or "error"
    """

    # Build detailed step information
    steps_detail = ""
    extracted_params = "{}"
    for i, step in enumerate(execution_data.get("steps", [])):
        agent = step.get("agent", "unknown")
        action = step.get("action", "unknown")
        input_preview = step.get("input_text", "")[:400]
        output_preview = step.get("output_text", "")[:600]

        if agent == "parameter_extractor":
            extracted_params = output_preview
            steps_detail += f"\nStep {i+1} — Parameter Extraction:\n"
            steps_detail += f"  Extracted: {output_preview}\n"
        else:
            steps_detail += f"\nStep {i+1} — Agent: {agent}, Action: {action}\n"
            steps_detail += f"  Prompt sent (preview): {input_preview}...\n"
            steps_detail += f"  Output produced (preview): {output_preview}...\n"

    # Final output preview
    final_output = execution_data.get("final_output", "")
    final_preview = final_output[:1500]
    if len(final_output) > 1500:
        final_preview += "\n... [truncated]"

    # Build the user reason section
    reason_text = ""
    if user_reason:
        reason_text = f"**User's explanation of what was wrong:**\n{user_reason}"

    # Build parameters info
    params_info = json.dumps(
        {k: {"type": v.type, "description": v.description} for k, v in skill.parameters.items()},
        indent=2
    )

    prompt = REFINEMENT_PROMPT.format(
        task=execution_data.get("task_input", ""),
        skill_name=skill.name,
        skill_description=skill.description,
        parameters=params_info,
        extracted_params=extracted_params,
        steps_detail=steps_detail,
        final_output_preview=final_preview,
        user_reason=reason_text,
    )

    try:
        response_text, tokens_used = await call_gemini(prompt)

        # Clean markdown fencing
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()

        result = json.loads(clean)

        # Validate
        if "improved_steps" not in result or not result["improved_steps"]:
            return {
                "success": False,
                "error": "No improved steps in Gemini response",
                "raw": response_text,
            }

        # Build improved workflow
        improved_steps = []
        for step_data in result["improved_steps"]:
            improved_steps.append(SkillWorkflowStep(
                agent=step_data.get("agent", "researcher"),
                action=step_data.get("action", "process"),
                description=step_data.get("description", ""),
                prompt_template=step_data.get("prompt_template", ""),
            ))

        return {
            "success": True,
            "diagnosis": result.get("diagnosis", ""),
            "changes": result.get("changes_made", []),
            "improved_description": result.get("improved_description"),
            "improved_workflow": SkillWorkflow(steps=improved_steps),
            "tokens_used": tokens_used,
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Failed to parse refinement response: {e}",
            "raw": response_text if 'response_text' in dir() else "",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Refinement failed: {e}",
        }


def apply_refinement(skill: Skill, refinement: dict) -> Skill:
    """
    Apply a successful refinement to a skill.
    Increments version and updates workflow.
    """
    skill.workflow = refinement["improved_workflow"]
    skill.version += 1

    if refinement.get("improved_description"):
        skill.description = refinement["improved_description"]

    return skill