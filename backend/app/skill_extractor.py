"""
SkillForge — Skill Extraction Engine (Phase 2).

After a successful explore-mode execution, this module analyzes the full
execution trace and extracts a generalized, reusable skill. The new skill
gets stored in the skill library so future similar tasks route to it
instead of re-planning from scratch.

Flow:
  1. Take the completed Execution trace (task + all agent steps + output)
  2. Ask Gemini to identify the task pattern, extract parameters, and
     generalize the workflow into reusable prompt templates
  3. Validate the extracted skill (basic sanity checks)
  4. Store it in the skill library with computed embeddings
"""

import json
import asyncio
from datetime import datetime

from app.models import (
    Skill, SkillParameter, SkillWorkflow, SkillWorkflowStep,
    SkillExample, SkillStats, SkillSource, Execution,
)
from app.agent_pipeline import call_gemini


EXTRACTION_PROMPT = """You are a skill extraction engine for an AI agent system called SkillForge.

You just watched an AI agent successfully complete a task using a dynamic explore pipeline. Your job is to analyze the execution trace and extract a REUSABLE SKILL that can handle similar tasks in the future without re-planning.

## The Completed Execution

**User's original task:**
{task_input}

**Agent steps that were executed:**
{steps_summary}

**Final output the user received:**
{final_output_preview}

## Your Job

Extract a generalized skill from this execution. The skill should work not just for this exact task, but for the CATEGORY of task it represents.

For example:
- "What are the pros and cons of using Rust for backend development" → general skill: "analyze_pros_cons" that works for ANY technology/topic
- "Explain how transformers work in simple terms" → general skill: "explain_concept" that works for ANY technical concept
- "What happened in AI this week" → general skill: "recent_developments" that works for ANY topic/timeframe

Return ONLY a valid JSON object (no markdown fencing, no extra text):

{{
    "name": "snake_case_skill_name",
    "description": "One paragraph describing what this skill does generally, not specific to the original task",
    "parameters": {{
        "param_name": {{
            "type": "string",
            "description": "What this parameter controls",
            "default": "optional default value or null"
        }}
    }},
    "workflow_steps": [
        {{
            "agent": "planner|researcher|formatter|analyzer",
            "action": "short_action_name",
            "description": "What this step does",
            "prompt_template": "The full prompt template with {{param_name}} placeholders for parameters. This should be a generalized version of what the agent actually did, not hardcoded to the original task."
        }}
    ],
    "examples": [
        {{
            "input": "The original task that triggered this extraction",
            "output_summary": "One-line summary of what was produced"
        }},
        {{
            "input": "A different task this skill could also handle",
            "output_summary": "What it would produce for that task"
        }},
        {{
            "input": "Another different example task",
            "output_summary": "What it would produce"
        }}
    ],
    "category": "research|comparison|analysis|explanation|summary|recommendation|other"
}}

IMPORTANT RULES:
- Make the skill GENERAL, not specific to the original task
- Parameter names should use {{curly_braces}} in prompt templates
- Include at least 2-3 workflow steps
- The first example should be the original task; others should be DIFFERENT tasks the same skill could handle
- Prompt templates should be detailed enough to produce quality output without additional planning
- Keep prompt templates practical — they'll be sent directly to Gemini"""


async def extract_skill_from_execution(execution: Execution) -> dict:
    """
    Analyze a completed explore-mode execution and extract a reusable skill.

    Returns a dict with:
      - "success": bool
      - "skill": Skill object (if successful)
      - "error": str (if failed)
      - "raw_response": str (the Gemini output for debugging)
    """

    # Build a summary of the execution steps
    steps_summary = ""
    for i, step in enumerate(execution.steps):
        steps_summary += f"\nStep {i+1} — Agent: {step.agent}, Action: {step.action}\n"
        steps_summary += f"  Input (preview): {step.input_text[:300]}...\n"
        steps_summary += f"  Output (preview): {step.output_text[:500]}...\n"
        steps_summary += f"  Duration: {step.duration_ms}ms, Tokens: {step.tokens_used}\n"

    # Preview of the final output (don't send the whole thing — save tokens)
    final_preview = (execution.final_output or "")[:1500]
    if len(execution.final_output or "") > 1500:
        final_preview += "\n... [truncated]"

    # Build the extraction prompt
    prompt = EXTRACTION_PROMPT.format(
        task_input=execution.task_input,
        steps_summary=steps_summary,
        final_output_preview=final_preview,
    )

    try:
        response_text, tokens_used = await call_gemini(prompt)

        # Clean up response — strip markdown fencing if present
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]  # Remove first line
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()

        # Parse the JSON
        skill_data = json.loads(clean)

        # Validate required fields
        required_fields = ["name", "description", "parameters", "workflow_steps", "examples"]
        for field in required_fields:
            if field not in skill_data:
                return {
                    "success": False,
                    "error": f"Missing required field: {field}",
                    "raw_response": response_text,
                }

        if not skill_data["workflow_steps"]:
            return {
                "success": False,
                "error": "No workflow steps extracted",
                "raw_response": response_text,
            }

        # Build the Skill object
        parameters = {}
        for param_name, param_data in skill_data.get("parameters", {}).items():
            parameters[param_name] = SkillParameter(
                type=param_data.get("type", "string"),
                description=param_data.get("description", ""),
                default=param_data.get("default"),
            )

        workflow_steps = []
        for step_data in skill_data["workflow_steps"]:
            workflow_steps.append(SkillWorkflowStep(
                agent=step_data.get("agent", "researcher"),
                action=step_data.get("action", "process"),
                description=step_data.get("description", ""),
                prompt_template=step_data.get("prompt_template", ""),
            ))

        examples = []
        for ex_data in skill_data.get("examples", []):
            examples.append(SkillExample(
                input=ex_data.get("input", ""),
                output_summary=ex_data.get("output_summary", ""),
            ))

        skill = Skill(
            name=skill_data["name"],
            description=skill_data["description"],
            version=1,
            parameters=parameters,
            workflow=SkillWorkflow(steps=workflow_steps),
            examples=examples,
            stats=SkillStats(),
            source=SkillSource.EXTRACTED,
        )

        return {
            "success": True,
            "skill": skill,
            "category": skill_data.get("category", "other"),
            "raw_response": response_text,
            "extraction_tokens": tokens_used,
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Failed to parse Gemini response as JSON: {str(e)}",
            "raw_response": response_text if 'response_text' in dir() else "No response",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Extraction failed: {str(e)}",
            "raw_response": "",
        }


def build_execution_from_dict(exec_dict: dict) -> Execution:
    """Reconstruct an Execution object from a stored dict."""
    from app.models import ExecutionStep, ExecutionMode

    steps = []
    for s in exec_dict.get("steps", []):
        steps.append(ExecutionStep(
            agent=s.get("agent", ""),
            action=s.get("action", ""),
            input_text=s.get("input_text", ""),
            output_text=s.get("output_text", ""),
            duration_ms=s.get("duration_ms", 0),
            tokens_used=s.get("tokens_used", 0),
            timestamp=s.get("timestamp", ""),
        ))

    return Execution(
        id=exec_dict.get("id", ""),
        task_input=exec_dict.get("task_input", ""),
        mode=ExecutionMode(exec_dict.get("mode", "explore")),
        skill_id=exec_dict.get("skill_id"),
        skill_confidence=exec_dict.get("skill_confidence"),
        steps=steps,
        final_output=exec_dict.get("final_output"),
        total_duration_ms=exec_dict.get("total_duration_ms", 0),
        total_tokens=exec_dict.get("total_tokens", 0),
        estimated_cost_usd=exec_dict.get("estimated_cost_usd", 0),
        user_feedback=exec_dict.get("user_feedback"),
        skill_extracted=exec_dict.get("skill_extracted", False),
        created_at=exec_dict.get("created_at", ""),
    )
