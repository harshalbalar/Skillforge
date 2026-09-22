"""
SkillForge — LangGraph Agent Pipeline.

Two modes:
  EXPLORE: Dynamic multi-agent pipeline (planner → researcher → formatter).
           Used when no matching skill exists. Logs full execution trace.
  EXECUTE: Runs a stored skill's pre-defined workflow with parameter extraction.
           Faster and cheaper — skips the planning step.
"""

import json
import time
import asyncio
from typing import AsyncGenerator
from datetime import datetime

import google.generativeai as genai
from langgraph.graph import StateGraph, END

from app.models import (
    Skill, Execution, ExecutionStep, ExecutionMode, SSEEvent,
)


# ── Gemini Client ──

def get_gemini_model():
    """Returns the Gemini model. Configure API key via environment."""
    return genai.GenerativeModel("gemini-3.6-flash")


async def call_gemini(prompt: str, system_instruction: str = "") -> tuple[str, int]:
    """Call Gemini and return (response_text, approximate_token_count)."""
    model = get_gemini_model()

    full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

    response = await asyncio.to_thread(
        model.generate_content, full_prompt
    )
    text = response.text
    # Approximate token count (Gemini doesn't always return exact counts)
    approx_tokens = len(full_prompt.split()) + len(text.split())
    return text, approx_tokens


# ── Explore Mode: Dynamic Multi-Agent Pipeline ──

class ExploreState(dict):
    """State passed between agents in explore mode."""
    pass


async def explore_pipeline(
    task: str,
    execution: Execution,
) -> AsyncGenerator[SSEEvent, None]:
    """
    Run the full explore pipeline for an unknown task.
    Yields SSE events as each agent completes.
    """

    # Step 1: Planner — break the task into a plan
    yield SSEEvent(
        event_type="agent_start",
        agent="planner",
        action="planning",
        content=f"Breaking down: {task}",
    )

    start = time.time()
    plan_prompt = f"""You are a task planner for an AI agent system.

The user wants: {task}

Break this into 2-4 concrete steps an AI agent should take.
For each step, specify:
- What to search for or analyze
- What output to produce

Return ONLY a JSON object:
{{
    "task_type": "research|compare|summarize|find_alternatives|analyze|other",
    "steps": [
        {{"description": "...", "search_query": "...", "expected_output": "..."}}
    ],
    "final_format": "report|comparison|list|summary"
}}"""

    plan_text, plan_tokens = await call_gemini(plan_prompt)
    plan_duration = int((time.time() - start) * 1000)

    execution.steps.append(ExecutionStep(
        agent="planner", action="planning",
        input_text=task, output_text=plan_text,
        duration_ms=plan_duration, tokens_used=plan_tokens,
    ))

    yield SSEEvent(
        event_type="agent_output",
        agent="planner",
        action="planning",
        content=plan_text,
        metadata={"duration_ms": plan_duration},
    )

    # Step 2: Researcher — execute each search step
    yield SSEEvent(
        event_type="agent_start",
        agent="researcher",
        action="researching",
        content="Gathering information from sources...",
    )

    start = time.time()
    research_prompt = f"""You are a thorough research agent.

Task: {task}
Plan: {plan_text}

Execute the research plan. For each step:
1. Provide detailed, factual information
2. Include specific names, numbers, and examples
3. Cite what you know from your training data

Compile ALL your research findings into a comprehensive JSON:
{{
    "findings": [
        {{
            "question": "what was researched",
            "answer": "detailed findings",
            "key_points": ["point1", "point2"],
            "confidence": "high|medium|low"
        }}
    ]
}}

Be thorough and specific. No vague generalities."""

    research_text, research_tokens = await call_gemini(research_prompt)
    research_duration = int((time.time() - start) * 1000)

    execution.steps.append(ExecutionStep(
        agent="researcher", action="researching",
        input_text=plan_text, output_text=research_text,
        duration_ms=research_duration, tokens_used=research_tokens,
    ))

    yield SSEEvent(
        event_type="agent_output",
        agent="researcher",
        action="researching",
        content=research_text,
        metadata={"duration_ms": research_duration},
    )

    # Step 3: Formatter — compile into final output
    yield SSEEvent(
        event_type="agent_start",
        agent="formatter",
        action="formatting",
        content="Compiling final output...",
    )

    start = time.time()
    format_prompt = f"""You are a report formatter. Compile research into a polished output.

Original task: {task}
Plan: {plan_text}
Research findings: {research_text}

Create a well-structured, complete response that directly answers the user's task.

Rules:
- Start with a 2-3 sentence executive summary
- Organize findings clearly with headers
- Include specific details, names, numbers
- End with key takeaways or recommendations
- Write in clear, professional prose — not bullet-point soup
- Be opinionated where appropriate

Output the final formatted response (plain text with markdown formatting)."""

    format_text, format_tokens = await call_gemini(format_prompt)
    format_duration = int((time.time() - start) * 1000)

    execution.steps.append(ExecutionStep(
        agent="formatter", action="formatting",
        input_text=research_text, output_text=format_text,
        duration_ms=format_duration, tokens_used=format_tokens,
    ))

    yield SSEEvent(
        event_type="agent_output",
        agent="formatter",
        action="formatting",
        content=format_text,
        metadata={"duration_ms": format_duration},
    )

    # Finalize execution
    execution.final_output = format_text
    execution.total_duration_ms = sum(s.duration_ms for s in execution.steps)
    execution.total_tokens = sum(s.tokens_used for s in execution.steps)
    # Gemini Flash pricing: ~$0.075/1M input, ~$0.30/1M output (approx)
    execution.estimated_cost_usd = round(execution.total_tokens * 0.0002 / 1000, 6)

    yield SSEEvent(
        event_type="complete",
        content=format_text,
        metadata={
            "mode": "explore",
            "total_duration_ms": execution.total_duration_ms,
            "total_tokens": execution.total_tokens,
            "estimated_cost": execution.estimated_cost_usd,
            "steps_count": len(execution.steps),
        },
    )


# ── Execute Mode: Run a Stored Skill ──

async def execute_skill(
    task: str,
    skill: Skill,
    execution: Execution,
) -> AsyncGenerator[SSEEvent, None]:
    """
    Run a stored skill's workflow. Faster because the agent
    doesn't need to plan — it follows the pre-defined steps.
    """

    yield SSEEvent(
        event_type="routing",
        content=f"Matched skill: {skill.name} (confidence: {execution.skill_confidence:.2f})",
        metadata={"skill_id": skill.id, "skill_name": skill.name},
    )

    # Step 1: Extract parameters from the task using the skill's param schema
    yield SSEEvent(
        event_type="agent_start",
        agent="parameter_extractor",
        action="extracting_params",
        content=f"Extracting parameters for {skill.name}...",
    )

    param_schema = {
        k: {"type": v.type, "description": v.description, "default": v.default}
        for k, v in skill.parameters.items()
    }

    start = time.time()
    extract_prompt = f"""Extract parameters from this task for the "{skill.name}" skill.

Task: {task}
Skill description: {skill.description}

Parameter schema:
{json.dumps(param_schema, indent=2)}

Return ONLY a JSON object mapping parameter names to extracted values.
Use defaults for any parameter not mentioned in the task.
Example: {{"topic": "machine learning", "depth": "detailed", "num_sources": 5}}"""

    params_text, params_tokens = await call_gemini(extract_prompt)
    params_duration = int((time.time() - start) * 1000)

    execution.steps.append(ExecutionStep(
        agent="parameter_extractor", action="extracting_params",
        input_text=task, output_text=params_text,
        duration_ms=params_duration, tokens_used=params_tokens,
    ))

    # Parse extracted parameters
    try:
        # Clean up potential markdown fencing
        clean = params_text.strip().strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        params = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        params = {"topic": task}  # Fallback

    yield SSEEvent(
        event_type="agent_output",
        agent="parameter_extractor",
        action="extracting_params",
        content=json.dumps(params),
        metadata={"duration_ms": params_duration},
    )

    # Step 2: Execute each workflow step with filled-in parameters
    accumulated_context = ""

    for i, step in enumerate(skill.workflow.steps):
        yield SSEEvent(
            event_type="agent_start",
            agent=step.agent,
            action=step.action,
            content=step.description,
        )

        # Fill in the prompt template with extracted params + accumulated context
        filled_prompt = step.prompt_template
        for key, value in params.items():
            filled_prompt = filled_prompt.replace(f"{{{key}}}", str(value))

        # Replace context placeholders with accumulated output
        filled_prompt = filled_prompt.replace("{findings}", accumulated_context)
        filled_prompt = filled_prompt.replace("{fetched_content}", accumulated_context)
        filled_prompt = filled_prompt.replace("{alternatives}", accumulated_context)
        filled_prompt = filled_prompt.replace("{tool_data}", accumulated_context)

        start = time.time()
        step_output, step_tokens = await call_gemini(filled_prompt)
        step_duration = int((time.time() - start) * 1000)

        accumulated_context += f"\n\n{step_output}"

        execution.steps.append(ExecutionStep(
            agent=step.agent, action=step.action,
            input_text=filled_prompt[:500], output_text=step_output,
            duration_ms=step_duration, tokens_used=step_tokens,
        ))

        yield SSEEvent(
            event_type="agent_output",
            agent=step.agent,
            action=step.action,
            content=step_output,
            metadata={"duration_ms": step_duration, "step": i + 1},
        )

    # Finalize
    execution.final_output = execution.steps[-1].output_text
    execution.total_duration_ms = sum(s.duration_ms for s in execution.steps)
    execution.total_tokens = sum(s.tokens_used for s in execution.steps)
    execution.estimated_cost_usd = round(execution.total_tokens * 0.0002 / 1000, 6)

    yield SSEEvent(
        event_type="complete",
        content=execution.final_output,
        metadata={
            "mode": "execute",
            "skill_used": skill.name,
            "total_duration_ms": execution.total_duration_ms,
            "total_tokens": execution.total_tokens,
            "estimated_cost": execution.estimated_cost_usd,
        },
    )
