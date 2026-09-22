"""
SkillForge — FastAPI Backend.

Endpoints:
  POST /api/task              Submit a task → routes to skill or explore mode
  GET  /api/task/{id}/stream  SSE stream of execution events
  GET  /api/skills            List all skills with stats
  GET  /api/skills/{id}       Get single skill details
  GET  /api/executions        Execution history
  POST /api/feedback          Submit feedback on an execution
  GET  /api/stats             Dashboard metrics
"""

import os
import json
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.models import (
    TaskRequest, TaskResponse, Execution, ExecutionMode,
    FeedbackType, SSEEvent,
)
from app.skill_store import SkillStore
from app.default_skills import get_default_skills
from app.agent_pipeline import explore_pipeline, execute_skill


# ── App Setup ──

# Store active executions for SSE streaming
active_executions: dict[str, asyncio.Queue] = {}

store: SkillStore = None  # Initialized in lifespan


def _is_duplicate_name(name: str) -> bool:
    """Check if a skill with a similar name already exists."""
    existing = store.list_skills()
    for s in existing:
        if s.name == name:
            return True
        if s.name in name or name in s.name:
            return True
        # Also check word overlap — "explain_technical_concept" vs "explain_concept"
        existing_words = set(s.name.split("_"))
        new_words = set(name.split("_"))
        overlap = existing_words & new_words
        if len(overlap) >= 2 and len(overlap) / max(len(existing_words), len(new_words)) > 0.6:
            return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize skill store and load default skills on startup."""
    global store

    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)

    # Initialize store and load defaults
    store = SkillStore()
    if store.skill_count() == 0:
        print("Loading default skills...")
        for skill in get_default_skills():
            store.add_skill(skill)
        print(f"Loaded {store.skill_count()} default skills.")
    else:
        print(f"Found {store.skill_count()} existing skills.")

    yield

    # Cleanup
    active_executions.clear()


app = FastAPI(
    title="SkillForge",
    description="Self-evolving agentic AI that builds its own skills",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Task Submission ──

@app.post("/api/task", response_model=TaskResponse)
async def submit_task(request: TaskRequest):
    """
    Submit a task. The router decides:
      - If a matching skill exists → EXECUTE mode (fast)
      - If no match → EXPLORE mode (full agent pipeline)

    Returns an execution ID. Stream events via /api/task/{id}/stream.
    """
    task = request.task.strip()
    if not task:
        raise HTTPException(400, "Task cannot be empty")

    # Route: find matching skill
    skill, confidence = store.find_matching_skill(task)

    if skill and confidence >= 0.55:
        mode = ExecutionMode.EXECUTE
        execution = Execution(
            task_input=task,
            mode=mode,
            skill_id=skill.id,
            skill_confidence=confidence,
        )
    else:
        mode = ExecutionMode.EXPLORE
        skill = None
        execution = Execution(
            task_input=task,
            mode=mode,
            skill_confidence=confidence if confidence > 0 else None,
        )

    # Create a queue for SSE streaming
    queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue()
    active_executions[execution.id] = queue

    # Start the pipeline in the background
    asyncio.create_task(
        _run_pipeline(execution, skill, queue)
    )

    return TaskResponse(
        execution_id=execution.id,
        mode=mode,
        skill_used=skill.name if skill else None,
        skill_confidence=confidence if confidence > 0 else None,
    )


async def _run_pipeline(
    execution: Execution,
    skill,
    queue: asyncio.Queue,
):
    """Run the appropriate pipeline and push events to the SSE queue."""
    try:
        if execution.mode == ExecutionMode.EXECUTE and skill:
            async for event in execute_skill(
                execution.task_input, skill, execution
            ):
                await queue.put(event)
        else:
            async for event in explore_pipeline(
                execution.task_input, execution
            ):
                await queue.put(event)

        # Save execution trace
        store.save_execution(execution.model_dump())

        # Update skill stats if we used one
        if execution.skill_id:
            stored_skill = store.get_skill(execution.skill_id)
            if stored_skill:
                stats = stored_skill.stats
                stats.times_used += 1
                stats.last_used_at = datetime.utcnow().isoformat()
                # Update rolling average execution time
                n = stats.times_used
                stats.avg_execution_time_ms = (
                    (stats.avg_execution_time_ms * (n - 1) + execution.total_duration_ms) / n
                )
                stats.avg_cost_usd = (
                    (stats.avg_cost_usd * (n - 1) + execution.estimated_cost_usd) / n
                )
                store.update_skill_stats(execution.skill_id, stats)

        # ── Phase 2: Auto-extract skill from explore-mode runs ──
        if execution.mode == ExecutionMode.EXPLORE and execution.final_output:
            await queue.put(SSEEvent(
                event_type="extracting_skill",
                agent="skill_extractor",
                content="Analyzing execution trace to extract a reusable skill...",
            ))

            from app.skill_extractor import extract_skill_from_execution
            result = await extract_skill_from_execution(execution)

            if result["success"]:
                new_skill = result["skill"]

                # Check 1: Name-based duplicate detection
                if _is_duplicate_name(new_skill.name):
                    await queue.put(SSEEvent(
                        event_type="skill_duplicate",
                        agent="skill_extractor",
                        content=f"Skill with similar name already exists: {new_skill.name}. Skipped.",
                    ))
                else:
                    # Check 2: Embedding similarity duplicate detection
                    existing, similarity = store.find_matching_skill(new_skill.description)
                    if existing and similarity > 0.60:
                        await queue.put(SSEEvent(
                            event_type="skill_duplicate",
                            agent="skill_extractor",
                            content=f"Similar skill already exists: {existing.name} ({similarity:.0%} match). Skipped.",
                            metadata={"existing_skill": existing.name, "similarity": similarity},
                        ))
                    else:
                        stored = store.add_skill(new_skill)
                        # Mark execution as having produced a skill
                        execution.skill_extracted = True
                        store.save_execution(execution.model_dump())

                        await queue.put(SSEEvent(
                            event_type="skill_extracted",
                            agent="skill_extractor",
                            content=f"New skill learned: {stored.name}",
                            metadata={
                                "skill_id": stored.id,
                                "skill_name": stored.name,
                                "skill_description": stored.description,
                                "parameters": list(stored.parameters.keys()),
                                "category": result.get("category", "other"),
                                "extraction_tokens": result.get("extraction_tokens", 0),
                            },
                        ))
            else:
                await queue.put(SSEEvent(
                    event_type="extraction_failed",
                    agent="skill_extractor",
                    content=f"Skill extraction failed: {result['error']}",
                    metadata={"error": result["error"]},
                ))

    except Exception as e:
        await queue.put(SSEEvent(
            event_type="error",
            content=f"Pipeline error: {str(e)}",
        ))
    finally:
        await queue.put(None)  # Signal end of stream


# ── SSE Streaming ──

@app.get("/api/task/{execution_id}/stream")
async def stream_execution(execution_id: str):
    """Stream execution events as Server-Sent Events."""
    queue = active_executions.get(execution_id)
    if not queue:
        raise HTTPException(404, "Execution not found or already completed")

    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=120)
                if event is None:
                    yield f"data: {json.dumps({'event_type': 'done'})}\n\n"
                    break
                yield f"data: {event.model_dump_json()}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event_type': 'timeout'})}\n\n"
        finally:
            active_executions.pop(execution_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Skills ──

@app.get("/api/skills")
async def list_skills():
    """List all skills with their stats."""
    skills = store.list_skills()
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "source": s.source.value,
            "version": s.version,
            "stats": {
                "times_used": s.stats.times_used,
                "success_rate": s.stats.success_rate,
                "avg_execution_time_ms": s.stats.avg_execution_time_ms,
                "avg_cost_usd": s.stats.avg_cost_usd,
                "created_at": s.stats.created_at,
                "last_used_at": s.stats.last_used_at,
            },
            "parameter_count": len(s.parameters),
            "example_count": len(s.examples),
        }
        for s in skills
    ]


@app.get("/api/skills/{skill_id}")
async def get_skill(skill_id: str):
    """Get full skill details including workflow and examples."""
    skill = store.get_skill(skill_id)
    if not skill:
        raise HTTPException(404, "Skill not found")
    return skill.model_dump()


# ── Executions ──

@app.get("/api/executions")
async def list_executions(limit: int = 50):
    """List recent executions with metrics."""
    return store.list_executions(limit=limit)


# ── Feedback ──

@app.post("/api/feedback")
async def submit_feedback(execution_id: str, feedback: str):
    """Submit user feedback on an execution result."""
    if feedback not in ("positive", "negative"):
        raise HTTPException(400, "Feedback must be 'positive' or 'negative'")

    executions = store.list_executions(limit=200)
    for ex in executions:
        if ex["id"] == execution_id:
            ex["user_feedback"] = feedback
            # Re-save (simple approach for SQLite)
            import sqlite3
            conn = sqlite3.connect(store.db_path)
            conn.execute(
                "UPDATE executions SET data = ? WHERE id = ?",
                (json.dumps(ex), execution_id),
            )
            conn.commit()
            conn.close()

            # Update skill stats if applicable
            if ex.get("skill_id") and feedback == "positive":
                skill = store.get_skill(ex["skill_id"])
                if skill:
                    skill.stats.success_count += 1
                    store.update_skill_stats(skill.id, skill.stats)
            elif ex.get("skill_id") and feedback == "negative":
                skill = store.get_skill(ex["skill_id"])
                if skill:
                    skill.stats.fail_count += 1
                    store.update_skill_stats(skill.id, skill.stats)

            return {"status": "ok", "execution_id": execution_id, "feedback": feedback}

    raise HTTPException(404, "Execution not found")


# ── Dashboard Stats ──

@app.get("/api/stats")
async def get_stats():
    """Aggregate stats for the dashboard."""
    skills = store.list_skills()
    executions = store.list_executions(limit=500)

    total_executions = len(executions)
    explore_count = sum(1 for e in executions if e.get("mode") == "explore")
    execute_count = sum(1 for e in executions if e.get("mode") == "execute")

    total_cost = sum(e.get("estimated_cost_usd", 0) for e in executions)
    total_time = sum(e.get("total_duration_ms", 0) for e in executions)

    avg_explore_time = 0
    avg_execute_time = 0
    if explore_count:
        avg_explore_time = sum(
            e.get("total_duration_ms", 0)
            for e in executions if e.get("mode") == "explore"
        ) / explore_count
    if execute_count:
        avg_execute_time = sum(
            e.get("total_duration_ms", 0)
            for e in executions if e.get("mode") == "execute"
        ) / execute_count

    hardcoded = sum(1 for s in skills if s.source.value == "hardcoded")
    extracted = sum(1 for s in skills if s.source.value == "extracted")

    return {
        "total_skills": len(skills),
        "hardcoded_skills": hardcoded,
        "extracted_skills": extracted,
        "total_executions": total_executions,
        "explore_count": explore_count,
        "execute_count": execute_count,
        "execute_ratio": execute_count / total_executions if total_executions else 0,
        "total_cost_usd": round(total_cost, 6),
        "total_time_ms": total_time,
        "avg_explore_time_ms": round(avg_explore_time),
        "avg_execute_time_ms": round(avg_execute_time),
        "speedup_factor": round(avg_explore_time / avg_execute_time, 1) if avg_execute_time else 0,
    }


# ── Health ──

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "skills_loaded": store.skill_count() if store else 0,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


# ── Manual Skill Extraction ──

@app.post("/api/extract-skill/{execution_id}")
async def extract_skill_manually(execution_id: str):
    """Manually trigger skill extraction from a past explore-mode execution."""
    from app.skill_extractor import extract_skill_from_execution, build_execution_from_dict

    executions = store.list_executions(limit=500)
    exec_dict = None
    for ex in executions:
        if ex["id"] == execution_id:
            exec_dict = ex
            break

    if not exec_dict:
        raise HTTPException(404, "Execution not found")

    if exec_dict.get("mode") != "explore":
        raise HTTPException(400, "Can only extract skills from explore-mode executions")

    if exec_dict.get("skill_extracted"):
        raise HTTPException(400, "Skill already extracted from this execution")

    execution = build_execution_from_dict(exec_dict)
    result = await extract_skill_from_execution(execution)

    if result["success"]:
        new_skill = result["skill"]

        # Check 1: Name duplicate
        if _is_duplicate_name(new_skill.name):
            return {
                "status": "duplicate",
                "message": f"Skill with similar name already exists: {new_skill.name}",
            }

        # Check 2: Embedding duplicate
        existing, similarity = store.find_matching_skill(new_skill.description)
        if existing and similarity > 0.60:
            return {
                "status": "duplicate",
                "message": f"Similar skill already exists: {existing.name} ({similarity:.0%} match)",
                "existing_skill_id": existing.id,
            }

        stored = store.add_skill(new_skill)

        # Mark execution
        exec_dict["skill_extracted"] = True
        import sqlite3
        conn = sqlite3.connect(store.db_path)
        conn.execute(
            "UPDATE executions SET data = ? WHERE id = ?",
            (json.dumps(exec_dict), execution_id),
        )
        conn.commit()
        conn.close()

        return {
            "status": "ok",
            "skill_id": stored.id,
            "skill_name": stored.name,
            "skill_description": stored.description,
            "parameters": list(stored.parameters.keys()),
            "category": result.get("category", "other"),
        }
    else:
        raise HTTPException(500, f"Extraction failed: {result['error']}")


# ── Skill Management ──

@app.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str):
    """Delete a skill from the library."""
    skill = store.get_skill(skill_id)
    if not skill:
        raise HTTPException(404, "Skill not found")
    store.delete_skill(skill_id)
    return {"status": "ok", "deleted": skill_id, "name": skill.name}