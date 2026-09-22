"""
SkillForge — Data models for skills, executions, and tasks.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ── Enums ──

class ExecutionMode(str, Enum):
    EXPLORE = "explore"
    EXECUTE = "execute"


class SkillSource(str, Enum):
    HARDCODED = "hardcoded"
    EXTRACTED = "extracted"


class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


# ── Task ──

class TaskRequest(BaseModel):
    """What the user sends from the frontend."""
    task: str = Field(..., description="Natural language task description")


class TaskResponse(BaseModel):
    """Initial response with execution ID for SSE streaming."""
    execution_id: str
    mode: ExecutionMode
    skill_used: Optional[str] = None
    skill_confidence: Optional[float] = None


# ── Skill ──

class SkillParameter(BaseModel):
    type: str = "string"
    description: str = ""
    enum: Optional[list[str]] = None
    default: Optional[str | int | bool] = None


class SkillWorkflowStep(BaseModel):
    agent: str
    action: str
    prompt_template: str
    description: str = ""


class SkillWorkflow(BaseModel):
    steps: list[SkillWorkflowStep]


class SkillExample(BaseModel):
    input: str
    output_summary: str


class SkillStats(BaseModel):
    times_used: int = 0
    success_count: int = 0
    fail_count: int = 0
    avg_execution_time_ms: float = 0
    avg_cost_usd: float = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used_at: Optional[str] = None

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0


class Skill(BaseModel):
    id: str = Field(default_factory=lambda: f"skill_{uuid.uuid4().hex[:8]}")
    name: str
    description: str
    version: int = 1
    parameters: dict[str, SkillParameter] = {}
    workflow: SkillWorkflow
    examples: list[SkillExample] = []
    stats: SkillStats = Field(default_factory=SkillStats)
    source: SkillSource = SkillSource.HARDCODED
    embedding: Optional[list[float]] = None


# ── Execution Trace ──

class ExecutionStep(BaseModel):
    agent: str
    action: str
    input_text: str
    output_text: str
    duration_ms: int = 0
    tokens_used: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Execution(BaseModel):
    id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}")
    task_input: str
    mode: ExecutionMode
    skill_id: Optional[str] = None
    skill_confidence: Optional[float] = None
    steps: list[ExecutionStep] = []
    final_output: Optional[str] = None
    total_duration_ms: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0
    user_feedback: Optional[FeedbackType] = None
    skill_extracted: bool = False
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── SSE Events ──

class SSEEvent(BaseModel):
    """Events streamed to the frontend during execution."""
    event_type: str  # "routing", "agent_start", "agent_output", "complete", "error"
    agent: Optional[str] = None
    action: Optional[str] = None
    content: str = ""
    metadata: dict = {}
