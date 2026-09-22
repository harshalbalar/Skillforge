"""
SkillForge — Skill storage and routing.

Uses SQLite for persistence and sentence-transformers for embedding-based
skill matching. In Phase 3, the embedding router can be swapped for Laya.
"""

import json
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.models import Skill, SkillStats, SkillSource


DB_PATH = Path(__file__).parent.parent / "data" / "skillforge.db"
SIMILARITY_THRESHOLD = 0.45  # Minimum cosine similarity to route to a skill


class SkillStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # all-MiniLM-L6-v2 is only ~80MB, runs fast on CPU
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    parameters TEXT DEFAULT '{}',
                    workflow TEXT NOT NULL,
                    examples TEXT DEFAULT '[]',
                    stats TEXT NOT NULL,
                    source TEXT DEFAULT 'hardcoded',
                    embedding TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    # ── Skill CRUD ──

    def add_skill(self, skill: Skill) -> Skill:
        """Add a skill and compute its embedding."""
        # Create embedding from name + description + example inputs
        embed_text = f"{skill.name}: {skill.description}"
        if skill.examples:
            embed_text += " | Examples: " + ", ".join(e.input for e in skill.examples)

        embedding = self.embedder.encode(embed_text).tolist()
        skill.embedding = embedding

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    skill.id,
                    skill.name,
                    skill.description,
                    skill.version,
                    json.dumps({k: v.model_dump() for k, v in skill.parameters.items()}),
                    skill.workflow.model_dump_json(),
                    json.dumps([e.model_dump() for e in skill.examples]),
                    skill.stats.model_dump_json(),
                    skill.source.value,
                    json.dumps(embedding),
                ),
            )
            conn.commit()
        return skill

    def get_skill(self, skill_id: str) -> Skill | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
        return self._row_to_skill(row) if row else None

    def list_skills(self) -> list[Skill]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM skills").fetchall()
        return [self._row_to_skill(r) for r in rows]

    def update_skill_stats(self, skill_id: str, stats: SkillStats):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE skills SET stats = ? WHERE id = ?",
                (stats.model_dump_json(), skill_id),
            )
            conn.commit()

    def delete_skill(self, skill_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            conn.commit()

    # ── Routing ──

    def find_matching_skill(self, task: str) -> tuple[Skill | None, float]:
        """
        Find the best matching skill for a task using cosine similarity.

        Returns (skill, confidence) or (None, 0.0) if no match above threshold.
        Phase 3: replace this method with Laya-based routing.
        """
        skills = self.list_skills()
        if not skills:
            return None, 0.0

        task_embedding = self.embedder.encode(task)

        best_skill = None
        best_score = 0.0

        for skill in skills:
            if skill.embedding is None:
                continue
            skill_emb = np.array(skill.embedding)
            score = float(np.dot(task_embedding, skill_emb) / (
                np.linalg.norm(task_embedding) * np.linalg.norm(skill_emb)
            ))
            if score > best_score:
                best_score = score
                best_skill = skill

        if best_score >= SIMILARITY_THRESHOLD and best_skill:
            return best_skill, best_score

        return None, best_score

    # ── Execution Storage ──

    def save_execution(self, execution_data: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO executions VALUES (?, ?, ?)",
                (
                    execution_data["id"],
                    json.dumps(execution_data),
                    execution_data["created_at"],
                ),
            )
            conn.commit()

    def list_executions(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM executions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ── Helpers ──

    def _row_to_skill(self, row: tuple) -> Skill:
        from app.models import SkillParameter, SkillWorkflow, SkillExample

        params_raw = json.loads(row[4])
        parameters = {k: SkillParameter(**v) for k, v in params_raw.items()}

        return Skill(
            id=row[0],
            name=row[1],
            description=row[2],
            version=row[3],
            parameters=parameters,
            workflow=SkillWorkflow(**json.loads(row[5])),
            examples=[SkillExample(**e) for e in json.loads(row[6])],
            stats=SkillStats(**json.loads(row[7])),
            source=SkillSource(row[8]),
            embedding=json.loads(row[9]) if row[9] else None,
        )

    def skill_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        return count
