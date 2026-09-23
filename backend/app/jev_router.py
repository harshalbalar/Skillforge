"""
SkillForge — Jev Router (Phase 3).

Routes tasks to skills using TypeSafe's Jev model via OpenRouter.
One API call (~70-500ms) decides which skill handles a task with
calibrated confidence scores. No fine-tuning needed — works zero-shot.

Jev answers two questions per task:
  1. choice: "Which skill?" → picks from your skill library + "none"
  2. noul: "Is this a good match?" → calibrated probability 0-1

Usage:
  router = JevRouter(skills, api_key="sk-or-...")
  skill, confidence = router.route("Compare MongoDB vs PostgreSQL")
  # -> (Skill(name="compare_tools"), 1.0)
"""

import os
import time
from typing import Optional

from typesafe_sdk import TypeSafeClient


CONFIDENCE_THRESHOLD = 0.55


class JevRouter:
    """
    Routes tasks to skills using Jev via OpenRouter.

    Zero-shot accuracy is already high (~73% on typed decisions),
    so no fine-tuning is needed — just plug in your API key and go.
    """

    def __init__(self, skills: list, api_key: str = None):
        self.skills = skills
        self.client = TypeSafeClient(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api",
        )
        print(f"[Jev] Router initialized with {len(skills)} skills")

    def _build_questions(self) -> dict:
        """Build Jev's typed question schema from current skills."""
        criteria = {}
        for skill in self.skills:
            criteria[skill.name] = skill.description

        criteria["none"] = (
            "No existing skill matches this task. "
            "The task requires a new approach via explore mode."
        )

        return {
            "skill_route": {
                "type": "choice",
                "instructions": (
                    "Which skill from the library should handle this task? "
                    "Pick the skill whose description best matches what the user wants. "
                    "Choose 'none' only if no skill is a reasonable fit."
                ),
                "criteria": criteria,
            },
            "is_good_match": {
                "type": "noul",
                "instructions": (
                    "Is there a skill in the library that is a genuinely "
                    "good match for this task?"
                ),
            },
        }

    def route(self, task: str) -> tuple[Optional[object], float]:
        """
        Route a task to the best matching skill using Jev.

        Returns (skill, confidence) or (None, confidence) if no match.
        """
        start = time.time()

        try:
            result = self.client.system_one(
                model="jev-1.13",
                state=task,
                questions=self._build_questions(),
            )
        except Exception as e:
            print(f"[Jev] API error: {e}")
            return None, 0.0

        elapsed_ms = (time.time() - start) * 1000
        answers = result.answers

        # Extract routing decision
        chosen = answers["skill_route"].choice
        choice_conf = answers["skill_route"].confidence
        match_prob = answers["is_good_match"].noul
        probabilities = answers["skill_route"].probabilities

        # Combined confidence
        combined = (choice_conf * match_prob) ** 0.5

        print(
            f"[Jev] '{chosen}' "
            f"(conf={choice_conf:.3f}, match={match_prob:.3f}, "
            f"combined={combined:.3f}) {elapsed_ms:.0f}ms"
        )

        if chosen == "none" or combined < CONFIDENCE_THRESHOLD:
            return None, combined

        # Find matching skill object
        for skill in self.skills:
            if skill.name == chosen:
                return skill, combined

        print(f"[Jev] WARNING: chose '{chosen}' but not found in library")
        return None, combined

    def update_skills(self, skills: list):
        """Update skill list when new skills are added."""
        self.skills = skills

    def get_diagnostics(self, task: str) -> dict:
        """Detailed routing diagnostics for benchmarking."""
        start = time.time()

        try:
            result = self.client.system_one(
                model="jev-1.13",
                state=task,
                questions=self._build_questions(),
            )
        except Exception as e:
            return {"error": str(e)}

        elapsed_ms = (time.time() - start) * 1000
        answers = result.answers

        return {
            "task": task,
            "router": "jev",
            "chosen_skill": answers["skill_route"].choice,
            "choice_confidence": answers["skill_route"].confidence,
            "match_probability": answers["is_good_match"].noul,
            "probabilities": answers["skill_route"].probabilities,
            "latency_ms": round(elapsed_ms, 1),
        }