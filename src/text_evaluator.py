from __future__ import annotations

import json
import re
from typing import Any, Optional

from config import JUDGE_PROMPT_TEMPLATE

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class TextEvaluator:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError("The openai package is not installed.")

        self.model_name = model_name.strip()
        if not self.model_name:
            raise ValueError("model_name must not be empty")

        self.temperature = temperature
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def _build_prompt(
        self,
        question: str,
        gt_answer: str,
        pred_answer: str,
        task_type: str,
        domain: str,
    ) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gt_answer=gt_answer,
            pred_answer=pred_answer,
            task_type=task_type,
            domain=domain,
        )

    def _call_openai(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )

        return (resp.choices[0].message.content or "").strip()

    def _parse_judge_response(self, raw_text: str) -> tuple[int, str]:
        data: dict[str, Any] = {}

        try:
            data = json.loads(raw_text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = {}

        if not data:
            return 0, f"Judge output cannot be parsed as JSON: {raw_text[:200]}"

        raw_correct = data.get("correct", 0)
        if isinstance(raw_correct, str):
            raw_correct = raw_correct.strip().lower()
            correct = 1 if raw_correct in {"1", "true", "yes"} else 0
        else:
            correct = 1 if int(raw_correct) == 1 else 0

        reason = str(data.get("reason", ""))
        return correct, reason

    def evaluate(
        self,
        image_id: int,
        question: str,
        gt_answer: str,
        pred_answer: str,
        task_type: str,
        domain: str,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(
            question=question,
            gt_answer=gt_answer,
            pred_answer=pred_answer,
            task_type=task_type,
            domain=domain,
        )

        _ = image_id
        raw_text = self._call_openai(prompt)

        correct, reason = self._parse_judge_response(raw_text)

        return {
            "correct": int(correct),
            "reason": reason,
            "judge_model": self.model_name,
        }
