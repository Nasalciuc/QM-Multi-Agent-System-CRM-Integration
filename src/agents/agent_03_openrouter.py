"""
Agent 3: OpenRouter QA Evaluation

Purpose: Evaluate call transcripts against 24 QA criteria using LLMs

Dependencies:
    - httpx (for API calls)
    - pyyaml (for loading criteria)

API Details:
    - Base URL: https://openrouter.ai/api/v1
    - Endpoint: POST /chat/completions (OpenAI-compatible)
    - Auth: Bearer token in Authorization header

Model Fallback Chain:
    1. anthropic/claude-sonnet-4-20250514   (best quality)
    2. google/gemini-2.0-flash-001          (good fallback)
    3. meta-llama/llama-3.1-70b-instruct    (budget fallback)

QA Criteria: 24 total, 4 categories x 6 each
    - Phone Skills (PS-01 to PS-06)
    - Sales Techniques (ST-01 to ST-06)
    - Urgency & Closing (UC-01 to UC-06)
    - Soft Skills (SS-01 to SS-06)

TODO:
    - evaluate(transcript, criteria) -> dict
    - _build_messages(transcript, criteria) -> list[dict]
    - _call_openrouter(messages, model) -> dict
    - _call_with_fallback(messages) -> dict
    - _parse_response(content) -> dict
    - _extract_json(text) -> dict  (handle markdown code blocks)
    - load_criteria(yaml_path) -> dict
"""

import os
import json
from typing import Optional

# import httpx   # TODO: Uncomment when implementing


class OpenRouterAgent:
    """
    Evaluates call transcripts against 24 QA criteria via LLM.

    Usage:
        agent = OpenRouterAgent(config)
        criteria = agent.load_criteria("config/qa_criteria.yaml")
        result = agent.evaluate(transcript_text, criteria)
        print(f"Overall score: {result['overall_score']}")

    Config keys (from config/agents.yaml):
        openrouter.primary_model    -> "anthropic/claude-sonnet-4-20250514"
        openrouter.fallback_model   -> "google/gemini-2.0-flash-001"
        openrouter.strict_mode      -> True

    Env vars (from .env):
        OPENROUTER_API_KEY
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Model fallback chain (try in order)
    MODELS = [
        "anthropic/claude-sonnet-4-20250514",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.1-70b-instruct",
    ]

    def __init__(self, config: dict):
        """
        TODO:
            - Read OPENROUTER_API_KEY from os.environ
            - Store model list from config (or use defaults above)
            - Set temperature=0.1 (low for consistency)
            - Set max_tokens=4096
        """
        # TODO: Implement
        pass

    def evaluate(self, transcript: str, criteria: dict) -> dict:
        """
        Evaluate a transcript against all 24 QA criteria.

        TODO:
            1. Build system + user messages
            2. Call OpenRouter with fallback
            3. Parse JSON response
            4. Validate all 24 scores present and in range 1-5
            5. Calculate category averages
            6. Calculate overall score
            7. Return result dict:
                {
                    'criterion_scores': {
                        'PS-01': {'score': 4, 'justification': '...', 'evidence': '...'},
                        'PS-02': {...},
                        ... (all 24)
                    },
                    'category_averages': {
                        'phone_skills': 3.8,
                        'sales_techniques': 3.2,
                        'urgency_closing': 2.5,
                        'soft_skills': 4.0
                    },
                    'overall_score': 3.4,
                    'strengths': ['...', '...', '...'],
                    'improvements': ['...', '...', '...'],
                    'coaching_notes': '...',
                    'model_used': 'anthropic/claude-sonnet-4-20250514',
                    'tokens_used': {'prompt': 1200, 'completion': 800},
                    'cost_usd': 0.012
                }
        """
        # TODO: Implement
        pass

    def _build_messages(self, transcript: str, criteria: dict) -> list[dict]:
        """
        Build the LLM prompt messages.

        TODO:
            System message:
                - You are an expert call center QA analyst
                - Evaluate against the provided criteria
                - Score each 1-5 with justification
                - Return ONLY valid JSON

            User message:
                - Include the full transcript
                - Include all 24 criteria with scoring rubrics
                - Specify the exact JSON output schema

            Return: [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ]
        """
        # TODO: Implement
        pass

    def _call_openrouter(self, messages: list[dict], model: str) -> dict:
        """
        Call OpenRouter API with a specific model.

        TODO:
            Request:
                POST https://openrouter.ai/api/v1/chat/completions
                Headers:
                    Authorization: Bearer {api_key}
                    Content-Type: application/json
                Body:
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 4096
                    }

            - Send with httpx.post()
            - Check status_code == 200
            - Return response.json()
            - On error: raise with details
        """
        # TODO: Implement
        pass

    def _call_with_fallback(self, messages: list[dict]) -> dict:
        """
        Try models in order until one succeeds.

        TODO:
            - For each model in self.MODELS:
                - Try _call_openrouter(messages, model)
                - On success: return response (include model_used)
                - On failure: log warning, try next
            - If all fail: raise error with all failure details
        """
        # TODO: Implement
        pass

    def _parse_response(self, raw_response: dict) -> dict:
        """
        Extract evaluation from LLM response.

        TODO:
            - Get content = response['choices'][0]['message']['content']
            - Extract JSON from content (_extract_json)
            - Validate structure
            - Return parsed evaluation dict
        """
        # TODO: Implement
        pass

    def _extract_json(self, text: str) -> dict:
        """
        Extract JSON from LLM response text.

        LLMs often wrap JSON in markdown code blocks.

        TODO:
            - Try json.loads(text) directly
            - Try extracting from ```json ... ``` blocks
            - Try finding first { to last } in text
            - Raise ValueError if no valid JSON found
        """
        # TODO: Implement
        pass

    def load_criteria(self, yaml_path: str = "config/qa_criteria.yaml") -> dict:
        """
        Load QA criteria from YAML file.

        TODO:
            - Open and parse YAML file
            - Validate 4 categories x 6 criteria = 24 total
            - Return criteria dict
        """
        # TODO: Implement
        pass
