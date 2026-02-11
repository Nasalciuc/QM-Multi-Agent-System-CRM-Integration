"""
Agent 4: Integration & Delivery

Purpose: Send QA results to target system and export reports

Dependencies:
    - httpx (for webhook calls)

Delivery options:
    1. Webhook POST (primary) - send JSON to configured URL
    2. JSON file export - save to data/evaluations/
    3. CSV export - save to data/exports/

TODO:
    - deliver(evaluation, call_metadata) -> bool
    - send_webhook(payload) -> bool
    - save_evaluation(evaluation, call_id) -> str  (filepath)
    - export_csv(evaluations) -> str  (filepath)
    - _retry_with_backoff(func, max_retries) -> result
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

# import httpx   # TODO: Uncomment when implementing


class IntegrationAgent:
    """
    Delivers QA results to target systems.

    Usage:
        agent = IntegrationAgent(config)
        agent.deliver(evaluation, call_metadata)
        agent.export_csv(all_evaluations)

    Config keys (from config/agents.yaml):
        integration.webhook_url     -> from .env WEBHOOK_URL
        integration.retry_attempts  -> 3

    Env vars (from .env):
        WEBHOOK_URL  (optional - if empty, only saves locally)
    """

    def __init__(self, config: dict):
        """
        TODO:
            - Read WEBHOOK_URL from os.environ (may be empty)
            - Store retry_attempts from config
            - Set output dirs: data/evaluations/, data/exports/
        """
        # TODO: Implement
        pass

    def deliver(self, evaluation: dict, call_metadata: dict) -> bool:
        """
        Deliver evaluation results.

        TODO:
            1. Save evaluation JSON locally (always)
            2. If webhook_url configured: send via webhook
            3. Return True if all deliveries succeeded
        """
        # TODO: Implement
        pass

    def send_webhook(self, payload: dict) -> bool:
        """
        POST evaluation to webhook URL.

        TODO:
            - Build payload with evaluation + metadata
            - POST to self.webhook_url with JSON body
            - Retry up to self.retry_attempts on failure
            - Use exponential backoff: 1s, 2s, 4s
            - Return True on success (2xx response)
            - Log errors on failure
        """
        # TODO: Implement
        pass

    def save_evaluation(self, evaluation: dict, call_id: str) -> str:
        """
        Save evaluation as JSON file.

        TODO:
            - Path: data/evaluations/{call_id}.json
            - Write with json.dump(indent=2)
            - Return filepath
        """
        # TODO: Implement
        pass

    def export_csv(self, evaluations: list[dict], filename: str = "qa_results.csv") -> str:
        """
        Export all evaluations to a single CSV file.

        TODO:
            Columns:
                call_id, date, duration, direction,
                overall_score,
                phone_skills_avg, sales_techniques_avg,
                urgency_closing_avg, soft_skills_avg,
                PS-01, PS-02, PS-03, PS-04, PS-05, PS-06,
                ST-01, ST-02, ST-03, ST-04, ST-05, ST-06,
                UC-01, UC-02, UC-03, UC-04, UC-05, UC-06,
                SS-01, SS-02, SS-03, SS-04, SS-05, SS-06,
                model_used, cost_usd

            - Write to data/exports/{filename}
            - Return filepath
        """
        # TODO: Implement
        pass

    def _retry_with_backoff(self, func, max_retries: int = 3):
        """
        Retry a function with exponential backoff.

        TODO:
            - Try func()
            - On failure: wait 2^attempt seconds, retry
            - After max_retries: return None / raise
        """
        # TODO: Implement
        pass
