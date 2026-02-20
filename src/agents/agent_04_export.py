"""
Agent 4: Integration & Export

Purpose: Generate Excel/CSV/JSON reports from evaluations
Exports:
- Excel (.xlsx) with Summary + Details sheets
- CSV (summary + details)
- JSON (full evaluation data)
- Optional webhook notification
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, date
import hashlib
import hmac
import json
import os
import random
import time
import logging

import pandas as pd
import httpx

from utils import json_serializer as _json_serializer  # HIGH-6: shared serializer

logger = logging.getLogger("qa_system.agents")


# HIGH-6: _json_serializer removed — now imported from utils


class IntegrationAgent:
    """Agent: Export results to Excel/CSV/JSON and optional webhook"""

    def __init__(self, output_folder: str = "data/evaluations", webhook_url: str = "",
                 webhook_secret: str = ""):
        """
        Initialize integration agent.

        Usage:
            agent_integration = IntegrationAgent(
                output_folder="data/evaluations",
                webhook_url=os.environ.get('WEBHOOK_URL', ''),
                webhook_secret=os.environ.get('WEBHOOK_SECRET', ''),
            )
        """
        self.output_folder = Path(output_folder)
        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"IntegrationAgent initialized | Output: {self.output_folder}")

    def export_all(self, evaluations: List[Dict], criteria_ref: Dict) -> Dict[str, str]:
        """
        Export evaluations to Excel + CSV + JSON.
        Returns dict of filepaths: {"excel": ..., "csv_summary": ..., "csv_details": ..., "json": ...}
        """
        # HIGH-9: Guard against empty evaluations list
        if not evaluations:
            logger.warning("export_all called with empty evaluations list")
            return {}

        # Use single timestamp for all exports to avoid filename drift
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = self.output_folder / f"QM_{timestamp}"

        # Build summary DataFrame
        summary_rows = []
        for e in evaluations:
            score_data = e.get("score_data", {})
            cat_scores = score_data.get("category_scores", {})
            breakdown = score_data.get("score_breakdown", {})

            row = {
                "File": e.get("filename", ""),
                "Type": e.get("call_type", ""),
                "Score": e.get("overall_score", 0),
            }
            # Dynamic category columns from criteria_ref
            all_categories = sorted(set(c.get("category", "unknown") for c in criteria_ref.values()))
            for cat in all_categories:
                row[cat] = cat_scores.get(cat, {}).get("score", 0)
            row["YES"] = breakdown.get("yes_count", 0)
            row["PARTIAL"] = breakdown.get("partial_count", 0)
            row["NO"] = breakdown.get("no_count", 0)
            row["Cost"] = e.get("cost_usd", 0)

            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)

        # Build detail DataFrame
        detail_rows = []
        for e in evaluations:
            filename = e.get("filename", "")
            criteria = e.get("criteria", {})
            for key, val in criteria.items():
                criteria_def = criteria_ref.get(key, {})
                detail_rows.append({
                    "File": filename,
                    "Category": criteria_def.get("category", "unknown"),
                    "Criterion": key,
                    "Weight": criteria_def.get("weight", 1.0),
                    "Score": val.get("score", ""),
                    "Evidence": val.get("evidence", "")
                })

        df_detail = pd.DataFrame(detail_rows)

        files = {}

        # Excel with 2 sheets
        excel_path = f"{base}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as w:
            df_summary.to_excel(w, sheet_name='Summary', index=False)
            df_detail.to_excel(w, sheet_name='Details', index=False)
            # MED-5: Auto-fit column widths for readability
            for sheet_name in ['Summary', 'Details']:
                ws = w.sheets[sheet_name]
                for col in ws.iter_cols(min_row=1, max_row=1):
                    for cell in col:
                        col_letter = cell.column_letter
                        max_width = max(
                            len(str(cell.value or "")),
                            max((len(str(c.value or "")) for c in ws[col_letter]), default=0),
                        )
                        ws.column_dimensions[col_letter].width = min(max_width + 2, 60)
        files["excel"] = excel_path
        logger.info(f"Excel: {excel_path}")

        # CSV files
        csv_summary_path = f"{base}_summary.csv"
        csv_details_path = f"{base}_details.csv"
        df_summary.to_csv(csv_summary_path, index=False)
        df_detail.to_csv(csv_details_path, index=False)
        files["csv_summary"] = csv_summary_path
        files["csv_details"] = csv_details_path
        logger.info(f"CSV: {csv_summary_path}, {csv_details_path}")

        # JSON (reuse same timestamp)
        json_path = self.export_json(
            evaluations,
            evaluations[0].get("model_used", "unknown") if evaluations else "unknown",
            timestamp=timestamp,
        )
        files["json"] = json_path

        # Webhook (if configured)
        if self.webhook_url:
            # MED-17: Log webhook dispatch (without exposing URL or payload data)
            logger.info("Sending webhook to configured URL (contains score/cost data)")
            total_cost = sum(e.get("cost_usd", 0) for e in evaluations)
            avg_score = sum(e.get("overall_score", 0) for e in evaluations) / len(evaluations) if evaluations else 0
            # HIGH-5: Only send filenames, not full filesystem paths
            safe_files = {k: os.path.basename(str(v)) for k, v in files.items()}
            self.send_webhook({
                "event": "qa_evaluation_complete",
                "calls_processed": len(evaluations),
                "average_score": round(avg_score, 1),
                "total_cost_usd": round(total_cost, 4),
                "files": safe_files
            })

        logger.info(f"Exported: {excel_path}")
        return files

    def export_json(self, evaluations: List[Dict], model_name: str,
                    timestamp: Optional[str] = None) -> str:
        """Export evaluations to JSON file. Returns filepath."""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = str(self.output_folder / f"QM_{timestamp}.json")

        total_cost = sum(e.get("cost_usd", 0) for e in evaluations)

        output = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "model": model_name,
                "calls": len(evaluations),
                "cost_usd": round(total_cost, 4)
            },
            "evaluations": evaluations
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=_json_serializer)

        logger.info(f"JSON: {json_path}")
        return json_path

    def send_webhook(self, payload: dict) -> bool:
        """POST payload to webhook URL with retry + exponential backoff.

        #8: Adds HMAC-SHA256 signature when webhook_secret is configured.
        HIGH-10: Uses shorter timeouts and caps retry delay.
        MED-NEW-9: Exponential backoff between retries.
        """
        if not self.webhook_url:
            return False

        max_attempts = 3
        body = json.dumps(payload, default=_json_serializer, sort_keys=True)

        headers = {"Content-Type": "application/json"}
        # #8: HMAC signing for webhook authentication
        if self.webhook_secret:
            signature = hmac.new(
                self.webhook_secret.encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Signature-256"] = f"sha256={signature}"

        for attempt in range(max_attempts):
            try:
                with httpx.Client(timeout=5) as client:
                    response = client.post(
                        self.webhook_url,
                        content=body,
                        headers=headers,
                    )
                    if 200 <= response.status_code < 300:
                        logger.info(f"Webhook sent: {response.status_code}")
                        return True
                    logger.warning(f"Webhook response: {response.status_code}")
            except Exception as e:
                logger.warning(f"Webhook attempt {attempt+1} failed: {e}")

            if attempt < max_attempts - 1:
                backoff = min(2 ** (attempt + 1), 8)  # MED-NEW-9: 2, 4, (capped 8)
                # MED-12: Add jitter (±25%) to prevent thundering-herd
                jitter = backoff * random.uniform(-0.25, 0.25)
                delay = max(0.1, backoff + jitter)
                logger.debug(f"Webhook retry backoff: {delay:.2f}s (base {backoff}s)")
                time.sleep(delay)

        logger.error("Webhook failed after 3 attempts")
        return False
