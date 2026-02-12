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
import json
import os
import time
import logging

import pandas as pd
import httpx

logger = logging.getLogger("qa_system.agents")


def _json_serializer(obj):
    """Explicit JSON serializer — replaces dangerous `default=str`.

    CRIT-4: Only handles known types; raises TypeError for unexpected objects.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Non-serializable object: {type(obj).__name__}")


class IntegrationAgent:
    """Agent: Export results to Excel/CSV/JSON and optional webhook"""

    def __init__(self, output_folder: str = "data/evaluations", webhook_url: str = ""):
        """
        Initialize integration agent.

        Usage:
            agent_integration = IntegrationAgent(
                output_folder="data/evaluations",
                webhook_url=os.environ.get('WEBHOOK_URL', '')
            )
        """
        self.output_folder = Path(output_folder)
        self.webhook_url = webhook_url
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"IntegrationAgent initialized | Output: {self.output_folder}")

    def export_all(self, evaluations: List[Dict], criteria_ref: Dict) -> Dict[str, str]:
        """
        Export evaluations to Excel + CSV + JSON.
        Returns dict of filepaths: {"excel": ..., "csv_summary": ..., "csv_details": ..., "json": ...}
        """
        # Use single timestamp for all exports to avoid filename drift
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = self.output_folder / f"QM_{timestamp}"

        # Build summary DataFrame
        summary_rows = []
        for e in evaluations:
            score_data = e.get("score_data", {})
            cat_scores = score_data.get("category_scores", {})
            breakdown = score_data.get("score_breakdown", {})

            summary_rows.append({
                "File": e.get("filename", ""),
                "Type": e.get("call_type", ""),
                "Score": e.get("overall_score", 0),
                "Phone": cat_scores.get("phone_skills", {}).get("score", 0),
                "Sales": cat_scores.get("sales_techniques", {}).get("score", 0),
                "Closing": cat_scores.get("urgency_closing", {}).get("score", 0),
                "Soft": cat_scores.get("soft_skills", {}).get("score", 0),
                "YES": breakdown.get("yes_count", 0),
                "PARTIAL": breakdown.get("partial_count", 0),
                "NO": breakdown.get("no_count", 0),
                "Cost": e.get("cost_usd", 0)
            })

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

        print(f"\n  Exported: {excel_path}")
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
        """POST payload to webhook URL with retry. Returns True on success.

        HIGH-10: Uses shorter timeouts and caps retry delay to minimize blocking.
        """
        if not self.webhook_url:
            return False

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with httpx.Client(timeout=5) as client:
                    response = client.post(self.webhook_url, json=payload)
                    if 200 <= response.status_code < 300:
                        logger.info(f"Webhook sent: {response.status_code}")
                        return True
                    logger.warning(f"Webhook response: {response.status_code}")
            except Exception as e:
                logger.warning(f"Webhook attempt {attempt+1} failed: {e}")

            if attempt < max_attempts - 1:
                time.sleep(min(2 ** attempt, 4))  # cap at 4s

        logger.error("Webhook failed after 3 attempts")
        return False
