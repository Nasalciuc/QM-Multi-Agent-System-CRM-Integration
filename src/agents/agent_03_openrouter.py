"""
Agent 3: Quality Management Evaluation

Purpose: Evaluate call transcripts using LLM (OpenAI/OpenRouter)
Style: Matches my existing QualityManagementAgent exactly

Model: gpt-4o-2024-11-20 (via OpenAI or OpenRouter)
Criteria: 24 total across 4 categories
Scoring: YES / PARTIAL / NO / N/A per criterion
Output: JSON with scores, evidence, assessment

TODO:
- Copy my working QualityManagementAgent class
- All 24 EVALUATION_CRITERIA are already defined below
- Implement evaluate_call() and calculate_score()
"""

from typing import Dict, List, Tuple
import json
import time
import re


class QualityManagementAgent:
    """Agent profesional pentru evaluarea calitatii apelurilor de vanzari"""

    OPENAI_MODEL = "gpt-4o-2024-11-20"

    PRICING = {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00
    }

    # ═══════════════════════════════════════════════════════════════
    # ALL 24 EVALUATION CRITERIA (from my working notebook)
    # ═══════════════════════════════════════════════════════════════

    EVALUATION_CRITERIA = {
        # --- PHONE SKILLS (5 criteria) ---
        "greeting_prepared": {
            "description": "Was the agent prepared and timely greeted the client? Did the agent identify himself? Did the agent request the name of the client and use it? Did the agent thank the guest and ask how he could assist?",
            "category": "phone_skills",
            "weight": 1.0,
            "first_call_only": False
        },
        "contact_info": {
            "description": "Did the agent ask what will be the best way to reach the customer back? Did the agent spell back client's email and confirmed the phone number?",
            "category": "phone_skills",
            "weight": 1.0,
            "first_call_only": False
        },
        "source_discovery": {
            "description": "Did the agent ask how client found out about our website?",
            "category": "phone_skills",
            "weight": 0.5,
            "first_call_only": True
        },
        "first_time_customer": {
            "description": "Did the agent ask if this is customer's first interaction with someone from our company?",
            "category": "phone_skills",
            "weight": 1.0,
            "first_call_only": True
        },
        "advertised_offer": {
            "description": "Did the agent inform and clearly explain the details about our advertised offer from the website? (if applicable)",
            "category": "phone_skills",
            "weight": 0.8,
            "first_call_only": False
        },

        # --- SALES TECHNIQUES (8 criteria) ---
        "flexibility_dates": {
            "description": "Did the agent ask about flexibility (travel dates/airports/preferred departure/arrival times)?",
            "category": "sales_techniques",
            "weight": 1.0,
            "first_call_only": False
        },
        "airline_preferences": {
            "description": "Did the agent ask about airline/other preferences? Development in case customer has preferences: reason (miles or experience)",
            "category": "sales_techniques",
            "weight": 1.0,
            "first_call_only": False
        },
        "product_presentation": {
            "description": "Product presentation: Product features (value), Value first / price last",
            "category": "sales_techniques",
            "weight": 1.5,
            "first_call_only": False
        },
        "budget_inquiry": {
            "description": "Did the agent ask about clients budget? (price expectations)",
            "category": "sales_techniques",
            "weight": 1.0,
            "first_call_only": False
        },
        "online_prices": {
            "description": "Did the agent ask if customer checked online prices?",
            "category": "sales_techniques",
            "weight": 0.8,
            "first_call_only": False
        },
        "fare_guarantee": {
            "description": "Did the agent offer fare guarantee and the ability to beat the prices?",
            "category": "sales_techniques",
            "weight": 1.0,
            "first_call_only": False
        },
        "objection_handling": {
            "description": "Good objection handling (if applicable)",
            "category": "sales_techniques",
            "weight": 1.2,
            "first_call_only": False
        },
        "next_call_scheduling": {
            "description": "Did the agent set a time for the next conversation with the client? Did the agent call at the agreed time?",
            "category": "sales_techniques",
            "weight": 1.0,
            "first_call_only": False
        },

        # --- URGENCY & CLOSING (3 criteria) ---
        "urgency_creation": {
            "description": "Did the agent create the sense of urgency based on seat availability and advance purchase?",
            "category": "urgency_closing",
            "weight": 1.2,
            "first_call_only": False
        },
        "additional_assistance": {
            "description": "Offers additional assistance before closing the interaction (Mr. Smith, now that I have your flight package confirmed, is there any other information I can provide you?)",
            "category": "urgency_closing",
            "weight": 0.8,
            "first_call_only": False
        },
        "thank_you_closing": {
            "description": "Did the agent thank the caller for calling Buy Business Class?",
            "category": "urgency_closing",
            "weight": 0.8,
            "first_call_only": False
        },

        # --- SOFT SKILLS (8 criteria) ---
        "straight_line_system": {
            "description": "Did the agent use the straight line persuasion system? Having control over the call (every time the customer tries to take the conversation away from the sale by talking about something irrelevant you quickly bring it right back)",
            "category": "soft_skills",
            "weight": 1.5,
            "first_call_only": False
        },
        "consultative_expertise": {
            "description": "Did the agent sound consultative and display expertise?",
            "category": "soft_skills",
            "weight": 1.5,
            "first_call_only": False
        },
        "positive_tone": {
            "description": "Upbeat & Positive Tone, uses appropriate vocal inflection / Positive Verbiage using a variety of 'Power Words'",
            "category": "soft_skills",
            "weight": 1.0,
            "first_call_only": False
        },
        "authenticity": {
            "description": "Responds to guest questions and conversation with /Authenticity/Appropriateness",
            "category": "soft_skills",
            "weight": 1.0,
            "first_call_only": False
        },
        "pace_matching": {
            "description": "Pace is appropriate and easy to understand, and matches the guest pace (agent is receptive to guests pace)",
            "category": "soft_skills",
            "weight": 0.8,
            "first_call_only": False
        },
        "answers_questions": {
            "description": "Answers questions/or offers to locate unknown information",
            "category": "soft_skills",
            "weight": 1.0,
            "first_call_only": False
        },
        "call_leadership": {
            "description": "Leads the guest through the call, makes the transaction easy for the guest Talking vs Listening percentage",
            "category": "soft_skills",
            "weight": 1.2,
            "first_call_only": False
        },
        "accurate_information": {
            "description": "Provides accurate information",
            "category": "soft_skills",
            "weight": 1.5,
            "first_call_only": False
        }
    }

    def __init__(self, openai_client):
        """
        TODO:
        - Store OpenAI client
        - Print init info (model, criteria count)

        Usage:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                           base_url="https://openrouter.ai/api/v1")
            agent_qm = QualityManagementAgent(client)
        """
        self.client = openai_client
        self.model = self.OPENAI_MODEL
        # TODO: Implement

    def detect_call_type(self, filename: str) -> Tuple[bool, str]:
        """
        Detect if call is first or follow-up from filename.

        TODO:
        - Check filename for: "2nd", "second", "follow", "follow-up", "followup"
        - Return (is_followup: bool, call_type: str)

        My working code:
            filename_lower = filename.lower()
            follow_up_indicators = ["2nd", "second", "follow", "follow-up", "followup"]
            is_followup = any(indicator in filename_lower for indicator in follow_up_indicators)
            call_type = "Follow-up Call" if is_followup else "First Call"
            return is_followup, call_type
        """
        # TODO: Implement
        pass

    def evaluate_call(self, transcript: str, filename: str, max_retries: int = 2) -> Dict:
        """
        Evaluate a call transcript against all applicable criteria.

        TODO:
        1. Detect call type (first vs follow-up)
        2. Filter criteria: skip first_call_only criteria for follow-ups
        3. Build system prompt with:
           - Call type
           - Criteria count
           - All criteria descriptions
           - Expected JSON output format
        4. Build user prompt with transcript + criteria list
        5. Call OpenAI API:
           - model=self.model
           - temperature=0.1
           - max_tokens=4096
           - response_format={"type": "json_object"}
        6. Parse JSON response
        7. Validate:
           - All criteria present
           - Scores are YES/PARTIAL/NO/N/A
        8. Retry if validation fails (up to max_retries)
        9. Add metadata: call_type, model_used, tokens_used, cost_usd
        10. Return evaluation dict

        Expected response structure:
            {
                "criteria": {
                    "greeting_prepared": {"score": "YES", "evidence": "..."},
                    "contact_info": {"score": "PARTIAL", "evidence": "..."},
                    ...
                },
                "overall_assessment": "2-3 sentence summary",
                "strengths": ["...", "...", "..."],
                "improvements": ["...", "...", "..."],
                "critical_gaps": ["...", "..."]
            }
        """
        # TODO: Implement
        pass

    def calculate_score(self, evaluation: Dict) -> Dict:
        """
        Calculate overall score (0-100) from YES/PARTIAL/NO scores.

        TODO:
        - YES = 1.0 * weight
        - PARTIAL = 0.5 * weight
        - NO = 0.0 * weight
        - N/A = skip (don't count)
        - overall_score = (total_points / total_weight) * 100
        - Calculate per-category scores
        - Count YES/PARTIAL/NO/N/A breakdown

        Return:
            {
                "overall_score": 75.5,
                "total_points": 15.3,
                "total_weight": 20.3,
                "category_scores": {
                    "phone_skills": {"score": 80.0, "count": 5},
                    "sales_techniques": {"score": 70.0, "count": 8},
                    "urgency_closing": {"score": 66.7, "count": 3},
                    "soft_skills": {"score": 85.0, "count": 8}
                },
                "score_breakdown": {
                    "yes_count": 12,
                    "partial_count": 5,
                    "no_count": 3,
                    "na_count": 4
                }
            }
        """
        # TODO: Implement
        pass

    def calculate_listening_ratio(self, transcript: str) -> Dict[str, float]:
        """
        TODO:
        - Estimate agent vs client talking percentage
        - Return {"agent_percentage": 60.0, "client_percentage": 40.0, "total_words": N}
        """
        # TODO: Implement
        pass
