"""Portfolio monitoring agent — mirrors agents/templates/reasoning_agent.py."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Optional

import openai
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel, Field

from ..agent import SolanaAgent
from ..models import ActionParams, ChainState, SolanaAction

logger = logging.getLogger(__name__)


class PortfolioAnalysis(BaseModel):
    """Structured analysis output from the portfolio monitor."""

    action: str = Field(description="Action to take: observe, swap, transfer_sol, transfer_token")
    reasoning: str = Field(description="Detailed reasoning for the decision")
    risk_assessment: str = Field(description="Assessment of current portfolio risk")
    recommendation: str = Field(description="Brief actionable recommendation")
    confidence: float = Field(description="Confidence 0.0-1.0 in this action", ge=0.0, le=1.0)


class PortfolioMonitor(SolanaAgent):
    """A reasoning agent that monitors portfolio health and suggests rebalancing.

    Uses structured LLM output to track portfolio value, concentration risk,
    and provide rebalancing recommendations with explicit confidence scores.
    """

    MAX_ACTIONS: int = 30
    LOOP_DELAY: float = 10.0  # slower cycle — monitoring, not trading
    MODEL: str = "gpt-4o-mini"
    MIN_CONFIDENCE: float = 0.7  # only act on high-confidence recommendations

    MAX_SOL_PER_TX: float = 0.1
    MIN_SOL_RESERVE: float = 0.05

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analyses: list[PortfolioAnalysis] = []
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

    @property
    def name(self) -> str:
        return "portfolio_monitor"

    def is_done(self, states: list[ChainState]) -> bool:
        if not states:
            return False
        # Stop if balance is critically low
        return states[-1].sol_balance < 0.01

    def choose_action(
        self, states: list[ChainState]
    ) -> tuple[SolanaAction, ActionParams]:
        """Use structured LLM output to analyze portfolio and decide action."""
        latest = states[-1]

        # Build context
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._build_analysis_prompt(states)},
        ]

        # Call LLM with tools that match our structured output
        tools = self._build_tools()
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
        except openai.BadRequestError as e:
            logger.error(f"OpenAI error: {e}")
            return SolanaAction.OBSERVE, ActionParams(reasoning="LLM error")

        # Parse response
        msg = response.choices[0].message
        if not msg.tool_calls:
            return SolanaAction.OBSERVE, ActionParams(reasoning="No analysis returned")

        tool_call = msg.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments)
            analysis = PortfolioAnalysis(
                action=tool_call.function.name,
                reasoning=args.get("reasoning", ""),
                risk_assessment=args.get("risk_assessment", ""),
                recommendation=args.get("recommendation", ""),
                confidence=args.get("confidence", 0.5),
            )
        except Exception as e:
            logger.warning(f"Failed to parse analysis: {e}")
            return SolanaAction.OBSERVE, ActionParams(reasoning="Parse error")

        self.analyses.append(analysis)
        logger.info(
            f"Analysis: {analysis.action} (confidence: {analysis.confidence:.0%}) "
            f"| {analysis.recommendation}"
        )

        # Only act on high-confidence recommendations
        if analysis.confidence < self.MIN_CONFIDENCE:
            logger.info(
                f"Confidence {analysis.confidence:.0%} below threshold "
                f"{self.MIN_CONFIDENCE:.0%}, observing instead"
            )
            return SolanaAction.OBSERVE, ActionParams(
                reasoning=f"Low confidence ({analysis.confidence:.0%}): {analysis.reasoning}"
            )

        # Map analysis to action
        action_map = {
            "observe": SolanaAction.OBSERVE,
            "swap": SolanaAction.SWAP,
            "transfer_sol": SolanaAction.TRANSFER_SOL,
            "transfer_token": SolanaAction.TRANSFER_TOKEN,
        }
        action = action_map.get(analysis.action, SolanaAction.OBSERVE)

        params = ActionParams(
            reasoning=analysis.reasoning,
            token_in_mint=args.get("token_in_mint"),
            token_out_mint=args.get("token_out_mint"),
            amount_in=args.get("amount"),
            recipient=args.get("recipient"),
            sol_amount=args.get("amount"),
        )

        return action, params

    def _system_prompt(self) -> str:
        return textwrap.dedent("""\
            You are a portfolio monitoring agent on Solana. Your role is to:
            
            1. Analyze the current portfolio composition and health
            2. Identify concentration risks or imbalances
            3. Track P&L trends across observation windows
            4. Recommend rebalancing when appropriate
            
            Be conservative. Only recommend actions with high confidence.
            Prefer observing over acting when uncertain.
            
            Always assess risk and provide a confidence score (0.0-1.0).
        """)

    def _build_analysis_prompt(self, states: list[ChainState]) -> str:
        latest = states[-1]
        lines = [
            "# Portfolio Analysis Request",
            "",
            f"## Current State (Observation #{len(states)})",
            latest.summary(),
            "",
        ]

        # Show trend if we have history
        if len(states) >= 2:
            prev = states[-2]
            sol_delta = latest.sol_balance - prev.sol_balance
            usd_delta = latest.total_usd_value - prev.total_usd_value
            lines.extend([
                "## Changes Since Last Observation",
                f"  SOL: {sol_delta:+.4f}",
                f"  USD: ${usd_delta:+,.2f}",
                "",
            ])

        # Previous analyses
        if self.analyses:
            lines.append("## Recent Analysis History")
            for a in self.analyses[-3:]:
                lines.append(
                    f"  [{a.action}] confidence={a.confidence:.0%}: {a.recommendation[:80]}"
                )
            lines.append("")

        lines.append("Analyze the portfolio and choose one action.")
        return "\n".join(lines)

    def _build_tools(self) -> list[dict[str, Any]]:
        """Tools for portfolio actions with reasoning fields."""
        base_props = {
            "reasoning": {"type": "string", "description": "Detailed analysis reasoning"},
            "risk_assessment": {"type": "string", "description": "Current risk assessment"},
            "recommendation": {"type": "string", "description": "Brief actionable recommendation"},
            "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
        }
        base_required = ["reasoning", "risk_assessment", "recommendation", "confidence"]

        return [
            {
                "type": "function",
                "function": {
                    "name": "observe",
                    "description": "Continue monitoring without action. Use when portfolio is healthy or uncertain.",
                    "parameters": {
                        "type": "object",
                        "properties": base_props,
                        "required": base_required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "swap",
                    "description": "Rebalance by swapping tokens. Use to reduce concentration risk.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            **base_props,
                            "token_in_mint": {"type": "string", "description": "Token to sell"},
                            "token_out_mint": {"type": "string", "description": "Token to buy"},
                            "amount": {"type": "number", "description": "Amount to swap"},
                        },
                        "required": [*base_required, "token_in_mint", "token_out_mint", "amount"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]
