"""Whale tracking agent — mirrors agents/templates/multimodal.py self-programming pattern."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any

import openai
from openai import OpenAI as OpenAIClient

from ..agent import SolanaAgent
from ..models import ActionParams, ChainState, SolanaAction

logger = logging.getLogger(__name__)


class WhaleTracker(SolanaAgent):
    """An agent that monitors whale wallets and suggests copy-trades.

    Uses the MultiModalLLM self-programming pattern: the LLM maintains
    and rewrites its own knowledge base of whale behavior patterns
    after each observation cycle.

    The agent watches a set of whale wallets, analyzes their moves,
    and proposes (in dry-run) corresponding trades for the user's wallet.
    """

    MAX_ACTIONS: int = 30
    LOOP_DELAY: float = 15.0  # slower cycle — whale watching
    MODEL: str = "gpt-4o-mini"
    MESSAGE_LIMIT: int = 8

    # Whale wallets to track (can be set via constructor or env var)
    WHALE_WALLETS: list[str] = []

    MAX_SOL_PER_TX: float = 0.1  # conservative for copy-trades
    MIN_SOL_RESERVE: float = 0.05

    def __init__(self, *args: Any, whale_wallets: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.WHALE_WALLETS = whale_wallets or self._load_whale_wallets()
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.messages: list[dict[str, Any]] = []

        # Self-programming memory — LLM updates this after each cycle
        self._whale_memory = textwrap.dedent("""\
            ## Tracked Whales
            No whale activity observed yet.

            ## Observed Patterns
            No patterns identified yet.

            ## Copy-Trade Strategy
            - Only copy-trade with high conviction
            - Never exceed MAX_SOL_PER_TX limit
            - Wait for confirmation before acting
        """)

    @property
    def name(self) -> str:
        return "whale_tracker"

    def _load_whale_wallets(self) -> list[str]:
        """Load whale wallet addresses from env var."""
        raw = os.environ.get("WHALE_WALLETS", "")
        return [w.strip() for w in raw.split(",") if w.strip()]

    def is_done(self, states: list[ChainState]) -> bool:
        if not states:
            return False
        if not self.WHALE_WALLETS:
            logger.warning("No whale wallets configured, stopping")
            return True
        return states[-1].sol_balance < 0.01

    def choose_action(
        self, states: list[ChainState]
    ) -> tuple[SolanaAction, ActionParams]:
        """Observe whale activity, analyze patterns, suggest copy-trades."""
        latest = states[-1]

        # Fetch whale wallet states
        whale_data = self._fetch_whale_states()

        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(latest, whale_data)

        # Phase 1: Analyze whale activity + update memory
        analysis_messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": analysis_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=analysis_messages,
            )
        except openai.BadRequestError as e:
            logger.error(f"OpenAI error: {e}")
            return SolanaAction.OBSERVE, ActionParams(reasoning="LLM error")

        analysis_text = response.choices[0].message.content or ""
        logger.info(f"Whale analysis: {analysis_text[:200]}")

        # Self-programming: update memory with analysis
        if "---" in analysis_text:
            before, _, after = analysis_text.partition("---")
            if after.strip():
                self._whale_memory = after.strip()
                logger.info("Updated whale memory from LLM analysis")

        # Phase 2: Decide action based on analysis
        action_messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": f"Based on this analysis:\n\n{analysis_text}\n\n"
                + f"Current memory:\n{self._whale_memory}\n\n"
                + "Choose one action.",
            },
        ]

        tools = self._build_tools()
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=action_messages,
                tools=tools,
                tool_choice="required",
            )
        except openai.BadRequestError as e:
            logger.error(f"OpenAI error in action phase: {e}")
            return SolanaAction.OBSERVE, ActionParams(reasoning="LLM error in action phase")

        msg = response.choices[0].message
        if not msg.tool_calls:
            return SolanaAction.OBSERVE, ActionParams(reasoning="No action from whale analysis")

        tool_call = msg.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except json.JSONDecodeError:
            args = {}

        # Record whale intelligence
        if hasattr(self, "recorder"):
            self.recorder.record({
                "type": "whale_intelligence",
                "whale_data": {w[:8]: "tracked" for w in self.WHALE_WALLETS},
                "memory_snapshot": self._whale_memory[:500],
                "action": tool_call.function.name,
            })

        return self._parse_action(tool_call.function.name, args)

    def _fetch_whale_states(self) -> dict[str, dict[str, Any]]:
        """Fetch basic state info for each whale wallet."""
        whale_data: dict[str, dict[str, Any]] = {}
        for wallet in self.WHALE_WALLETS[:5]:  # limit to 5 whales
            try:
                state = self.env.get_state(wallet)
                whale_data[wallet] = {
                    "sol_balance": state.sol_balance,
                    "token_count": len(state.token_balances),
                    "top_tokens": [
                        {"symbol": t.symbol, "amount": t.amount, "mint": t.mint[:12]}
                        for t in state.token_balances[:5]
                    ],
                    "recent_txns": [
                        {"type": t.type, "desc": t.description[:60]}
                        for t in state.recent_transactions[:3]
                    ],
                }
            except Exception as e:
                logger.warning(f"Failed to fetch whale {wallet[:8]}: {e}")
                whale_data[wallet] = {"error": str(e)}
        return whale_data

    def _system_prompt(self) -> str:
        return textwrap.dedent("""\
            You are a whale-tracking intelligence agent on Solana.
            
            Your role:
            1. Monitor whale wallet activity (large holders, smart money)
            2. Identify patterns in their trading behavior
            3. Assess whether their moves are worth copying
            4. Suggest copy-trades with explicit reasoning
            
            After each analysis, update the knowledge base below the "---"
            separator to preserve learnings for future cycles.
            
            Be very selective — most whale moves should NOT be copied.
            Only suggest copy-trades when you see strong conviction signals.
        """)

    def _build_analysis_prompt(
        self, our_state: ChainState, whale_data: dict[str, dict[str, Any]]
    ) -> str:
        lines = [
            "# Whale Tracking Report",
            "",
            "## Our Wallet",
            f"SOL: {our_state.sol_balance:.4f}",
            f"Tokens: {len(our_state.token_balances)}",
            "",
            "## Whale Activity",
        ]
        for wallet, data in whale_data.items():
            lines.append(f"\n### Whale {wallet[:8]}...")
            if "error" in data:
                lines.append(f"  Error: {data['error']}")
                continue
            lines.append(f"  SOL: {data.get('sol_balance', 0):.4f}")
            lines.append(f"  Token count: {data.get('token_count', 0)}")
            if data.get("top_tokens"):
                lines.append("  Top holdings:")
                for t in data["top_tokens"]:
                    lines.append(f"    {t['symbol']}: {t['amount']:.4f}")
            if data.get("recent_txns"):
                lines.append("  Recent activity:")
                for tx in data["recent_txns"]:
                    lines.append(f"    [{tx['type']}] {tx['desc']}")

        lines.extend([
            "",
            "## Current Knowledge Base",
            self._whale_memory,
            "",
            "Analyze the whale activity above. Provide your analysis, then after '---' "
            "provide the updated knowledge base.",
        ])
        return "\n".join(lines)

    def _parse_action(
        self, action_name: str, args: dict[str, Any]
    ) -> tuple[SolanaAction, ActionParams]:
        reasoning = args.get("reasoning", "")
        if action_name == "observe":
            return SolanaAction.OBSERVE, ActionParams(reasoning=reasoning)
        elif action_name == "copy_trade_swap":
            return SolanaAction.SWAP, ActionParams(
                token_in_mint=args.get("token_in_mint", ""),
                token_out_mint=args.get("token_out_mint", ""),
                amount_in=args.get("amount", 0.0),
                reasoning=reasoning,
            )
        else:
            return SolanaAction.OBSERVE, ActionParams(reasoning=f"Unknown: {action_name}")

    def _build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "observe",
                    "description": "Continue monitoring whales without acting. Use when no clear copy-trade signal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string", "description": "Why no copy-trade now"},
                        },
                        "required": ["reasoning"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_trade_swap",
                    "description": "Copy a whale's swap. Only use with high conviction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_in_mint": {"type": "string", "description": "Token to sell (mint address)"},
                            "token_out_mint": {"type": "string", "description": "Token to buy (mint address)"},
                            "amount": {"type": "number", "description": "Amount to swap"},
                            "whale_wallet": {"type": "string", "description": "Which whale we're copying"},
                            "reasoning": {"type": "string", "description": "Why this copy-trade is worth it"},
                        },
                        "required": ["token_in_mint", "token_out_mint", "amount", "whale_wallet", "reasoning"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]
