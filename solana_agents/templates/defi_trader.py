"""LLM-powered DeFi trading agent — mirrors agents/templates/llm_agents.py."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Optional

import openai
from openai import OpenAI as OpenAIClient

from ..agent import SolanaAgent
from ..models import ActionParams, ChainState, SolanaAction

logger = logging.getLogger(__name__)


class DeFiTrader(SolanaAgent):
    """An agent that uses an LLM to make DeFi trading decisions.

    Observes wallet state + token balances, presents them to the LLM,
    and lets the LLM choose actions via OpenAI tool calling.
    """

    MAX_ACTIONS: int = 20
    LOOP_DELAY: float = 5.0
    MODEL: str = "gpt-4o-mini"
    MESSAGE_LIMIT: int = 10

    MAX_SOL_PER_TX: float = 0.5
    MIN_SOL_RESERVE: float = 0.05

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages: list[dict[str, Any]] = []
        self.token_counter: int = 0

    @property
    def name(self) -> str:
        return f"defi_trader.{self.MODEL.replace('/', '-')}"

    def is_done(self, states: list[ChainState]) -> bool:
        """Stop if balance drops too low."""
        if not states:
            return False
        latest = states[-1]
        if latest.sol_balance < self.MIN_SOL_RESERVE:
            logger.info("Stopping: SOL balance below minimum reserve")
            return True
        return False

    def choose_action(
        self, states: list[ChainState]
    ) -> tuple[SolanaAction, ActionParams]:
        """Use LLM to decide the next trading action."""
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))
        latest = states[-1]

        # Build state context for LLM
        state_text = self._build_state_prompt(latest)

        if len(self.messages) == 0:
            # First turn: system + state
            self.messages.append({"role": "system", "content": self._system_prompt()})
            self.messages.append({"role": "user", "content": state_text})
        else:
            # Subsequent turns: append new state
            self.messages.append({"role": "user", "content": state_text})

        # Trim message history
        if len(self.messages) > self.MESSAGE_LIMIT:
            system = self.messages[0]
            self.messages = [system] + self.messages[-(self.MESSAGE_LIMIT - 1):]

        # Call LLM with tools
        tools = self._build_tools()
        try:
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages,
                tools=tools,
                tool_choice="required",
            )
        except openai.BadRequestError as e:
            logger.error(f"OpenAI error: {e}")
            return SolanaAction.OBSERVE, ActionParams(reasoning="LLM error, observing")

        # Track tokens
        if response.usage:
            self.token_counter += response.usage.total_tokens
            logger.info(f"Tokens used: {response.usage.total_tokens} (total: {self.token_counter})")

        # Parse tool call
        msg = response.choices[0].message
        self.messages.append(msg)

        if not msg.tool_calls:
            return SolanaAction.OBSERVE, ActionParams(reasoning="No tool call from LLM")

        tool_call = msg.tool_calls[0]
        action_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except json.JSONDecodeError:
            args = {}

        # Acknowledge tool call
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": "Action queued for execution.",
        })

        return self._parse_action(action_name, args)

    def _parse_action(
        self, action_name: str, args: dict[str, Any]
    ) -> tuple[SolanaAction, ActionParams]:
        """Convert LLM tool call into SolanaAction + ActionParams."""
        reasoning = args.get("reasoning", "")

        if action_name == "observe":
            return SolanaAction.OBSERVE, ActionParams(reasoning=reasoning)

        elif action_name == "swap":
            return SolanaAction.SWAP, ActionParams(
                token_in_mint=args.get("token_in_mint", ""),
                token_out_mint=args.get("token_out_mint", ""),
                amount_in=args.get("amount", 0.0),
                reasoning=reasoning,
            )

        elif action_name == "transfer_sol":
            return SolanaAction.TRANSFER_SOL, ActionParams(
                recipient=args.get("recipient", ""),
                sol_amount=args.get("amount", 0.0),
                reasoning=reasoning,
            )

        elif action_name == "transfer_token":
            return SolanaAction.TRANSFER_TOKEN, ActionParams(
                token_mint=args.get("token_mint", ""),
                token_amount=args.get("amount", 0.0),
                recipient=args.get("recipient", ""),
                reasoning=reasoning,
            )

        else:
            return SolanaAction.OBSERVE, ActionParams(
                reasoning=f"Unknown action {action_name}, defaulting to observe"
            )

    def _system_prompt(self) -> str:
        return textwrap.dedent("""\
            You are a DeFi trading agent operating on the Solana blockchain.
            
            Your objective is to analyze the wallet's current state and make
            intelligent trading decisions. You can:
            - OBSERVE: Refresh wallet state without taking action
            - SWAP: Exchange one token for another
            - TRANSFER_SOL: Send SOL to another wallet
            - TRANSFER_TOKEN: Send an SPL token to another wallet
            
            Guidelines:
            - Always provide clear reasoning for your decisions
            - Be conservative with position sizes
            - Monitor P&L and stop if losses exceed tolerance
            - Consider gas/priority fees in your calculations
            - Never risk more than you can afford to lose
            
            You will be shown the current wallet state including SOL balance,
            token holdings with USD values, and recent transaction history.
            
            Choose exactly one action per turn.
        """)

    def _build_state_prompt(self, state: ChainState) -> str:
        """Present chain state to the LLM."""
        lines = [
            "# Current Wallet State",
            f"Wallet: {state.wallet_address}",
            f"SOL Balance: {state.sol_balance:.4f} SOL",
            f"Total Portfolio: ${state.total_usd_value:,.2f}",
            f"Slot: {state.slot} | Epoch: {state.epoch}",
            "",
            "## Token Holdings",
        ]

        if state.token_balances:
            for t in state.token_balances:
                usd = f" (${t.usd_value:,.2f})" if t.usd_value else ""
                lines.append(f"  {t.symbol}: {t.amount:.6f}{usd} | Mint: {t.mint[:12]}...")
        else:
            lines.append("  No SPL tokens held")

        lines.append("")
        lines.append("## Recent Transactions")
        if state.recent_transactions:
            for tx in state.recent_transactions[:5]:
                lines.append(f"  [{tx.type}] {tx.description[:80]} | Sig: {tx.signature[:12]}...")
        else:
            lines.append("  No recent transactions")

        if state.priority_fees:
            lines.append("")
            lines.append("## Priority Fees (microlamports)")
            for level, fee in state.priority_fees.items():
                lines.append(f"  {level}: {fee}")

        lines.append("")
        lines.append(f"P&L since start: ${self.pnl:+,.2f}")
        lines.append("")
        lines.append("Choose one action.")

        return "\n".join(lines)

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build OpenAI tool definitions for Solana actions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "observe",
                    "description": "Refresh wallet state without taking any action. Use when you need more data or want to wait.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string", "description": "Why you chose to observe"},
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
                    "name": "swap",
                    "description": "Swap one token for another via a DEX (e.g. Jupiter).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_in_mint": {"type": "string", "description": "Mint address of the token to sell"},
                            "token_out_mint": {"type": "string", "description": "Mint address of the token to buy"},
                            "amount": {"type": "number", "description": "Amount of token_in to swap"},
                            "reasoning": {"type": "string", "description": "Trading thesis for this swap"},
                        },
                        "required": ["token_in_mint", "token_out_mint", "amount", "reasoning"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "transfer_sol",
                    "description": "Send SOL to another wallet address.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string", "description": "Recipient wallet address (base58)"},
                            "amount": {"type": "number", "description": "Amount of SOL to send"},
                            "reasoning": {"type": "string", "description": "Why you are sending SOL"},
                        },
                        "required": ["recipient", "amount", "reasoning"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "transfer_token",
                    "description": "Send an SPL token to another wallet.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_mint": {"type": "string", "description": "Mint address of the token to send"},
                            "recipient": {"type": "string", "description": "Recipient wallet address (base58)"},
                            "amount": {"type": "number", "description": "Amount of tokens to send"},
                            "reasoning": {"type": "string", "description": "Why you are sending this token"},
                        },
                        "required": ["token_mint", "recipient", "amount", "reasoning"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]
