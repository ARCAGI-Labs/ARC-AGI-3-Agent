"""Data models for the Solana agent framework."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SolanaAction(str, Enum):
    """Actions a Solana agent can take."""

    OBSERVE = "observe"
    SWAP = "swap"
    TRANSFER_SOL = "transfer_sol"
    TRANSFER_TOKEN = "transfer_token"
    STAKE = "stake"
    UNSTAKE = "unstake"

    def needs_params(self) -> bool:
        return self != SolanaAction.OBSERVE


class TokenBalance(BaseModel):
    """A single SPL token holding."""

    mint: str
    symbol: str = "UNKNOWN"
    name: str = ""
    amount: float = 0.0
    decimals: int = 0
    usd_price: Optional[float] = None
    usd_value: Optional[float] = None


class ParsedTransaction(BaseModel):
    """A simplified parsed transaction from history."""

    signature: str
    timestamp: Optional[int] = None
    type: str = ""
    description: str = ""
    fee: int = 0
    source: str = ""
    native_transfers: list[dict[str, Any]] = Field(default_factory=list)
    token_transfers: list[dict[str, Any]] = Field(default_factory=list)


class ChainState(BaseModel):
    """Snapshot of on-chain state for a wallet — replaces FrameData."""

    wallet_address: str
    sol_balance: float = 0.0
    sol_usd_price: Optional[float] = None
    token_balances: list[TokenBalance] = Field(default_factory=list)
    recent_transactions: list[ParsedTransaction] = Field(default_factory=list)
    priority_fees: dict[str, int] = Field(default_factory=dict)
    slot: int = 0
    epoch: int = 0
    timestamp: float = 0.0

    @property
    def total_usd_value(self) -> float:
        sol_value = (
            self.sol_balance * self.sol_usd_price if self.sol_usd_price else 0.0
        )
        token_value = sum(t.usd_value or 0.0 for t in self.token_balances)
        return sol_value + token_value

    def summary(self) -> str:
        lines = [
            f"Wallet: {self.wallet_address}",
            f"SOL: {self.sol_balance:.4f}",
            f"Tokens: {len(self.token_balances)}",
            f"Portfolio USD: ${self.total_usd_value:,.2f}",
            f"Slot: {self.slot}",
        ]
        for t in self.token_balances[:10]:
            usd = f" (${t.usd_value:,.2f})" if t.usd_value else ""
            lines.append(f"  {t.symbol}: {t.amount:.6f}{usd}")
        return "\n".join(lines)


class ActionParams(BaseModel):
    """Parameters for a SolanaAction."""

    # For SWAP
    token_in_mint: Optional[str] = None
    token_out_mint: Optional[str] = None
    amount_in: Optional[float] = None

    # For TRANSFER_SOL
    recipient: Optional[str] = None
    sol_amount: Optional[float] = None

    # For TRANSFER_TOKEN
    token_mint: Optional[str] = None
    token_amount: Optional[float] = None

    # General
    reasoning: str = ""


class ActionResult(BaseModel):
    """Result of executing a SolanaAction."""

    action: SolanaAction
    params: ActionParams
    success: bool = False
    dry_run: bool = True
    signature: Optional[str] = None
    error: Optional[str] = None
    fee_lamports: int = 0
    state_before: Optional[ChainState] = None
    state_after: Optional[ChainState] = None

    @property
    def fee_sol(self) -> float:
        return self.fee_lamports / 1e9
