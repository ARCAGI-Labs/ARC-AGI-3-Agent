"""Solana agent framework — module init with agent registry."""

from typing import Type

from .agent import SolanaAgent
from .environment import SolanaEnvironment
from .models import ActionParams, ActionResult, ChainState, SolanaAction, TokenBalance
from .recorder import SolanaRecorder
from .swarm import SolanaSwarm
from .templates.defi_trader import DeFiTrader
from .templates.portfolio_monitor import PortfolioMonitor
from .templates.whale_tracker import WhaleTracker

AVAILABLE_AGENTS: dict[str, Type[SolanaAgent]] = {
    "defi_trader": DeFiTrader,
    "portfolio_monitor": PortfolioMonitor,
    "whale_tracker": WhaleTracker,
}

__all__ = [
    "SolanaAgent",
    "SolanaEnvironment",
    "SolanaSwarm",
    "SolanaRecorder",
    "ChainState",
    "SolanaAction",
    "ActionParams",
    "ActionResult",
    "TokenBalance",
    "DeFiTrader",
    "PortfolioMonitor",
    "WhaleTracker",
    "AVAILABLE_AGENTS",
]
