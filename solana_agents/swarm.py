"""Swarm orchestrator for Solana agents — mirrors agents/swarm.py."""

from __future__ import annotations

import json
import logging
from threading import Thread
from typing import Optional, Type

from .agent import SolanaAgent
from .environment import SolanaEnvironment

logger = logging.getLogger(__name__)


class SolanaSwarm:
    """Orchestration for many Solana agents operating across wallets/strategies."""

    def __init__(
        self,
        agent_class: Type[SolanaAgent],
        agent_name: str,
        wallets: list[str],
        env: SolanaEnvironment,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.agent_class = agent_class
        self.agent_name = agent_name
        self.wallets = wallets
        self.env = env
        self.tags = tags or []
        self.agents: list[SolanaAgent] = []
        self.threads: list[Thread] = []

    def main(self) -> None:
        """Run all agents to completion."""
        logger.info(f"Starting swarm: {self.agent_name} across {len(self.wallets)} wallet(s)")

        # Create agents
        for wallet in self.wallets:
            agent = self.agent_class(
                agent_name=self.agent_name,
                wallet_address=wallet,
                env=self.env,
                record=True,
                tags=self.tags,
            )
            self.agents.append(agent)

        # Create threads
        for agent in self.agents:
            self.threads.append(Thread(target=agent.main, daemon=True))

        # Start all
        for t in self.threads:
            t.start()

        # Wait for completion
        for t in self.threads:
            t.join()

        # Summary
        logger.info("--- SWARM SUMMARY ---")
        total_pnl = 0.0
        for agent in self.agents:
            pnl = agent.pnl
            total_pnl += pnl
            logger.info(
                f"  {agent.wallet_address[:8]}... | "
                f"{agent.action_counter} actions | "
                f"P&L: ${pnl:+,.2f}"
            )
        logger.info(f"  Total P&L: ${total_pnl:+,.2f}")

        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup all agents."""
        for agent in self.agents:
            agent.cleanup()
        self.env.close()
