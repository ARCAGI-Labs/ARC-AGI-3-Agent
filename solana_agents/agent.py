"""Abstract SolanaAgent base class — mirrors agents/agent.py for blockchain operations."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from .environment import SolanaEnvironment
from .models import ActionParams, ActionResult, ChainState, SolanaAction
from .recorder import SolanaRecorder

logger = logging.getLogger(__name__)


class SolanaAgent(ABC):
    """Interface for an agent that operates on the Solana blockchain.

    Mirrors the ARC-AGI Agent class but replaces game frames with chain state
    and game actions with blockchain operations.
    """

    MAX_ACTIONS: int = 50
    LOOP_DELAY: float = 2.0  # seconds between action loops

    agent_name: str
    wallet_address: str
    action_counter: int
    timer: float
    states: list[ChainState]
    results: list[ActionResult]
    recorder: SolanaRecorder
    env: SolanaEnvironment
    tags: list[str]

    # Risk management
    MAX_SOL_PER_TX: float = 1.0  # max SOL per transaction
    MIN_SOL_RESERVE: float = 0.05  # always keep this much SOL for fees

    def __init__(
        self,
        agent_name: str,
        wallet_address: str,
        env: SolanaEnvironment,
        record: bool = True,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.agent_name = agent_name
        self.wallet_address = wallet_address
        self.env = env
        self.action_counter = 0
        self.timer = 0.0
        self.states = []
        self.results = []
        self.tags = tags or []
        self._cleanup_done = False

        if record:
            self.recorder = SolanaRecorder(
                prefix=f"{self.wallet_address[:8]}.{self.name}"
            )
            logger.info(f"Recording to {self.recorder.filename}")

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def latest_state(self) -> Optional[ChainState]:
        return self.states[-1] if self.states else None

    @property
    def seconds(self) -> float:
        return round(time.time() - self.timer, 2)

    @property
    def pnl(self) -> float:
        """Calculate P&L from first to latest state."""
        if len(self.states) < 2:
            return 0.0
        initial = self.states[0].total_usd_value
        current = self.states[-1].total_usd_value
        return current - initial if initial > 0 else 0.0

    def main(self) -> None:
        """The main agent loop. Observe → decide → act until done or max actions."""
        self.timer = time.time()
        logger.info(f"Starting {self.name} for wallet {self.wallet_address}")

        # Initial observation
        initial_state = self.env.get_state()
        self.states.append(initial_state)
        self._record_state(initial_state)
        logger.info(f"Initial state:\n{initial_state.summary()}")

        while not self.is_done(self.states) and self.action_counter < self.MAX_ACTIONS:
            try:
                # Choose action
                action, params = self.choose_action(self.states)
                logger.info(
                    f"[{self.action_counter}/{self.MAX_ACTIONS}] "
                    f"Action: {action.value} | Reasoning: {params.reasoning[:100]}"
                )

                # Risk check
                if not self._risk_check(action, params):
                    logger.warning(f"Risk check failed for {action.value}, skipping")
                    self.action_counter += 1
                    continue

                # Execute
                result = self.env.step(action, params)
                self.results.append(result)
                self._record_action(action, params, result)

                if result.success:
                    # Refresh state after action
                    new_state = result.state_after or self.env.get_state()
                    self.states.append(new_state)
                    self._record_state(new_state)
                    logger.info(
                        f"  → Success | SOL: {new_state.sol_balance:.4f} | "
                        f"P&L: ${self.pnl:+,.2f} | "
                        f"{'[DRY RUN]' if result.dry_run else result.signature or ''}"
                    )
                else:
                    logger.warning(f"  → Failed: {result.error}")

            except Exception as e:
                logger.error(f"Error in action loop: {e}", exc_info=True)

            self.action_counter += 1
            if self.LOOP_DELAY > 0:
                time.sleep(self.LOOP_DELAY)

        self.cleanup()

    def _risk_check(self, action: SolanaAction, params: ActionParams) -> bool:
        """Basic risk management checks."""
        if action == SolanaAction.OBSERVE:
            return True

        state = self.latest_state
        if not state:
            return False

        # Check SOL reserve
        if action == SolanaAction.TRANSFER_SOL and params.sol_amount:
            if params.sol_amount > self.MAX_SOL_PER_TX:
                logger.warning(
                    f"Amount {params.sol_amount} SOL exceeds MAX_SOL_PER_TX "
                    f"({self.MAX_SOL_PER_TX})"
                )
                return False
            if state.sol_balance - params.sol_amount < self.MIN_SOL_RESERVE:
                logger.warning("Would breach MIN_SOL_RESERVE")
                return False

        return True

    def _record_state(self, state: ChainState) -> None:
        if hasattr(self, "recorder"):
            self.recorder.record({
                "type": "state",
                "data": state.model_dump(),
            })

    def _record_action(
        self, action: SolanaAction, params: ActionParams, result: ActionResult
    ) -> None:
        if hasattr(self, "recorder"):
            self.recorder.record({
                "type": "action",
                "action": action.value,
                "params": params.model_dump(),
                "success": result.success,
                "dry_run": result.dry_run,
                "signature": result.signature,
                "error": result.error,
            })

    def cleanup(self) -> None:
        """Called after main loop is finished."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        logger.info(
            f"Agent {self.name} finished: "
            f"{self.action_counter} actions in {self.seconds}s | "
            f"P&L: ${self.pnl:+,.2f}"
        )
        if hasattr(self, "recorder"):
            self.recorder.record({
                "type": "summary",
                "agent": self.name,
                "wallet": self.wallet_address,
                "actions": self.action_counter,
                "duration_seconds": self.seconds,
                "pnl_usd": self.pnl,
                "dry_run": self.env.dry_run,
            })
            logger.info(f"Recording saved: {self.recorder.filename}")

    @abstractmethod
    def is_done(self, states: list[ChainState]) -> bool:
        """Decide if the agent should stop operating."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, states: list[ChainState]
    ) -> tuple[SolanaAction, ActionParams]:
        """Choose which action to take and return it with parameters."""
        raise NotImplementedError
