"""Solana environment wrapper — the blockchain equivalent of EnvironmentWrapper.

Uses httpx to call Helius APIs for chain state and transaction submission.
Supports dry_run mode (default) to prevent real transactions.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx

from .models import (
    ActionParams,
    ActionResult,
    ChainState,
    ParsedTransaction,
    SolanaAction,
    TokenBalance,
)

logger = logging.getLogger(__name__)

HELIUS_MAINNET = "https://mainnet.helius-rpc.com"
HELIUS_API_V0 = "https://api.helius.xyz/v0"


class SolanaEnvironment:
    """Wraps Helius/Solana RPC behind a get_state() / step() interface."""

    def __init__(
        self,
        wallet_address: str,
        helius_api_key: Optional[str] = None,
        dry_run: bool = True,
    ) -> None:
        self.wallet_address = wallet_address
        self.api_key = helius_api_key or os.getenv("HELIUS_API_KEY", "")
        self.dry_run = dry_run
        self.rpc_url = f"{HELIUS_MAINNET}/?api-key={self.api_key}"
        self.api_url = f"{HELIUS_API_V0}"
        self._client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._client.close()

    # ── RPC helpers ──────────────────────────────────────────────

    def _rpc_call(self, method: str, params: list[Any] | None = None) -> Any:
        """Make a JSON-RPC call to Helius."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }
        resp = self._client.post(self.rpc_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"RPC error: {data['error']}")
        return data.get("result")

    def _api_call(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Make a REST API call to Helius v0 endpoint."""
        url = f"{self.api_url}/{endpoint}"
        all_params = {"api-key": self.api_key, **(params or {})}
        resp = self._client.get(url, params=all_params)
        resp.raise_for_status()
        return resp.json()

    def _das_call(self, method: str, params: dict[str, Any]) -> Any:
        """Make a DAS (Digital Asset Standard) API call."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        resp = self._client.post(self.rpc_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"DAS error: {data['error']}")
        return data.get("result")

    # ── State observation ────────────────────────────────────────

    def get_state(self, wallet: Optional[str] = None) -> ChainState:
        """Fetch current on-chain state for a wallet."""
        address = wallet or self.wallet_address
        state = ChainState(wallet_address=address, timestamp=time.time())

        # SOL balance
        try:
            result = self._rpc_call("getBalance", [address])
            if result and "value" in result:
                state.sol_balance = result["value"] / 1e9
        except Exception as e:
            logger.warning(f"Failed to fetch SOL balance: {e}")

        # Token balances (via DAS searchAssets or getTokenAccountsByOwner)
        try:
            state.token_balances = self._fetch_token_balances(address)
        except Exception as e:
            logger.warning(f"Failed to fetch token balances: {e}")

        # Priority fee estimate
        try:
            state.priority_fees = self._fetch_priority_fees()
        except Exception as e:
            logger.warning(f"Failed to fetch priority fees: {e}")

        # Recent transactions
        try:
            state.recent_transactions = self._fetch_recent_transactions(address)
        except Exception as e:
            logger.warning(f"Failed to fetch recent transactions: {e}")

        # Network info
        try:
            epoch_info = self._rpc_call("getEpochInfo")
            if epoch_info:
                state.slot = epoch_info.get("absoluteSlot", 0)
                state.epoch = epoch_info.get("epoch", 0)
        except Exception as e:
            logger.warning(f"Failed to fetch epoch info: {e}")

        return state

    def _fetch_token_balances(self, address: str) -> list[TokenBalance]:
        """Fetch SPL token balances using getTokenAccountsByOwner."""
        result = self._rpc_call(
            "getTokenAccountsByOwner",
            [
                address,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"},
            ],
        )
        balances: list[TokenBalance] = []
        if not result or "value" not in result:
            return balances

        for account in result["value"]:
            info = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            token_amount = info.get("tokenAmount", {})
            amount = float(token_amount.get("uiAmountString", "0"))
            if amount > 0:
                balances.append(
                    TokenBalance(
                        mint=info.get("mint", ""),
                        amount=amount,
                        decimals=token_amount.get("decimals", 0),
                    )
                )
        return balances

    def _fetch_priority_fees(self) -> dict[str, int]:
        """Fetch priority fee estimates via Helius getPriorityFeeEstimate."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getPriorityFeeEstimate",
                "params": [{"options": {"includeAllPriorityFeeLevels": True}}],
            }
            resp = self._client.post(self.rpc_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", {})
            levels = result.get("priorityFeeLevels", {})
            return {k: int(v) for k, v in levels.items()} if levels else {}
        except Exception:
            return {}

    def _fetch_recent_transactions(
        self, address: str, limit: int = 10
    ) -> list[ParsedTransaction]:
        """Fetch recent parsed transactions via Helius enhanced API."""
        try:
            # First get signatures
            sigs_result = self._rpc_call(
                "getSignaturesForAddress",
                [address, {"limit": limit}],
            )
            if not sigs_result:
                return []

            signatures = [s["signature"] for s in sigs_result]

            # Parse via Helius
            resp = self._client.post(
                f"{self.api_url}/transactions",
                params={"api-key": self.api_key},
                json={"transactions": signatures},
            )
            if resp.status_code != 200:
                return []

            txns = []
            for tx in resp.json():
                txns.append(
                    ParsedTransaction(
                        signature=tx.get("signature", ""),
                        timestamp=tx.get("timestamp"),
                        type=tx.get("type", ""),
                        description=tx.get("description", ""),
                        fee=tx.get("fee", 0),
                        source=tx.get("source", ""),
                        native_transfers=tx.get("nativeTransfers", []),
                        token_transfers=tx.get("tokenTransfers", []),
                    )
                )
            return txns
        except Exception as e:
            logger.warning(f"Failed to parse transactions: {e}")
            return []

    # ── Action execution ─────────────────────────────────────────

    def step(self, action: SolanaAction, params: ActionParams) -> ActionResult:
        """Execute an action (or simulate in dry-run mode)."""
        state_before = self.get_state()

        if action == SolanaAction.OBSERVE:
            return ActionResult(
                action=action,
                params=params,
                success=True,
                dry_run=self.dry_run,
                state_before=state_before,
                state_after=state_before,
            )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute {action.value}: {params.model_dump()}")
            return ActionResult(
                action=action,
                params=params,
                success=True,
                dry_run=True,
                state_before=state_before,
                state_after=state_before,
            )

        # Real execution paths
        try:
            if action == SolanaAction.TRANSFER_SOL:
                return self._execute_transfer_sol(params, state_before)
            elif action == SolanaAction.TRANSFER_TOKEN:
                return self._execute_transfer_token(params, state_before)
            elif action == SolanaAction.SWAP:
                logger.warning("SWAP requires DEX integration — returning dry run")
                return ActionResult(
                    action=action,
                    params=params,
                    success=False,
                    dry_run=False,
                    error="SWAP requires Jupiter/Raydium DEX integration",
                    state_before=state_before,
                )
            else:
                return ActionResult(
                    action=action,
                    params=params,
                    success=False,
                    dry_run=False,
                    error=f"Action {action.value} not yet implemented for live mode",
                    state_before=state_before,
                )
        except Exception as e:
            logger.error(f"Action {action.value} failed: {e}")
            return ActionResult(
                action=action,
                params=params,
                success=False,
                dry_run=False,
                error=str(e),
                state_before=state_before,
            )

    def _execute_transfer_sol(
        self, params: ActionParams, state_before: ChainState
    ) -> ActionResult:
        """Execute a SOL transfer via Helius sendTransaction."""
        if not params.recipient or not params.sol_amount:
            return ActionResult(
                action=SolanaAction.TRANSFER_SOL,
                params=params,
                success=False,
                error="Missing recipient or amount",
                state_before=state_before,
            )

        # NOTE: Real implementation would build + sign + send transaction
        # This is a placeholder — actual signing requires a keypair
        logger.warning(
            "Live SOL transfer requires keypair integration. "
            "Use solana-py or Helius transferSol for real transactions."
        )
        return ActionResult(
            action=SolanaAction.TRANSFER_SOL,
            params=params,
            success=False,
            dry_run=False,
            error="Keypair signing not yet integrated — use Helius MCP transferSol",
            state_before=state_before,
        )

    def _execute_transfer_token(
        self, params: ActionParams, state_before: ChainState
    ) -> ActionResult:
        """Execute an SPL token transfer."""
        logger.warning(
            "Live token transfer requires keypair integration. "
            "Use Helius MCP transferToken for real transactions."
        )
        return ActionResult(
            action=SolanaAction.TRANSFER_TOKEN,
            params=params,
            success=False,
            dry_run=False,
            error="Keypair signing not yet integrated — use Helius MCP transferToken",
            state_before=state_before,
        )
