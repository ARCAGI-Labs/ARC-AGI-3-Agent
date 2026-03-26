# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import os
import signal
import sys
import threading
from functools import partial
from types import FrameType
from typing import Optional

from solana_agents import AVAILABLE_AGENTS, SolanaEnvironment, SolanaSwarm

logger = logging.getLogger()


def run_swarm(swarm: SolanaSwarm) -> None:
    swarm.main()
    os.kill(os.getpid(), signal.SIGINT)


def cleanup(
    swarm: SolanaSwarm,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, shutting down...")
    swarm.cleanup()
    sys.exit(0)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("solana_logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser(description="Solana Agent Framework")
    parser.add_argument(
        "-a",
        "--agent",
        choices=AVAILABLE_AGENTS.keys(),
        required=True,
        help="Choose which agent to run.",
    )
    parser.add_argument(
        "-w",
        "--wallet",
        type=str,
        help="Solana wallet address to monitor/operate on. Defaults to SOLANA_WALLET_ADDRESS env var.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable live transactions (default: dry-run mode).",
    )
    parser.add_argument(
        "--whale-wallets",
        type=str,
        help="Comma-separated whale wallet addresses (for whale_tracker agent).",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        help="Comma-separated tags for the session (e.g., 'experiment,v1.0')",
        default=None,
    )

    args = parser.parse_args()

    # Resolve wallet address
    wallet = args.wallet or os.environ.get("SOLANA_WALLET_ADDRESS", "")
    if not wallet:
        logger.error(
            "No wallet address specified. Use --wallet or set SOLANA_WALLET_ADDRESS."
        )
        return

    # Live mode safety check
    dry_run = not args.live
    if not dry_run:
        print("\n" + "=" * 60)
        print("⚠️  LIVE MODE ENABLED — REAL TRANSACTIONS WILL BE SUBMITTED")
        print("=" * 60)
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    # Helius API key
    helius_key = os.environ.get("HELIUS_API_KEY", "")
    if not helius_key:
        logger.error("HELIUS_API_KEY not set. Add it to .env")
        return

    # Tags
    tags: list[str] = ["solana", args.agent]
    if args.tags:
        tags.extend([t.strip() for t in args.tags.split(",")])
    if dry_run:
        tags.append("dry-run")

    # Create environment
    env = SolanaEnvironment(
        wallet_address=wallet,
        helius_api_key=helius_key,
        dry_run=dry_run,
    )

    # Create swarm (single wallet for now)
    agent_class = AVAILABLE_AGENTS[args.agent]
    swarm = SolanaSwarm(
        agent_class=agent_class,
        agent_name=args.agent,
        wallets=[wallet],
        env=env,
        tags=tags,
    )

    logger.info(f"Starting {args.agent} agent for wallet {wallet[:12]}...")
    logger.info(f"Mode: {'🔴 LIVE' if not dry_run else '🟢 DRY RUN'}")

    # Run in thread with signal handling
    swarm_thread = threading.Thread(target=partial(run_swarm, swarm))
    swarm_thread.daemon = True
    swarm_thread.start()

    signal.signal(signal.SIGINT, partial(cleanup, swarm))

    try:
        while swarm_thread.is_alive():
            swarm_thread.join(timeout=5)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
        cleanup(swarm, signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        cleanup(swarm, None, None)


if __name__ == "__main__":
    main()
