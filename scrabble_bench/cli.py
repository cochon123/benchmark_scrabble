from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli_agents import list_cli_agents
from .config import DATASET_PATH, ensure_directories
from .dataset import generate_dataset, load_dataset
from .openrouter import model_search
from .quackle import setup_quackle
from .runner import execute_run, prepare_cli_run, prepare_run
from .storage import (
    active_run,
    default_export_path,
    export_all_csv,
    export_run_csv,
    get_run,
    init_db,
    leaderboard,
    list_runs,
    update_run_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="scrabble_bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", help="OpenRouter model id")
    run_parser.add_argument("--preset", default="smoke", choices=["smoke", "full", "custom"])
    run_parser.add_argument("--boards", type=int)
    run_parser.add_argument("--reasoning-effort", default="medium", choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--cli-agent")
    run_parser.add_argument("--cli-model")

    prepare_parser = subparsers.add_parser("prepare-run")
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--preset", default="smoke", choices=["smoke", "full", "custom"])
    prepare_parser.add_argument("--boards", type=int)
    prepare_parser.add_argument("--reasoning-effort", default="medium", choices=["none", "minimal", "low", "medium", "high", "xhigh"])

    prepare_cli_parser = subparsers.add_parser("prepare-cli-run")
    prepare_cli_parser.add_argument("--agent", required=True)
    prepare_cli_parser.add_argument("--model")
    prepare_cli_parser.add_argument("--preset", default="smoke", choices=["smoke", "full", "custom"])
    prepare_cli_parser.add_argument("--boards", type=int)
    prepare_cli_parser.add_argument("--reasoning-effort", default="cli", choices=["cli", "none", "minimal", "low", "medium", "high", "xhigh", "max"])

    cli_run_parser = subparsers.add_parser("cli-run")
    cli_run_parser.add_argument("--agent", required=True, help='CLI agent name: "OpenCode", "Codex CLI", "Claude Code"')
    cli_run_parser.add_argument("--model", help="Model id for agents that support it (e.g. zai-coding-plan/glm-5.2)")
    cli_run_parser.add_argument("--preset", default="smoke", choices=["smoke", "full", "custom"])
    cli_run_parser.add_argument("--boards", type=int)
    cli_run_parser.add_argument("--reasoning-effort", default="cli", choices=["cli", "none", "minimal", "low", "medium", "high", "xhigh", "max"])
    cli_run_parser.add_argument("--run-id")

    subparsers.add_parser("generate-dataset")
    subparsers.add_parser("setup-quackle")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--kind", choices=["all", "run"], required=True)
    export_parser.add_argument("--run-id")
    export_parser.add_argument("--output")

    api_parser = subparsers.add_parser("api")
    api_parser.add_argument(
        "name",
        choices=["dataset", "leaderboard", "runs", "run", "active-run", "search-models", "cli-agents", "cancel-run"],
    )
    api_parser.add_argument("--run-id")
    api_parser.add_argument("--query")

    args = parser.parse_args()
    ensure_directories()
    init_db()

    if args.command == "generate-dataset":
        dataset = generate_dataset(DATASET_PATH)
        print(f"Generated {len(dataset)} dataset positions at {DATASET_PATH}")
        return

    if args.command == "setup-quackle":
        setup_quackle()
        print("Quackle build completed.")
        return

    if args.command == "prepare-run":
        run = prepare_run(args.model, args.preset, args.boards, args.reasoning_effort)
        print(json.dumps(run))
        return

    if args.command == "prepare-cli-run":
        run = prepare_cli_run(args.agent, args.preset, args.boards, args.model, getattr(args, "reasoning_effort", None))
        print(json.dumps(run))
        return

    if args.command == "run":
        if args.run_id:
            run = get_run(args.run_id)
            if not run:
                raise SystemExit(f"Unknown run id: {args.run_id}")
        else:
            if not args.model:
                raise SystemExit("--model is required unless --run-id is provided")
            run = prepare_run(args.model, args.preset, args.boards, args.reasoning_effort)
        execute_run(run, concurrency=args.concurrency, cli_agent=args.cli_agent, cli_model=args.cli_model)
        return

    if args.command == "cli-run":
        if args.run_id:
            run = get_run(args.run_id)
            if not run:
                raise SystemExit(f"Unknown run id: {args.run_id}")
        else:
            run = prepare_cli_run(args.agent, args.preset, args.boards, args.model, getattr(args, "reasoning_effort", None))
        execute_run(run, cli_agent=args.agent, cli_model=args.model)
        return

    if args.command == "export":
        output = Path(args.output) if args.output else None
        if args.kind == "all":
            path = export_all_csv(output or default_export_path("all"))
        else:
            if not args.run_id:
                raise SystemExit("--run-id is required when --kind run")
            path = export_run_csv(args.run_id, output or default_export_path("run", args.run_id))
        print(str(path))
        return

    if args.command == "api":
        payload = None
        if args.name == "dataset":
            payload = load_dataset()
        elif args.name == "leaderboard":
            payload = leaderboard()
        elif args.name == "runs":
            payload = list_runs()
        elif args.name == "run":
            if not args.run_id:
                raise SystemExit("--run-id is required")
            payload = get_run(args.run_id)
        elif args.name == "active-run":
            payload = active_run()
        elif args.name == "search-models":
            payload = model_search(args.query or "")
        elif args.name == "cli-agents":
            payload = list_cli_agents()
        elif args.name == "cancel-run":
            if not args.run_id:
                raise SystemExit("--run-id is required")
            update_run_status(args.run_id, "cancelled")
            payload = {"ok": True, "run_id": args.run_id}
        print(json.dumps(payload))
        return
