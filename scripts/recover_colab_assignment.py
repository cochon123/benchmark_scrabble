#!/usr/bin/env python3
"""Adopt an existing server-side Colab assignment into the local CLI state.

This is intentionally conservative: it never creates or releases an assignment,
and it does not print the runtime proxy token.
"""

from __future__ import annotations

import argparse

from colab_cli.commands.session import spawn_keep_alive
from colab_cli.common import state
from colab_cli.state import SessionState


VARIANT_NAMES = {0: "DEFAULT", 1: "GPU", 2: "TPU"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--endpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = [
        assignment
        for assignment in state.client.list_assignments()
        if assignment.endpoint == args.endpoint
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one assignment for {args.endpoint!r}; found {len(matches)}."
        )

    assignment = matches[0]
    existing = state.store.get(args.name)
    session = SessionState(
        name=args.name,
        token=assignment.runtime_proxy_info.token,
        url=assignment.runtime_proxy_info.url,
        endpoint=assignment.endpoint,
        variant=VARIANT_NAMES[int(assignment.variant)],
        accelerator=assignment.accelerator.value,
        kernel_id=existing.kernel_id if existing else None,
        session_id=existing.session_id if existing else None,
        keep_alive_pid=existing.keep_alive_pid if existing else None,
    )
    state.store.add(session)
    if session.keep_alive_pid is None:
        session.keep_alive_pid = spawn_keep_alive(
            session.endpoint,
            session.name,
            auth_provider=state.auth_provider,
            config_path=state.config_path,
        )
    state.store.add(session)
    print(
        "Recovered "
        f"name={session.name} endpoint={session.endpoint} "
        f"variant={session.variant} accelerator={session.accelerator}"
    )


if __name__ == "__main__":
    main()
