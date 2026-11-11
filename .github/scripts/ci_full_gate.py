# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Wait for /ci full within the existing PR run using read-only GitHub access."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


def api(path: str) -> Any:
    result = subprocess.run(["gh", "api", path], text=True, capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


def current_pr(repo: str, original: dict[str, Any]) -> None:
    current = api(f"repos/{repo}/pulls/{original['number']}")
    if current["state"] != "open" or any(
        current[side]["sha"] != original[side]["sha"]
        or current[side]["repo"]["id"] != original[side]["repo"]["id"]
        for side in ("head", "base")
    ):
        raise ValueError("PR head/base changed or PR closed; use the new PR run")
    if current["base"]["ref"] != original["base"]["ref"]:
        raise ValueError("PR target changed; use the new PR run")


def approval(repo: str, number: int, created: str) -> int | None:
    for page in range(1, 11):
        query = urlencode({"since": created, "per_page": 100, "page": page})
        comments = api(f"repos/{repo}/issues/{number}/comments?{query}")
        for comment in comments:
            if (
                comment["created_at"] >= created
                and comment["body"].strip() == "/ci full"
                and comment.get("user", {}).get("type") == "User"
            ):
                return int(comment["id"])
        if len(comments) < 100:
            return None
    raise ValueError("Too many comments to verify approval safely")


def wait_for_comment(repo: str, pr: dict[str, Any], run_id: int, wait_seconds: int, poll_seconds: int) -> int:
    if pr["base"]["repo"]["full_name"] != repo or run_id <= 0:
        raise ValueError("Unexpected PR repository or run ID")
    run = api(f"repos/{repo}/actions/runs/{run_id}")
    if run["event"] != "pull_request" or run["head_sha"] != pr["head"]["sha"]:
        raise ValueError("The run does not test this PR head")
    deadline = time.monotonic() + wait_seconds
    while True:
        current_pr(repo, pr)
        comment = approval(repo, pr["number"], run["created_at"])
        if comment is not None:
            # A push may race the comment lookup. Recheck before releasing tests.
            current_pr(repo, pr)
            return comment
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("No /ci full received before timeout; comment and rerun failed jobs")
        time.sleep(min(poll_seconds, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=int, default=350 * 60)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 350 * 60 or not 1 <= args.poll_seconds <= 60:
        parser.error("wait must be 1..21000 seconds; poll must be 1..60 seconds")
    if os.environ["GITHUB_EVENT_NAME"] != "pull_request":
        raise ValueError("The comment gate only supports pull_request runs")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repo = os.environ["GITHUB_REPOSITORY"]
    comment = wait_for_comment(
        repo,
        event["pull_request"],
        int(os.environ["GITHUB_RUN_ID"]),
        args.wait_seconds,
        args.poll_seconds,
    )
    message = f"Full CI released by https://github.com/{repo}/issues/{event['number']}#issuecomment-{comment}"
    print(message)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(message + "\n")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError:
        raise SystemExit("GitHub metadata request failed; rerun the gate to retry") from None
    except ValueError as error:
        raise SystemExit(str(error)) from None
