# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise revision-bound comment approval without network access or delays."""

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def gate(monkeypatch):
    path = Path(__file__).resolve().parents[3] / ".github/scripts/ci_full_gate.py"
    spec = importlib.util.spec_from_file_location("ci_full_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "api", lambda *a: pytest.fail("Unexpected API call"))
    monkeypatch.setattr(module.time, "sleep", lambda *a: pytest.fail("Unexpected wait"))
    return module


@pytest.fixture
def pr():
    return {
        "number": 1,
        "state": "open",
        "head": {"sha": "a" * 40, "repo": {"id": 2}},
        "base": {"sha": "b" * 40, "repo": {"id": 1, "full_name": "owner/repo"}, "ref": "main"},
    }


def comment(body="/ci full", created="2026-11-11T03:00:01Z"):
    return {"id": 12, "body": body, "created_at": created, "user": {"type": "User"}}


def harness(gate, monkeypatch, pr, comments):
    live = copy.deepcopy(pr)
    calls = []

    def request(path):
        calls.append(path)
        if path.endswith("/actions/runs/7"):
            return {
                "event": "pull_request",
                "head_sha": pr["head"]["sha"],
                "created_at": "2026-11-11T03:00:00Z",
            }
        if path.endswith("/pulls/1"):
            return live
        if "/issues/1/comments?" in path:
            return comments
        raise AssertionError(path)

    monkeypatch.setattr(gate, "api", request)
    return live, calls


def test_comment_releases_same_run_after_revision_recheck(gate, monkeypatch, pr):
    _, calls = harness(gate, monkeypatch, pr, [comment()])
    assert gate.wait_for_comment("owner/repo", pr, 7, 10, 1) == 12
    assert sum(path.endswith("/pulls/1") for path in calls) == 2
    assert all("dispatch" not in path and "statuses" not in path for path in calls)


@pytest.mark.parametrize("kind", ["old", "different-command", "bot"])
def test_old_or_unrelated_comment_does_not_release_gate(gate, monkeypatch, pr, kind):
    item = comment()
    if kind == "old":
        item["created_at"] = "2026-11-11T02:59:59Z"
    elif kind == "different-command":
        item["body"] = "/ci full later"
    else:
        item["user"]["type"] = "Bot"
    harness(gate, monkeypatch, pr, [item])
    with pytest.raises(ValueError, match="timeout"):
        gate.wait_for_comment("owner/repo", pr, 7, 0, 1)


@pytest.mark.parametrize("change", ["head", "base", "closed", "target"])
def test_changed_revision_cannot_be_approved(gate, monkeypatch, pr, change):
    live, _ = harness(gate, monkeypatch, pr, [comment()])
    if change in {"head", "base"}:
        live[change]["sha"] = "c" * 40
    elif change == "closed":
        live["state"] = "closed"
    else:
        live["base"]["ref"] = "another-base"
    with pytest.raises(ValueError, match="changed|closed"):
        gate.wait_for_comment("owner/repo", pr, 7, 10, 1)


def test_push_during_comment_lookup_is_rejected(gate, monkeypatch, pr):
    live, _ = harness(gate, monkeypatch, pr, [])

    def approve(*args):
        live["head"]["sha"] = "c" * 40
        return 12

    monkeypatch.setattr(gate, "approval", approve)
    with pytest.raises(ValueError, match="changed"):
        gate.wait_for_comment("owner/repo", pr, 7, 10, 1)


def test_waiting_polls_until_new_comment_arrives(gate, monkeypatch, pr):
    comments = []
    harness(gate, monkeypatch, pr, comments)
    waits = []

    def advance(seconds):
        waits.append(seconds)
        comments.append(comment())

    monkeypatch.setattr(gate.time, "sleep", advance)
    assert gate.wait_for_comment("owner/repo", pr, 7, 10, 1) == 12
    assert waits == [1]


def test_comment_lookup_follows_pagination(gate, monkeypatch):
    pages = []

    def request(path):
        pages.append(path)
        return [comment("irrelevant")] * 100 if path.endswith("&page=1") else [comment()]

    monkeypatch.setattr(gate, "api", request)
    assert gate.approval("owner/repo", 1, "2026-11-11T03:00:00Z") == 12
    assert len(pages) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", f"--confcutdir={Path(__file__).parent}"]))
