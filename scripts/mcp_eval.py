#!/usr/bin/env python3
"""Evaluation runner for ehrapy MCP server (T21).

Runs multi-step workflow tasks defined in tests/mcp/eval_tasks.yaml and reports metrics.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import yaml
from fastmcp import Client

from ehrapy.mcp.server import mcp


async def run_eval() -> int:
    """Run all evaluation tasks defined in eval_tasks.yaml and print results."""
    tasks_file = Path(__file__).resolve().parent.parent / "tests" / "mcp" / "eval_tasks.yaml"
    if not tasks_file.exists():
        print(f"Error: {tasks_file} not found.", file=sys.stderr)
        return 1

    with tasks_file.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tasks = data.get("tasks", [])
    print(f"Loaded {len(tasks)} evaluation tasks from {tasks_file.name}\n")
    print("=" * 80)

    total_steps = 0
    passed_steps = 0
    failed_steps = 0
    total_time = 0.0

    async with Client(mcp) as client:
        for task in tasks:
            task_id = task.get("id")
            task_name = task.get("name")
            steps = task.get("steps", [])
            print(f"\n▶ Running Task: [{task_id}] {task_name} ({len(steps)} steps)")

            task_passed = True
            for i, step in enumerate(steps, 1):
                total_steps += 1
                tool_name = step["tool"]
                args = step.get("args", {})
                expected_status = step.get("assert_status", "ok")

                start_t = time.perf_counter()
                try:
                    res = await client.call_tool(tool_name, args)
                    elapsed = time.perf_counter() - start_t
                    total_time += elapsed

                    struct = res.structured_content or {}
                    actual_status = struct.get("status", "error" if res.is_error else "ok")

                    content_text = ""
                    if res.content:
                        content_text = " ".join(getattr(c, "text", "") for c in res.content)

                    if actual_status != expected_status:
                        print(
                            f"  ❌ Step {i} ({tool_name}): expected status '{expected_status}', got '{actual_status}'"
                        )
                        print(f"     Reason: {struct.get('reason')}")
                        task_passed = False
                        failed_steps += 1
                        break

                    assert_contains = step.get("assert_contains", [])
                    missing_keys = [k for k in assert_contains if k not in struct and k not in content_text]
                    if missing_keys:
                        print(f"  ❌ Step {i} ({tool_name}): missing expected fields {missing_keys}")
                        task_passed = False
                        failed_steps += 1
                        break

                    expected_code = step.get("assert_error_code")
                    if expected_code and struct.get("error_code") != expected_code:
                        print(
                            f"  ❌ Step {i} ({tool_name}): expected error_code "
                            f"'{expected_code}', got '{struct.get('error_code')}'"
                        )
                        task_passed = False
                        failed_steps += 1
                        break

                    # A plot step must actually return an image, not just report success.
                    if step.get("assert_image"):
                        kinds = [type(c).__name__ for c in (res.content or [])]
                        if "ImageContent" not in kinds:
                            print(f"  ❌ Step {i} ({tool_name}): expected an image, got {kinds}")
                            task_passed = False
                            failed_steps += 1
                            break

                    passed_steps += 1
                    approx_tokens = int(len(content_text) / 4)
                    print(f"  ✔ Step {i}: {tool_name} ({elapsed * 1000:.1f}ms, ~{approx_tokens} tokens)")

                except Exception as exc:  # noqa: BLE001
                    elapsed = time.perf_counter() - start_t
                    total_time += elapsed
                    print(f"  ❌ Step {i} ({tool_name}): Exception {exc}")
                    task_passed = False
                    failed_steps += 1
                    break

            if task_passed:
                print(f"  🏆 Task [{task_id}] PASSED")
            else:
                print(f"  💥 Task [{task_id}] FAILED")

    print("\n" + "=" * 80)
    print(f"Eval Summary: {passed_steps}/{total_steps} steps passed. Total time: {total_time:.2f}s")
    return 0 if failed_steps == 0 else 1


def main() -> None:
    """Run CLI entrypoint for evaluation harness."""
    sys.exit(asyncio.run(run_eval()))


if __name__ == "__main__":
    main()
