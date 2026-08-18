from __future__ import annotations

import os
import threading

import pandas as pd
from clearml import Task


def _get_metrics_sync(task_id: str) -> pd.DataFrame:
    task = Task.get_task(task_id=task_id)
    scalars = task.get_reported_scalars()

    rows = []
    for title, series in scalars.items():
        for name, values in series.items():
            for step, value in zip(values.get("x", []), values.get("y", [])):
                rows.append({
                    "metric": title,
                    "series": name,
                    "step": step,
                    "value": value,
                })

    return pd.DataFrame(rows)


def _run_with_timeout(action, timeout_seconds: int):
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["value"] = action()
        except BaseException as exc:
            error["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(
            f"ClearML request timed out after {timeout_seconds} seconds"
        )
    if "error" in error:
        raise error["error"]
    return result["value"]


def get_metrics(task_id: str) -> pd.DataFrame:
    timeout_seconds = int(os.getenv("CLEARML_REQUEST_TIMEOUT_SECONDS", "30"))
    return _run_with_timeout(
        action=lambda: _get_metrics_sync(task_id),
        timeout_seconds=timeout_seconds,
    )


def load_clearml_model(task_id: str):
    timeout_seconds = int(os.getenv("CLEARML_REQUEST_TIMEOUT_SECONDS", "30"))

    def _load_sync():
        task = Task.get_task(task_id=task_id)
        models = task.models.get("output", [])
        if not models:
            raise RuntimeError("No output model registered in ClearML")
        return models[-1].get_local_copy()

    return _run_with_timeout(action=_load_sync, timeout_seconds=timeout_seconds)
