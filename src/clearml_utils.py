from __future__ import annotations

import pandas as pd
from clearml import Task


def get_metrics(task_id: str) -> pd.DataFrame:
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


def load_clearml_model(task_id: str):
    task = Task.get_task(task_id=task_id)
    models = task.models.get("output", [])
    if not models:
        raise RuntimeError("No output model registered in ClearML")
    return models[-1].get_local_copy()
