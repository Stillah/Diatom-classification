"""Утилиты для опционального трекинга обучения в ClearML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _output_uri() -> Any:
    value = os.getenv("CLEARML_OUTPUT_URI", "true").strip()
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off", "none", ""}:
        return None
    return value


def init_clearml_task(
    default_task_name: str,
    config: Mapping[str, Any],
) -> Optional[Any]:
    """Создаёт ClearML Task, если ``CLEARML_ENABLED=1``.

    При недоступном сервере обучение по умолчанию продолжается без трекинга.
    Для fail-fast поведения можно задать ``CLEARML_STRICT=1``.
    """
    if not _env_flag("CLEARML_ENABLED"):
        return None

    try:
        from clearml import Task

        task = Task.init(
            project_name=os.getenv("CLEARML_PROJECT", "Diatom Classification"),
            task_name=os.getenv("CLEARML_TASK_NAME", default_task_name),
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
            output_uri=_output_uri(),
            auto_connect_frameworks={"pytorch": True, "tensorboard": True},
        )
        task.connect(dict(config), name="training_config")
        task.add_tags(["diatoms", "computer-vision"])
        return task
    except Exception as exc:
        if _env_flag("CLEARML_STRICT"):
            raise
        print(f"ClearML недоступен, обучение продолжится без трекинга: {exc}")
        return None


def register_model_file(
    task: Any,
    weights_path: str | Path,
    name: str,
    class_names: list[str],
    iteration: int | None = None,
) -> None:
    """Регистрирует локальный checkpoint как output model ClearML."""
    if task is None:
        return

    from clearml import OutputModel

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {path}")

    model = OutputModel(
        task=task,
        name=name,
        framework="PyTorch",
        label_enumeration={label: index for index, label in enumerate(class_names)},
    )
    model.update_weights(
        weights_filename=str(path),
        iteration=iteration,
        auto_delete_file=False,
        async_enable=False,
    )
