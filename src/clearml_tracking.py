"""Утилиты для опционального трекинга обучения в ClearML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional


REQUIRED_CLEARML_VARS = (
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.startswith("${") and normalized.endswith("}"):
        return None
    return normalized


def _load_clearml_env_from_dotenv() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("CLEARML_"):
            continue

        normalized = _normalize_env_value(value)
        if normalized is None:
            continue
        if _normalize_env_value(os.getenv(key)) is None:
            os.environ[key] = normalized


def _validate_clearml_credentials() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_CLEARML_VARS:
        if _normalize_env_value(os.getenv(name)) is None:
            missing.append(name)
    return missing


def _output_uri() -> Any:
    value = _normalize_env_value(os.getenv("CLEARML_OUTPUT_URI")) or "true"
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
    _load_clearml_env_from_dotenv()

    if not _env_flag("CLEARML_ENABLED"):
        return None

    missing_credentials = _validate_clearml_credentials()
    if missing_credentials:
        message = (
            "Не заданы обязательные credentials ClearML: "
            + ", ".join(missing_credentials)
            + ". Передайте их через env vars DataSphere или в .env."
        )
        if _env_flag("CLEARML_STRICT"):
            raise RuntimeError(message)
        print(f"{message} Обучение продолжится без трекинга.")
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
