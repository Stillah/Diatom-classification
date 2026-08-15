"""Минимальная проверка подключения к ClearML из локальной среды или DataSphere."""

from __future__ import annotations

from clearml_tracking import init_clearml_task


def main() -> None:
    task = init_clearml_task(
        default_task_name="ClearML connectivity smoke",
        config={"smoke_test": True, "purpose": "connectivity"},
    )
    if task is None:
        raise RuntimeError(
            "ClearML Task не создан. Проверьте CLEARML_ENABLED и credentials."
        )

    try:
        logger = task.get_logger()
        logger.report_scalar(
            title="Smoke test",
            series="connection",
            value=1.0,
            iteration=0,
        )
        logger.report_text(
            "DataSphere/локальная среда успешно подключилась к ClearML.",
            print_console=True,
        )
        print(f"CLEARML_TASK_ID={task.id}")
    finally:
        task.close()


if __name__ == "__main__":
    main()
