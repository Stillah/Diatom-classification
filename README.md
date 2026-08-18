## Графики с обучением находятся в папке runs

## Запуск в датасфере

```bash
datasphere project job execute -p bt122c1et3ueiori55qf -c configs/config.yaml
```

Для ClearML в DataSphere убедитесь, что в окружении задания заданы
`CLEARML_API_ACCESS_KEY` и `CLEARML_API_SECRET_KEY`
(а также `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`, если требуется ваш инстанс).

## Запуск фронтенда с готовыми моделями

```bash
streamlit run src/streamlit_app.py
```