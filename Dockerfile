FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR /app
COPY "Pipfile" /app
COPY "Pipfile.lock" /app

RUN pipenv install --system --deploy

COPY "predict.py" /app

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]