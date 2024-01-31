FROM python:3.10-slim-bullseye
ARG GITHUB_ACCESS_TOKEN
RUN apt-get update && apt-get install -y  \
    gcc \
    python3-dev \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR app/

COPY ./streamlit_app.py /app/streamlit_app.py
COPY ./src/shared/schema_generator.py /app/shared/schema_generator.py
COPY ./src/constants.py /app/constants.py
COPY ./data/test/fact_schema_v41.html /app/data/test/fact_schema_v41.html
COPY ./serialized_models/inselbert_qa_hf /app/serialized_models/inselbert_qa_hf
COPY ./.streamlit /app/.streamlit

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]