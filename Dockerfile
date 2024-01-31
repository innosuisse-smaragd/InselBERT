# build stage
FROM python:3.11-slim-bullseye as builder
RUN apt-get update && apt-get install -y gcc python3-dev git && apt-get clean && rm -rf /var/lib/apt/lists/*
# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# copy files
COPY pyproject.toml pdm.lock README.md streamlit_app.py /project/
COPY src/shared /project/src/shared
COPY serialized_models/inselbert_qa_hf /project/serialized_models/inselbert_qa_hf
COPY ./.streamlit /project/.streamlit



# install dependencies and project into the local packages directory
WORKDIR /project
RUN mkdir __pypackages__ && pdm sync --prod --no-editable


# run stage
FROM python:3.11-slim-bullseye

ENV LISTEN_PORT 8501

EXPOSE 8501
# retrieve packages from build stage
ENV PYTHONPATH=/project/pkgs
COPY --from=builder /project/__pypackages__/3.11/lib /project/pkgs

# retrieve executables
COPY --from=builder /project/__pypackages__/3.11/bin/* /bin/

# set command/entrypoint, adapt to fit your needs
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8080"]