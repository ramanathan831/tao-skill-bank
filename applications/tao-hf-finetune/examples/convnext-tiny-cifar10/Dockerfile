ARG NGC_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${NGC_IMAGE}

ENTRYPOINT ["/bin/bash", "-c"]
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY config.yaml ./
