FROM python:3.8-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

WORKDIR /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc
RUN pip install dvc[gs]
RUN dvc init --no-scm
RUN dvc pull

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]