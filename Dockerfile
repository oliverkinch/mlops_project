FROM nvidia/cuda:10.2-base
CMD nvidia-smi

FROM python:3.8-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY src/ /app/src/
COPY models/ /app/models/
COPY .git/ /app/.git/
COPY .dvc/ /app/.dvc/
COPY data.dvc /app/data.dvc

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc --no-cache-dir
RUN pip install dvc[gs] --no-cache-dir
RUN pip install wandb --no-cache-dir 
RUN dvc pull

ENTRYPOINT ["python", "src/models/train_model.py"]