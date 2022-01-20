FROM nvidia/cuda:10.2-base
CMD nvidia-smi

FROM python:3.8-slim
WORKDIR /app

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get install wget

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup




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