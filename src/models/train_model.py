import json
import logging
import os
from typing import Tuple

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data.dataset import Subset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

import wandb
from src.data.make_dataset import make_data_split


MODEL_FILE_NAME = "bert.model"

base_models = {"bert": {"checkpoint": "bert-base-cased", "save": "bert", "cased": True}}

def compute_metrics(eval_pred: torch.Tensor) -> dict:
    predictions, true_labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro"
    )
    acc = accuracy_score(true_labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Log in Wandb
docker_api = os.environ.get("WANDB_API")
wandb.login(key=docker_api)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config):
    print(f"Training configuration: \n {OmegaConf.to_yaml(config)}")
    # use GPU
    with wandb.init(
        project=config["model"]["name"], config=dict(config["hyperparameters"])
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.empty_cache()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        mname = config["model"]["name"]
        if mname not in base_models:
            print("Model name not in base models")
            exit(0)
        cwd = os.getcwd().split("outputs/")[0]
        base_model = base_models[mname]
        # data_dir = cwd + config["dirs"]["data"]

        epochs = wandb.config["epochs"]
        devset_ratio = wandb.config["devset_ratio"]
        assert 0 < devset_ratio < 1

        assert os.path.exists(
            cwd + config["dirs"]["models"]
        ), f"cwd: {cwd}, dir:{config['dirs']['models']}"
        save_dir = os.path.join(cwd + config["dirs"]["models"], base_model["save"])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # DATA PREPROCESSING
        data = load_dataset("tweets_hate_speech_detection", split="train")

        train, validation, test = make_data_split(data)

        # train = torch.load(data_dir + "train.pth")
        # validation = torch.load(data_dir + "validation.pth")
        # test = torch.load(data_dir + "test.pth")

        labels = ["hate-speech", "no-hate-speech"]
        labels_str2int = {l: i for i, l in enumerate(labels)}
        with open(os.path.join(save_dir, "labels.json"), "wt") as f:
            json.dump(labels_str2int, f)

        train_data = [
            t["tweet"].lower() if not base_model["cased"] else t["tweet"] for t in train
        ]
        train_labels = [t["label"] for t in train]

        assert len(train_data) == len(train_labels)

        validation_data = [
            t["tweet"].lower() if not base_model["cased"] else t["tweet"]
            for t in validation
        ]
        validation_labels = [t["label"] for t in validation]

        assert len(validation_data) == len(validation_labels)

        test_data = [
            t["tweet"].lower() if not base_model["cased"] else t["tweet"] for t in test
        ]
        test_labels = [t["label"] for t in test]

        assert len(test_data) == len(test_labels)

        # LOAD PRE-TRAINED MODEL and TOKENIZER

        model_checkpoint = base_model["checkpoint"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, do_lower_case=(not base_model["cased"])
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=len(labels)
        )

        if not wandb.config["pretrained"]:
            model.init_weights()

        batch_size = wandb.config["batch_size"]
        max_length = wandb.config["maxlength"]

        # CREATE DATASETS -- TOKENIZE

        train_dataset = Dataset.from_dict({"text": train_data, "class": train_labels})
        validation_dataset = Dataset.from_dict(
            {"text": validation_data, "class": validation_labels}
        )
        test_dataset = Dataset.from_dict({"text": test_data, "class": test_labels})

        def preprocess(examples):
            tokenized_batch = tokenizer(
                examples["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            tokenized_batch["labels"] = [label for label in examples["class"]]
            return tokenized_batch

        train_encodings = train_dataset.map(preprocess, batched=True)
        dev_encodings = validation_dataset.map(preprocess, batched=True)
        test_encodings = test_dataset.map(preprocess, batched=True)

        # TRAINING ARGUMENTS

        model = model.to(device)
        metric_name = wandb.config["metric_name"]

        t_args = TrainingArguments(
            save_dir,  # directory : where to save the model
            evaluation_strategy=wandb.config["evaluation_strategy"],
            save_strategy=wandb.config["save_strategy"],
            learning_rate=wandb.config["lr"],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=wandb.config["per_device_eval_batch_size"],
            num_train_epochs=epochs,
            weight_decay=wandb.config["weight_decay"],
            load_best_model_at_end=wandb.config["load_best_model_at_end"],
            metric_for_best_model=metric_name,
        )

        # TRAIN

        print("#" * 20 + "\nTRAINING\n")
        print("model = ", mname)
        print("epochs = ", epochs)
        print("batch size =", batch_size)
        print("max length = ", max_length)
        print("train data = ", len(train_data))
        print("dev data = ", len(validation_data), "(", devset_ratio, "% of train )")
        print("\n")

        trainer = Trainer(
            model,
            t_args,
            train_dataset=train_encodings,
            eval_dataset=dev_encodings,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        wandb.watch(model)

        trainer.train()

        # SAVE MODEL

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        torch.save(model.state_dict(), cwd + "models/bert.pth")
        print("MODEL SAVED")

        # if config['dirs']['cloud']:
        #     tmp_model_file = os.path.join('/tmp', MODEL_FILE_NAME)
        #     torch.save(model.state_dict(), tmp_model_file)
        #     subprocess.check_call([
        #         'gsutil', 'cp', tmp_model_file,
        #         os.path.join(config['dirs']['cloud'], MODEL_FILE_NAME)])
        # EVALUATE

        trainer.evaluate()

        predictions, labels, _ = trainer.predict(test_encodings)

        results = compute_metrics((predictions, labels))
        print(results)

        print("#" * 30 + " DONE " + "#" * 30)


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    main()
