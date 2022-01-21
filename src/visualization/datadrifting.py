import copy
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import sklearn.manifold as manifold
import torch
import torchdrift
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TweetDataset(Dataset):
    def __init__(self, tweets, labels, transform=None):

        self.labels = labels
        self.tweets = tweets
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]
        if self.transform:
            tweet = self.transform(tweet)
        return tweet, label


def classify_tweet(tweet):
    d = {0: "no-hate-speech", 1: "hate-speech"}
    device = "cpu"
    inputs = tokenizer(
        [tweet["text"]],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    corr_input_ids = corruption_function(copy.deepcopy(input_ids))
    attention_mask = inputs.attention_mask.to(device)
    output = model(input_ids, attention_mask=attention_mask)
    corr_output = model(corr_input_ids, attention_mask=attention_mask)
    p_no_hate = torch.exp(output.logits[0][0]) / torch.sum(torch.exp(output.logits[0]))
    corr_p_no_hate = torch.exp(corr_output.logits[0][0]) / torch.sum(
        torch.exp(corr_output.logits[0])
    )
    pred = torch.argmax(output.logits[0]).item()
    corr_pred = torch.argmax(corr_output.logits[0]).item()
    return p_no_hate, corr_p_no_hate, d[pred], d[corr_pred], input_ids, corr_input_ids


def corruption_function(x: torch.Tensor):
    x[x != 0] = x[x != 0] + torch.randint(
        low=-1000, high=1000, size=x[x != 0].shape
    ) * torch.randint(low=0, high=2, size=x[x != 0].shape)
    x[x < 0] = 1
    x[x > 28996] = 28996
    return x


def make_data_split(
    data: Dataset,
    train_split: float = 0.7,
    validation_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[Subset, Subset, Subset]:
    """
    Description:
        Making train, validation, test split of data and returning these.

    Inputs:
        data: The dataset used to make the splits.
            type: Dataset
        train_split: the amount of data used for training.
            type: float between 0 and 1
        validation_split: the amount of data used for validation.
            type: float between 0 and 1
        test_split: the amount of data used for testing.
            type: float between 0 and 1
    """

    total_split = train_split + validation_split + test_split

    train_split = train_split / total_split
    validation_split = validation_split / total_split
    test_split = test_split / total_split

    train_len = int(len(data) * train_split)
    validation_len = int(len(data) * validation_split)
    test_len = int(len(data) * test_split)
    # Making sure that lengths add up, by adding the rest of the data to the training split.
    train_len = train_len + len(data) - (train_len + validation_len + test_len)

    train, validation, test = torch.utils.data.random_split(
        data, [train_len, validation_len, test_len]
    )

    return train, validation, test


if __name__ == "__main__":

    seed = 13500
    torch.manual_seed(seed)
    random.seed(seed)

    data = load_dataset("tweets_hate_speech_detection", split="train")

    train, validation, test = make_data_split(data)
    cwd = os.getcwd().split("tests")[0]
    data_dir = cwd + "/data/processed/"

    data_test = torch.load(data_dir + "test.pth")

    checkpoint = "models/bert/final_checkpoint"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    test_data = [t["tweet"] for t in test]
    test_data_tokenized = torch.stack(
        [
            tokenizer(
                [t],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )["input_ids"]
            for t in test_data
        ]
    )
    test_labels = [t["label"] for t in test]
    test_labels_tokenized = torch.stack([torch.Tensor([t]) for t in test_labels])

    test_dataset = Dataset.from_dict({"text": test_data, "class": test_labels})
    tokenized_test_dataset = Dataset.from_dict(
        {"text": test_data_tokenized, "class": test_labels}
    )
    iter_dataset = iter(test_dataset)

    test_tokenized_dataset = TweetDataset(test_data_tokenized, test_labels_tokenized)

    model.eval()

    inputs = []

    for i in range(10):
        inp = next(iter_dataset)
        inputs.append(copy.deepcopy(inp))

    d = {0: "no-hate-speech", 1: "hate-speech"}
    inputs_pred = []
    inputs_ood_pred = []
    inputs_p_no_hate = []
    inputs_ood_p_no_hate = []
    correct_pred = []
    all_input_ids = []
    all_corr_input_ids = []

    for inp in inputs:
        (
            p_no_hate,
            corr_p_no_hate,
            pred,
            corr_pred,
            input_ids,
            corr_input_ids,
        ) = classify_tweet(inp)
        correct_pred.append(d[inp["class"]])
        inputs_pred.append(pred)
        inputs_ood_pred.append(corr_pred)
        inputs_ood_p_no_hate.append(corr_p_no_hate)
        inputs_p_no_hate.append(p_no_hate)
        all_input_ids.append(input_ids)
        all_corr_input_ids.append(corr_input_ids)

    dataloader = DataLoader(test_tokenized_dataset.tweets.view(-1, 512), batch_size=8)
    feature_extractor = copy.deepcopy(model).bert
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(dataloader, feature_extractor, drift_detector, num_batches=10)
    drift_detection_model = torch.nn.Sequential(feature_extractor, drift_detector)

    features = feature_extractor(torch.stack(all_input_ids).view(-1, 512))
    features = features.last_hidden_state
    features_ood = feature_extractor(torch.stack(all_corr_input_ids).view(-1, 512))
    features_ood = features_ood.last_hidden_state
    score = drift_detector(features.view(len(all_input_ids), -1))
    score_ood = drift_detector(features_ood.view(len(all_corr_input_ids), -1))
    p_val = drift_detector.compute_p_value(features.view(len(all_input_ids), -1))
    p_val_ood = drift_detector.compute_p_value(
        features_ood.view(len(all_corr_input_ids), -1)
    )

    N_base = drift_detector.base_outputs.size(0)
    mapper = manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs.view(N_base, -1))
    features_embedded = mapper.transform(features.view(features.shape[0], -1).detach())
    features_ood_embedded = mapper.transform(
        features_ood.view(features_ood.shape[0], -1).detach()
    )

    plt.figure()
    plt.scatter(
        base_embedded[:, 0], base_embedded[:, 1], s=2, c="r", label="base_embedding"
    )
    plt.scatter(
        features_embedded[:, 0],
        features_embedded[:, 1],
        s=4,
        c="b",
        label="features_in_dist",
    )
    plt.title(f"score {score:.2f} p-value {p_val:.2f}")
    plt.legend()
    plt.savefig(
        "/reports/figures/datadrift_in_dist.pdf"
    )

    plt.figure()
    plt.scatter(
        base_embedded[:, 0], base_embedded[:, 1], s=2, c="r", label="base_embedding"
    )
    plt.scatter(
        features_ood_embedded[:, 0],
        features_ood_embedded[:, 1],
        s=4,
        c="g",
        label="features_ood",
    )
    plt.title(f"score {score_ood:.2f} p-value {p_val_ood:.2f}")
    plt.legend()
    plt.savefig(
        "/reports/figures/datadrift_ood.pdf"
    )
