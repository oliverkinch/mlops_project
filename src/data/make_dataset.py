from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from typing import Tuple


def make_data_split(
    data: Dataset, train_split: float = 0.7, validation_split: float = 0.15, test_split: float = 0.15
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


def main():
    """
    Description:
        Downloading and saving the raw data to raw data folder.
        Diving the raw data into training, validation and testing
        and saving these into the processed data folder
    """
    data = load_dataset("tweets_hate_speech_detection", split="train")
    torch.save(data, "data/raw/data.pt")

    train_split = 0.7
    validation_split = 0.15
    test_split = 0.15

    train, validation, test = make_data_split(
        data, train_split, validation_split, test_split
    )

    torch.save(train, "data/processed/train.pth")
    torch.save(validation, "data/processed/validation.pth")
    torch.save(test, "data/processed/test.pth")


if __name__ == "__main__":
    main()
