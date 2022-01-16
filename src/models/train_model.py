import torch
import os, json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
from datasets import Dataset

import random

base_models = {
        "bert" : {
                "checkpoint": "Maltehb/danish-bert-botxo", 
                "save":"bert",
                "cased": False,
        },
        "electra-c": {
            "checkpoint": "Maltehb/-l-ctra-danish-electra-small-cased",
            "save":"electra-c",
            "cased": True,
        },
        "electra-u": {
            "checkpoint": "Maltehb/-l-ctra-danish-electra-small-uncased", 
            "save":"electra-u",
            "cased": False,
        },
        "xlmb": {
            "checkpoint": "xlm-roberta-base", 
            "save": "xlmb",
            "cased": True,
        },
        "xlml": {
            "checkpoint": "xlm-roberta-large", 
            "save": "xlml",
            "cased": True,
        }
}

def compute_metrics(eval_pred): 
    predictions, true_labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='macro')
    acc = accuracy_score(true_labels, preds)
    
    return {'accuracy': acc, 
            'f1': f1,
            'precision': precision,
            'recall': recall
            }

if __name__== "__main__":

    ### use GPU

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--model", default="bert", type=str)
    argparser.add_argument("-s", "--datadir", default='data/processed/', type=str)
    argparser.add_argument("-p", "--savepath", default="models", type=str)
    argparser.add_argument("-e", "--epochs", default=1, type=int)
    argparser.add_argument("-b", "--batch", default=32, type=int)
    argparser.add_argument("-l", "--maxlength", default=512, type=int)
    argparser.add_argument("-d", "--devset", default=0.2, type=float)
    args = argparser.parse_args()

    mname = args.model
    if mname not in base_models:
        exit(0)

    base_model = base_models[mname]
    data_dir = args.datadir

    epochs = args.epochs
    devset_ratio = args.devset
    assert(0<devset_ratio<1)

    assert(os.path.exists(args.savepath))
    save_dir = os.path.join(args.savepath, base_model["save"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ### DATA PREPROCESSING

    train = torch.load(data_dir + "train.pt")
    validation = torch.load(data_dir + "validation.pt")
    test = torch.load(data_dir + "test.pt")

    # df_train = pd.read_csv(data_dir + 'train.tsv', sep='\t')
    # df_test = pd.read_csv(data_dir + 'test.tsv', sep='\t')

    # df_train = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=',', engine='python', encoding='utf-8').dropna() ###
    # df_test = pd.read_csv(os.path.join(data_dir, "test.csv"), sep=',', engine='python', encoding='utf-8').dropna() ###

    labels = ['hate-speech', 'no-hate-speech'] ###
    labels_str2int = {l:i for i,l in enumerate(labels)}
    with open(os.path.join(save_dir, 'labels.json'), 'wt') as f:
        json.dump(labels_str2int, f)

    train_data = [t["tweet"].lower() if not base_model['cased'] else t["tweet"] for t in train] ###
    train_labels = [t["label"] for t in train] ###

    assert(len(train_data) == len(train_labels))

    validation_data = [t["tweet"].lower() if not base_model['cased'] else t["tweet"] for t in validation] ###
    validation_labels = [t["label"] for t in validation] ###

    assert(len(validation_data) == len(validation_labels))

    test_data = [t["tweet"].lower() if not base_model['cased'] else t["tweet"] for t in test] ###
    test_labels = [t["label"] for t in test] ###

    assert(len(test_data) == len(test_labels))

    ### LOAD PRE-TRAINED MODEL and TOKENIZER

    model_checkpoint = base_model["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=(not base_model['cased']))
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(labels))

    batch_size = args.batch
    max_length = args.maxlength

    ### CREATE DATASETS -- TOKENIZE

    train_dataset = Dataset.from_dict({"text":train_data, "class":train_labels})
    validation_dataset = Dataset.from_dict({"text":validation_data, "class":validation_labels})
    test_dataset = Dataset.from_dict({"text":test_data, "class":test_labels})

    def preprocess(examples):
        tokenized_batch = tokenizer(examples['text'], add_special_tokens=True, truncation=True, max_length=max_length)
        tokenized_batch["labels"] = [label for label in examples["class"]]
        return tokenized_batch

    train_encodings = train_dataset.map(preprocess, batched=True)
    dev_encodings = validation_dataset.map(preprocess, batched=True)
    test_encodings = test_dataset.map(preprocess, batched=True)

    ### TRAINING ARGUMENTS

    model = model.to(device)
    metric_name = "f1"

    
    t_args = TrainingArguments(
        save_dir, # directory : where to save the model
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = 16,
        num_train_epochs = epochs,
        weight_decay = 0.01,
        load_best_model_at_end = True,
        metric_for_best_model = metric_name,
    )

    ### TRAIN 

    print("#"*20+"\nTRAINING\n")
    print("model = ", mname)
    print("epochs = ", epochs)
    print("batch size =", batch_size)
    print("max length = ", max_length)
    print("train data = ", len(train_data))
    print("dev data = ", len(validation_data), "(", devset_ratio, "% of train )")
    print('\n')

    trainer = Trainer(
        model,
        t_args,
        train_dataset=train_encodings,
        eval_dataset=dev_encodings,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # SAVE MODEL

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    ### EVALUATE

    trainer.evaluate()

    predictions, labels, _ = trainer.predict(test_encodings)

    preds = predictions.argmax(-1)
    results = compute_metrics((predictions, labels))
    print(results)