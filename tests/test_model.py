from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def classify_tweet(tweet):
    d = {0: "no-hate-speech", 1: "hate-speech"}
    device = "cpu"
    inputs = tokenizer(
        [tweet],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model(input_ids, attention_mask=attention_mask)
    pred = torch.argmax(output.logits[0]).item()
    return d[pred]


n_labels = 2
checkpoint = "models/checkpoint-16782"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=n_labels
)

assert model.classifier.out_features == 2, "There should be two output features"

assert isinstance(
    classify_tweet("Life is nice"), str
), "Classifier should output hate speech or no hate speech"


# print(classify_tweet('you asshole piece of shit'))
# print(classify_tweet('What a nice nice nice day'))