
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



def classify_tweet(tweet):
    d = {0: "no-hate-speech", 1: "hate-speech"}
    inputs = tokenizer(
        [tweet],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model(input_ids, attention_mask=attention_mask)
    pred = np.argmax(output.logits[0].detach().numpy())
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
