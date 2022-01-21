def entry(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return classify_tweet(request.args.get('message'))
    elif request_json and 'message' in request_json:
        return classify_tweet(request_json['message'])
    else:
        return 'No message supplied'


def classify_tweet(tweet):
    from google.cloud import storage
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import os

    BUCKET_NAME = "mlopsproject_bucket"
    blob_prefix = "checkpoint-16782"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=blob_prefix)
    test_path = '/tmp/test'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for blob in blobs:
        blob.download_to_filename(test_path + '/' + os.path.basename(blob.name))

    n_labels = 2
    checkpoint = test_path
    print("Got here at least")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=n_labels
    )

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
