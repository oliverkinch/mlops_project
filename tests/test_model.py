from transformers import AutoTokenizer, T5ForConditionalGeneration

def classify_tweet(tweet):
    device = 'cpu'
    inputs = tokenizer([tweet], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)


base_models = {
        'byt5': 
        {
            'checkpoint': 'Narrativa/byt5-base-tweet-hate-detection',
            'save': 'byt5',
        }
}

c = base_models['byt5']['checkpoint']
tokenizer = AutoTokenizer.from_pretrained(c)
model = T5ForConditionalGeneration.from_pretrained(c)

nice_result = classify_tweet('what a nice nice nice nice day')
bad_result = classify_tweet('Are you stupid, asshole?')

assert nice_result == 'no-hate-speech', 'Wrong prediction, is the model trained?'
assert bad_result == 'hate-speech', 'Wrong prediction, is the model trained?'
