import torch

data_dir = 'data/processed/'

data_test = torch.load(data_dir + 'test.pt')
data_train = torch.load(data_dir + 'train.pt')
data_valid = torch.load(data_dir + 'validation.pt')


data = [
    data_test, 
    data_train,
    data_valid
    ]


for d in data:
    sample = next(iter(d))
    assert isinstance(sample, dict)
    assert 'label' in sample.keys() and 'tweet' in sample.keys()
    assert isinstance(sample['label'], int)
    assert isinstance(sample['tweet'], str)

print('Data is good')