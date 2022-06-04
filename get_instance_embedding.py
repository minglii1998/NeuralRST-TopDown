import torch
from torch.autograd import Variable

from in_out.preprocess_lm import batch_data_variable

from models.WordEncoder_lm import WordEncoder
word_encoder = WordEncoder({'encoder':'t5', 'word_dim':1024}).cuda()

vocab = torch.load('saved_t5_embeddings/t5_vocab.pt')
config = torch.load('saved_t5_embeddings/t5_config.pt')
train_ins = torch.load('saved_t5_embeddings/t5token_train_instances.pt')
dev_ins = torch.load('saved_t5_embeddings/t5token_dev_instances.pt')
test_ins = torch.load('saved_t5_embeddings/t5token_test_instances.pt')

def get_embedding(tokens):
    word = word_encoder(tokens)
    return word

def get_embedding_instances(instances):
    for i in range(len(instances)):
        print(i)
        ins_i = instances[i]
        embedding = get_embedding(ins_i.tokens)
        instances[i].token_embedding = embedding
    return instances

train_ins_ = get_embedding_instances(train_ins)
dev_ins_ = get_embedding_instances(dev_ins)
test_ins_ = get_embedding_instances(test_ins)

torch.save(train_ins_,'saved_t5_embeddings/t5_train_instances_1024.pt')
torch.save(dev_ins_,'saved_t5_embeddings/t5_dev_instances_1024.pt')
torch.save(test_ins_,'saved_t5_embeddings/t5_test_instances_1024.pt')

pass