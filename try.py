from in_out.reader_lm import Reader

train_path = 'NeuralRST/rst.train312'
train_syn_feat_path = 'NeuralRST/SyntaxBiaffine/train.conll.dump.results'

reader = Reader(train_path, train_syn_feat_path, 't5')
train_instances  = reader.read_data()

import torch
train_instances = torch.load('train_instances.pt')
for i in range(len(train_instances)):
    ins_i = train_instances[i]
    word_list = ins_i.total_words
    if len(word_list) < 10:
        print(i) # 267
        # Part_NN of_IN a_DT Series_NN <P>
pass