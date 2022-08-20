import torch
import torch.nn as nn
# from allennlp.modules.elmo import Elmo, batch_to_ids

from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel
from transformers import BartTokenizer, BartModel
from transformers import T5Tokenizer, T5EncoderModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import AlbertTokenizer, AlbertModel
from transformers import RobertaTokenizer, RobertaModel

device = 'cuda'

class WordEncoder(nn.Module):
    def __init__(self, config, long_out=False):
        super(WordEncoder, self).__init__()
        '''
        long_out: out put the whole doc word embeddings rather than batch them
        '''

        self.long_out = long_out
        self.num_pad = 0

        self.encoder_type = config['encoder']
        if self.encoder_type == 'bert':
            # default is large, since the encoding size is 1024
            # every sentence is encoded seperatly 
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased",cache_dir="../cache/")
            self.model = BertModel.from_pretrained("bert-large-uncased",cache_dir="../cache/")
            self.num_pad = 2
        elif self.encoder_type in ['longformer3']:
            # default is large, since the encoding size is 1024
            # (longformer) every sentence is TOKENIZED seperatly and ENCODED together # discarded
            # (longformer2) every sentence is TOKENIZED and ENCODED as a whole document # discarded
            self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096",cache_dir="../cache/")
            self.model =  LongformerModel.from_pretrained("allenai/longformer-large-4096",cache_dir="../cache/")
            self.num_pad = 2
        elif self.encoder_type in ['bart']:
            # though want to make 3 different type of bart, but the max size of 1024 is not that enough
            # so here only bart1 can be used
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large",cache_dir="../cache/")
            self.model =  BartModel.from_pretrained("facebook/bart-large",cache_dir="../cache/")
            self.num_pad = 2
        elif self.encoder_type in ['t51','t53']:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large",cache_dir="../cache/")
            self.model =  T5EncoderModel.from_pretrained("t5-large",cache_dir="../cache/")
            self.num_pad = 1
        elif self.encoder_type in ['xlnet']:
            # ignore
            self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased",cache_dir="../cache/")
            self.model =  XLNetModel.from_pretrained("xlnet-large-cased",cache_dir="../cache/")
        elif self.encoder_type in ['electra']:
            self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-large-discriminator",cache_dir="../cache/")
            self.model =  ElectraModel.from_pretrained("google/electra-large-discriminator",cache_dir="../cache/")
            self.num_pad = 2
        elif self.encoder_type in ['albert']:
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2",cache_dir="../cache/")
            self.model =  AlbertModel.from_pretrained("albert-large-v2",cache_dir="../cache/")
            self.num_pad = 2
        elif self.encoder_type in ['roberta']:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large",cache_dir="../cache/")
            self.model =  RobertaModel.from_pretrained("roberta-large",cache_dir="../cache/")
            self.num_pad = 2
        elif self.encoder_type == 'elmo':
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.model = Elmo(options_file, weight_file, 1, dropout=0.0, requires_grad=False)
            self.num_pad = 0

    def forward(self, sentence, length):

        if self.encoder_type == 'elmo':
            # character_ids = batch_to_ids(sentence) # shape: num_sentence,max_sentecne_len,word_len
            # character_ids = character_ids.to(device)
            # embedding = self.model(character_ids)
            # outp_ctxt = embedding['elmo_representations'][0] # shape: num_sentence,max_sentecne_len,dim
            # ctxt_mask = embedding['mask'] # shape: shape: num_sentence,max_sentecne_len

            character_ids = batch_to_ids(sentence).to(device)
            embedding = self.model(character_ids)

            outp_ctxt = embedding['elmo_representations'][0]
            ctxt_mask = embedding['mask']

            length_ = []
            for i in range(len(ctxt_mask)):
                num = sum(ctxt_mask[i])
                length_.append(num)
            length = torch.LongTensor(length_).to(device)
            all_len = sum(length) - len(length_)*self.num_pad + self.num_pad

            if self.long_out:

                embed_dim = outp_ctxt.shape[2]
                input_ids_made = torch.ones((1,all_len)).int()
                outp_ctxt_made = torch.ones((1,all_len,embed_dim))

                pointer_v = 0
                for i in range(ctxt_mask.shape[0]):
                    outp_ctxt_made[0,pointer_v:pointer_v+length[i],:] = outp_ctxt[i,0:length[i],:]
                    pointer_v += length[i]
            
                out_temp = outp_ctxt_made.to(device)
                # long embedding, long token
                # batched embedding, batched mask, batched len
                # long_id seems not necessary
                return out_temp, None, self.num_pad, outp_ctxt, ctxt_mask, length 

            return outp_ctxt, ctxt_mask, length

        # Type 1: sentences are tokenized and encoded seperately
        elif self.encoder_type in ['bert','bart','t51','xlnet','electra','albert','roberta']:
            batch_sent_str = []
            for sent_list in sentence:
                sent_str = ' '.join(sent_list)
                batch_sent_str.append(sent_str)

            toke = self.tokenizer(batch_sent_str,padding=True,return_tensors="pt").to(device)

            # for i in range(len(toke["input_ids"])):
            #     toke_decode = self.tokenizer.decode(toke["input_ids"][i])
            #     print(toke_decode)

            ctxt_mask = toke['attention_mask']
            length_ = []
            for i in range(len(ctxt_mask)):
                num = sum(ctxt_mask[i])
                length_.append(num)
            length = torch.LongTensor(length_).to(device)
            all_len = sum(length) - len(length_)*self.num_pad + self.num_pad

            with torch.no_grad():
                outp_ctxt = self.model(**toke)['last_hidden_state']

            if self.long_out:

                embed_dim = outp_ctxt.shape[2]

                pointer_v = 1
                input_ids_made = torch.ones((1,all_len)).int()
                outp_ctxt_made = torch.ones((1,all_len,embed_dim))
                input_ids_made[0,0] = toke["input_ids"][0,0]
                outp_ctxt_made[0,0,:] = outp_ctxt[0,0,:]
                for i in range(ctxt_mask.shape[0]):
                    input_ids_made[0,pointer_v:pointer_v+length[i]-self.num_pad] = toke["input_ids"][i,1:length[i]-self.num_pad+1]
                    outp_ctxt_made[0,pointer_v:pointer_v+length[i]-self.num_pad,:] = outp_ctxt[i,1:length[i]-self.num_pad+1,:]
                    pointer_v += (length[i]-self.num_pad)
                if self.num_pad == 2:
                    input_ids_made[0,pointer_v] = toke["input_ids"][0,length[0]-1]
                    outp_ctxt_made[0,pointer_v,:] = outp_ctxt[0,length[0]-1,:]
                elif self.num_pad == 1:
                    input_ids_made[0,pointer_v-1] = toke["input_ids"][0,length[0]-1]
                    outp_ctxt_made[0,pointer_v-1,:] = outp_ctxt[0,length[0]-1,:]

            
                out_temp = outp_ctxt_made.to(device)
                # long embedding, long token
                # batched embedding, batched mask, batched len
                return out_temp, input_ids_made, self.num_pad, outp_ctxt, ctxt_mask, length 

            return outp_ctxt, ctxt_mask, length

        # Type 2: sentences are tokenized seperately and concated together, then encoded together
        # This type is discarded, no much performance or insight
        elif self.encoder_type in ['xxx']:
            batch_sent_str = []
            for sent_list in sentence:
                sent_str = ' '.join(sent_list)
                batch_sent_str.append(sent_str)
                
            toke = self.tokenizer(batch_sent_str,padding=True,return_tensors="pt").to(device)
            ctxt_mask = toke['attention_mask']
            length = []
            for i in range(len(ctxt_mask)):
                num = sum(ctxt_mask[i])
                length.append(num)
            length = torch.LongTensor(length).to(device)

            bs, num_sent = toke['input_ids'].shape[0], toke['input_ids'].shape[1]
            toke['input_ids'] = toke['input_ids'].view(-1).unsqueeze(0)
            toke['attention_mask'] = toke['attention_mask'].view(-1).unsqueeze(0)
            with torch.no_grad():
                outp_ctxt = self.model(**toke)['last_hidden_state']
                outp_ctxt = outp_ctxt.view(bs,num_sent,-1)

            return outp_ctxt, ctxt_mask, length

        # Type 3: sentences are tokenized and encoded together as a whole document
        # The key point here is to encode the document as a whole
        # type 2 inserts a lot of useless padding tokens, which spoil the document structure
        # However, in order to soleve the possible problem of mismatching, 
        # the sentences will be tokenized seperately and organzied as a new input_ids, after removing padding
        # then encoded as a whole document
        # longformer used to called as longformer2, now change to longformer3
        elif self.encoder_type in ['longformer3','t53']:

            batch_sent_str = []
            for sent_list in sentence:
                sent_str = ' '.join(sent_list)
                batch_sent_str.append(sent_str)
                
            toke_ = self.tokenizer(batch_sent_str,padding=True,return_tensors="pt").to(device)
            ctxt_mask = toke_['attention_mask']
            with torch.no_grad():
                outp_ctxt = self.model(**toke_)['last_hidden_state']

            length_ = []
            for i in range(len(ctxt_mask)):
                num = sum(ctxt_mask[i])
                length_.append(num)
            length = torch.LongTensor(length_).to(device)
            all_len = sum(length) - len(length_)*self.num_pad + self.num_pad
            
            pointer_v = 1
            input_ids_made = torch.ones((1,all_len)).int()
            input_ids_made[0,0] = toke_["input_ids"][0,0]
            for i in range(ctxt_mask.shape[0]):
                input_ids_made[0,pointer_v:pointer_v+length[i]-self.num_pad] = toke_["input_ids"][i,1:length[i]-self.num_pad+1]
                # debug
                # x1 = input_ids_made[0,pointer_v:pointer_v+length[i]-self.num_pad]
                # x2 = toke_["input_ids"][i,1:length[i]-self.num_pad+1]
                # print('x1',x1)
                # print('x2',x2)
                pointer_v += (length[i]-self.num_pad)
            if self.num_pad == 2:
                input_ids_made[0,pointer_v] = toke_["input_ids"][0,length[0]-1]
            elif self.num_pad == 1:
                input_ids_made[0,pointer_v-1] = toke_["input_ids"][0,length[0]-1]

            toke_made = toke_.copy()
            toke_made["input_ids"] = input_ids_made
            toke_made["attention_mask"] = torch.ones_like(input_ids_made).int()
            toke_made.to(device)
            with torch.no_grad():
                out_temp = self.model(**toke_made)['last_hidden_state']

            if self.long_out:
                # long embedding, long token
                # batched embedding, batched mask, batched len
                return out_temp, input_ids_made, self.num_pad, outp_ctxt, ctxt_mask, length 

            pointer_v = 1
            for i in range(outp_ctxt.shape[0]):
                outp_ctxt[i,1:length[i]-self.num_pad+1,:] = out_temp[0,pointer_v:pointer_v+length[i]-self.num_pad,:]
                pointer_v += (length[i]-self.num_pad)


            return outp_ctxt, ctxt_mask, length