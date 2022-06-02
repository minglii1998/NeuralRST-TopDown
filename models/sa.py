from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cpu'

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mid_dim):
        super(SelfAttention, self).__init__()

        self.ws1 = nn.Linear(hidden_size, mid_dim)
        self.ws2 = nn.Linear(mid_dim, 1)

        self.drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_tensor, word_mask):
        '''
        typically input should be: batch_size, edu_size, word_in_edu, hidden_size
        '''
        batch_size, edu_size, word_in_edu, hidden_size = input_tensor.shape # hidden_size 400
        input_tensor = input_tensor.view(batch_size * edu_size, word_in_edu, hidden_size)
        word_mask = word_mask.view(batch_size * edu_size, word_in_edu)

        self_attention = F.tanh(self.ws1(self.drop(input_tensor)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze()
        self_attention = self_attention + -10000*(word_mask == 0).float()
        self_attention = self.softmax(self_attention)
        weighted_embedding = torch.sum(input_tensor*self_attention.unsqueeze(-1), dim=1)

        return weighted_embedding


class DocSelfAttention(nn.Module):
    def __init__(self, hidden_size, mid_dim):
        super(DocSelfAttention, self).__init__()

        '''
        self attention should be done after word and syntax are concate together
        '''

        self.ws1 = nn.Linear(hidden_size, mid_dim)
        self.ws2 = nn.Linear(mid_dim, 1)
        self.ws3 = nn.Linear(hidden_size, mid_dim)

        self.drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_all, word_weighted, syn_all, syn_weighted, word_mask):

        batch_size, edu_size, word_in_edu, word_dim = word_all.shape 
        _, _, _, syn_dim = syn_all.shape 
        word_all = word_all.view(batch_size, edu_size * word_in_edu, -1)
        syn_all = syn_all.view(batch_size, edu_size * word_in_edu, -1)
        word_mask = word_mask.view(batch_size, edu_size * word_in_edu)

        out_holder = torch.zeros_like(torch.cat([word_weighted, syn_weighted], dim=-1)).to(device)

        for i in range(batch_size):
            mask = word_mask[i].unsqueeze(0).unsqueeze(-1).bool()
            word_all_ = torch.masked_select(word_all[i], mask).view(1,-1, word_dim)
            syn_all_ = torch.masked_select(syn_all[i], mask).view(1,-1, syn_dim)

            emb_all = torch.cat([word_all_, syn_all_], dim=-1)

            self_attention = emb_all - torch.cat([word_weighted[i], syn_weighted[i]], dim=-1).unsqueeze(1)
            self_attention = F.tanh(self.ws1(self.drop(self_attention)))
            self_attention = self.ws2(self.drop(self_attention)).squeeze(-1)
            self_attention = self.softmax(self_attention)
            self_attention = torch.sum(emb_all*self_attention.unsqueeze(-1), dim=1)
            out_holder[i] = self_attention
            pass

        out_holder = out_holder + torch.cat([word_weighted, syn_weighted], dim=-1)
        out_holder = self.ws3(out_holder)

        return out_holder



    def forward_loop(self, word_all, word_weighted, syn_all, syn_weighted, word_mask):

        batch_size, edu_size, word_in_edu, word_dim = word_all.shape 
        _, _, _, syn_dim = syn_all.shape 
        word_all = word_all.view(batch_size, edu_size * word_in_edu, -1)
        syn_all = syn_all.view(batch_size, edu_size * word_in_edu, -1)
        word_mask = word_mask.view(batch_size, edu_size * word_in_edu)

        out_holder = torch.zeros_like(torch.cat([word_weighted, syn_weighted], dim=-1)).to(device)

        for i in range(batch_size):
            mask = word_mask[i].unsqueeze(0).unsqueeze(-1).bool()
            mask_sum = mask.sum() 
            # real num of word embedding in this instance
            word_all_ = torch.masked_select(word_all[i], mask).view(-1, word_dim)
            syn_all_ = torch.masked_select(syn_all[i], mask).view(-1, syn_dim)
            emb_all = torch.cat([word_all_, syn_all_], dim=-1)
            emb_weighted = torch.cat([word_weighted[i], syn_weighted[i]], dim=-1)

            self_att_holder = torch.zeros(edu_size,mask_sum).to(device)

            for j in range(emb_weighted.shape[0]):
                temp = emb_all - emb_weighted[j] 

                self_attention = F.tanh(self.ws1(self.drop(temp.unsqueeze(0).to(device))))
                self_attention = self.ws2(self.drop(self_attention))
                self_attention = self.softmax(self_attention).squeeze()
                self_att_holder[j] = self_attention
                pass

            emb_weighted = torch.sum(emb_all*self_att_holder.unsqueeze(-1), dim=1)
            out_holder[i] = emb_weighted
            pass

        emb_weighted = self.ws3(out_holder)
        emb_weighted = emb_weighted + torch.cat([word_weighted, syn_weighted], dim=-1)

        return emb_weighted


if __name__ == '__main__':
    sa = SelfAttention(20,20)
    input_t = torch.rand((2,4,5,20))