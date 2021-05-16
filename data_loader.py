"""
data_loader
Author: Wei
Date: 2021/4/24
"""
import torch
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
import random
import json
from transformers import BertTokenizer, BertModel




class NYTDataset(Data.Dataset):
    def __init__(self, root, N, batch_num, support_shot, query_shot, mode):
        self.root = root
        self.N = N
        self.support_shot=support_shot
        self.query_shot=query_shot
        path = root
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert 0
        if support_shot <= 0 or query_shot <= 0:
            print("[ERROR] support shots and query shots must be larger than 0!")
            assert 0
        self.Data = json.load(open(path))
        self.classes = list(self.Data.keys())
        if mode == 'train':
            self.classes = self.classes[:17]
        else:
            self.classes = self.classes[17:22]
        self.batch_num = batch_num

        # sentence=self.json_data[self.classes[0]][0]['sentText']
        # print(sentence)
        # tokenzier = BertTokenizer.from_pretrained('bert-base-cased')
        # tok=tokenzier.tokenize(sentence)
        # input_ids = tokenzier.convert_tokens_to_ids(tok)
        # print(tok)
        # print(input_ids)

    def find_label(self, c, sent, ent1, ent2, labels):
        lbs = np.zeros((len(labels)))
        for rel in self.Data[c][sent]['relationMentions']:
            if rel['em1Text'] == ent1 and rel['em2Text'] == ent2 and rel['label'] in labels:
                lbs[labels.index(rel['label'])] = 1
        return lbs

    def __getitem__(self, index):
        tokenzier = BertTokenizer.from_pretrained('bert-base-cased')
        target_classes = random.sample(self.classes, self.N)
        sentences = [[] for i in range(self.N)]
        # index of instance sentence
        idx_of_instances=[[] for i in range(self.N)]
        labels = []
        for i in range(len(target_classes)):
            for sent in self.Data[target_classes[i]]:
                sentences[i].append(sent['sentText'])
        idx_val=[[] for i in range(self.N)]
        for i in range(len(target_classes)):
            idx_val[i]=random.sample(list(enumerate(sentences[i])), self.support_shot+self.query_shot)
        sample_sentences=[]
        for i, i_v in enumerate(idx_val):
            for ints in i_v:
                idx_of_instances[i].append(ints[0])
                sample_sentences.append(ints[1])
        print("class: {}, sentenceID: {}".format(target_classes[0], idx_of_instances[0][0]))
        print(sample_sentences[0])

        max_len = 0
        shots=self.support_shot+self.query_shot
        # get labels from support set and get max_len
        for n in range(self.N):
            for i in range(n*shots,(n+1)*shots):
                sample_sentences[i] = tokenzier.convert_tokens_to_ids(tokenzier.tokenize(sample_sentences[i]))
                max_len = max(max_len, len(sample_sentences[i]))
                if shots*n+self.support_shot<=i<shots*n+shots:
                    continue
                c = target_classes[n]
                idx=idx_of_instances[n][i-n*shots]
                for rel in self.Data[c][idx]['relationMentions']:
                    if rel['label'] not in labels:
                        labels.append(rel['label'])

        label_num = len(labels)
        self.labels = labels
        print('max_len is: {}'.format(max_len))
        mask = []
        entities = []
        context = []
        label = []
        for n in range(self.N):
            for i in range(n*shots,(n+1)*shots):
                tmp_mask = len(sample_sentences[i]) * [1]
                if len(sample_sentences[i]) < max_len:
                    padding = (max_len - len(sample_sentences[i])) * [0]
                    sample_sentences[i].extend(padding)
                    tmp_mask.extend(padding)
                mask.append(tmp_mask)

                c = target_classes[n]
                idx = idx_of_instances[n][i-n*shots]
                # entities of sentence i
                e_s = []
                for ent in self.Data[c][idx]['entityMentions']:
                    if ent['text'] not in e_s:
                        e_s.append(ent['text'])
                # entity masks of sentence i
                ents = np.zeros((len(e_s), max_len))
                for id, ent in enumerate(e_s):
                    idxs = []
                    token = tokenzier.convert_tokens_to_ids(tokenzier.tokenize(ent))
                    for j in range(max_len+1-len(token)):
                        if sample_sentences[i][j: j + len(token)] == token:
                            idxs.append((j, j + len(token)))
                    for start, end in idxs:
                        ents[id][start:end] = 1
                entities.append(torch.from_numpy(ents))
                ctxt = np.zeros((len(ents), len(ents), max_len))
                lb = np.zeros((len(ents), len(ents), label_num))

                for id in range(len(ents)):
                    for j in range(len(ents)):
                        if j == id:
                            continue
                        first = min(np.where(ents[id] == 1)[0][0], np.where(ents[j] == 1)[0][0])
                        last = max(np.where(ents[id] == 1)[0][-1], np.where(ents[j] == 1)[0][-1])
                        # if i == 0 and id == 0:
                        #     print("i: {} j: {} first: {} last: {}".format(id, j, first, last))
                        for k in range(first, last):
                            if ents[id][k] == ents[j][k]:
                                ctxt[id][j][k] = 1
                        lb[id][j] = self.find_label(c, idx, e_s[id], e_s[j], labels)
                # if i==0:
                #     print("ctxt======================")
                #     print(ctxt)
                #     print("test====================")
                #     print(np.where(ctxt[0][1]==1)[0])
                context.append(torch.from_numpy(ctxt))
                label.append(torch.from_numpy(lb))

        # print(sentences)
        # print(mask)

        print(entities[0])
        print('-----------------------------------------------------------------------------------------------')
        print(context[0])
        print('------------------------------------------------------------------------------------------------')
        print(labels)
        print(label[0])
        support_sentence=[]
        query_sentence=[]
        support_mask=[]
        query_mask=[]
        support_entities=[]
        query_entities=[]
        support_context=[]
        query_context=[]
        support_label=[]
        query_label=[]
        for n in range(self.N):
            support_sentence+=sample_sentences[n*shots:shots*n+self.support_shot]
            query_sentence+=sample_sentences[shots*n+self.support_shot:shots*n+shots]
            support_mask+=mask[n*shots:shots*n+self.support_shot]
            query_mask += mask[shots * n + self.support_shot:shots * n + shots]
            support_entities+=entities[n*shots:shots*n+self.support_shot]
            query_entities+=entities[shots * n + self.support_shot:shots * n + shots]
            support_context+=context[n*shots:shots*n+self.support_shot]
            query_context+=context[shots * n + self.support_shot:shots * n + shots]
            support_label+=label[n*shots:shots*n+self.support_shot]
            query_label+=label[shots * n + self.support_shot:shots * n + shots]
        support_set = torch.tensor(support_sentence), torch.tensor(support_mask),support_entities, support_context, support_label
        query_set = torch.tensor(query_sentence), torch.tensor(query_mask), query_entities, query_context, query_label
        return support_set, query_set, labels

    def __len__(self):
        return self.batch_num


def collate_fn(data):
    support_set, query_set, labels = zip(*data)
    return support_set, query_set, labels


def get_data_loader(root, N, batch_num, support_size, query_size, mode):
    dataset = NYTDataset(root, N, batch_num, support_size, query_size, mode)
    data_loader = Data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    return data_loader


# root='data/dict.json'
# data_loader=get_data_loader(root, 2, 1, 1, 1,'train')
# encoder=BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
#
# for data in data_loader:
#     print("ooooooooooooooooooooooooooooooo")
#     spt, qry, label=data
#     support_set=spt[0]
#     query_set=qry[0]
#     labels=label[0]
#     output=encoder(support_set[0],support_set[1])
#     hidden_states=output[2][-1]
#
#     # masks of first sentence's entities
#     h_0=torch.randn(2, 1, 100)
#     c_0=torch.randn(2, 1, 100)
#     for entity_mask in support_set[2][0]:
#         entity=[]
#         for i, val in enumerate(entity_mask):
#             if val==0 and len(entity)!=0:
#                 break
#             else:
#                 if val==0:
#                     continue
#                 else:
#                     entity.append(list(hidden_states[0][i]))
#         seq_len=len(entity)
#         entity=torch.tensor(entity)
#         entity=torch.unsqueeze(entity,1)
#         print(entity.shape)
#         aggregator=nn.LSTM(768, 100, bidirectional=True)
#         # entity: seq_len*1*768
#         output, (hn, cn)=aggregator(entity, (h_0, c_0))
#         print(hn.shape)
#         print((hn[0]+hn[1])/2.0)
#
#
#     print(support_set[0].shape)
#     print(hidden_states.shape)
#     print(hidden_states[0][0][0:20])

