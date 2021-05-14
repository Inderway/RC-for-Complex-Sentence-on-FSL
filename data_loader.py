"""
data_loader
Author: Wei
Date: 2021/4/24
"""
import torch
import torch.utils.data as Data
import os
import numpy as np
import random
import json
from transformers import BertTokenizer


class NYTDataset(Data.Dataset):
    def __init__(self, root, N, batch_num, support_shot, query_shot, mode):
        self.root=root
        self.N = N
        path = root
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert 0
        if support_shot<=0 or query_shot<=0:
            print("[ERROR] support shots and query shots must be larger than 0!")
            assert 0
        self.Data = json.load(open(path))
        self.classes = list(self.Data.keys())
        if mode=='train':
            self.classes = self.classes[:17]
        else:
            self.classes=self.classes[17:22]
        self.batch_num=batch_num


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
        sentences = []
        sentences_of_first_class = []
        idx_of_instances_in_first_class = []
        sentences_of_second_class = []
        idx_of_instances_in_second_class = []
        labels = []

        for sent in self.Data[target_classes[0]]:
            sentences_of_first_class.append(sent['sentText'])
        for sent in self.Data[target_classes[1]]:
            sentences_of_second_class.append(sent['sentText'])
        idx_val_1 = random.sample(list(enumerate(sentences_of_first_class)), 17)
        idx_val_2 = random.sample(list(enumerate(sentences_of_second_class)), 18)
        # idx_val_1 = idx_sample_1[:12]
        # idx_val_2 = idx_sample_2[:13]
        # idx_val_query_1 = idx_sample_1[12:]
        # idx_val_query_2 = idx_sample_2[13:]

        for idx, val in idx_val_1:
            idx_of_instances_in_first_class.append(idx)
            sentences.append(val)
        for idx, val in idx_val_2:
            idx_of_instances_in_second_class.append(idx)
            sentences.append(val)

        print("class: {}, sentenceID: {}".format(target_classes[0],idx_of_instances_in_first_class[0]))
        print(sentences[0])
        # for c in target_classes:
        #     sents = []
        #     print(c)
        #     for sent in self.Data[c]:
        #         sents.append(sent['sentText'])
        #     if len(sentences)<12:
        #         sentences.extend(random.sample(sents,12))
        #     else:
        #         sentences.extend(random.sample(sents,13))
        max_len = 0

        # get labels from support set and get max_len
        for i in range(len(sentences)):
            sentences[i] = tokenzier.convert_tokens_to_ids(tokenzier.tokenize(sentences[i]))
            max_len = max(max_len, len(sentences[i]))
            if (11 < i < 17) or (i > 29):
                continue
            c = target_classes[0] if i < 17 else target_classes[1]
            idx = idx_of_instances_in_first_class[i] if i < 17 else idx_of_instances_in_second_class[i - 17]
            for rel in self.Data[c][idx]['relationMentions']:
                if rel['label'] not in labels:
                    labels.append(rel['label'])
        label_num = len(labels)
        self.labels=labels
        print(max_len)
        # print(sentences)
        mask = []
        entities = []
        context = []
        label = []
        for i in range(len(sentences)):
            tmp_mask = len(sentences[i]) * [1]
            if len(sentences[i]) < max_len:
                padding = (max_len - len(sentences[i])) * [0]
                sentences[i].extend(padding)
                tmp_mask.extend(padding)
            mask.append(tmp_mask)

            c = target_classes[0] if i < 17 else target_classes[1]
            idx = idx_of_instances_in_first_class[i] if i < 17 else idx_of_instances_in_second_class[i - 17]
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
                for j in range(len(sentences[i])):
                    if sentences[i][j: j + len(token)] == token:
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

        # print(entities[0])
        # print('-----------------------------------------------------------------------------------------------')
        # print(context[0])
        # print('------------------------------------------------------------------------------------------------')
        # print(labels)
        # print(label[0])

        support_set = torch.tensor(sentences[:12] + sentences[17:30]),torch.tensor(mask[:12] + mask[17:30]),entities[:12] + entities[17:30],context[:12] + context[17:30],label[:12] + label[17:30]
        query_set = torch.tensor(sentences[12:17] + sentences[30:]),torch.tensor(mask[12:17] + mask[30:]),entities[12:17] + entities[30:],context[12:17] + context[30:],label[12:17] + label[30:]
        return support_set, query_set, labels

    def __len__(self):
        return self.batch_num

def collate_fn(data):
    support_set, query_set, labels=zip(*data)
    return support_set, query_set, labels

def get_data_loader(root, N, batch_num, support_size, query_size):
    dataset=NYTDataset(root, N,batch_num, support_size,query_size)
    data_loader=Data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    return data_loader

# root='data/dict.json'
# data_loader=get_data_loader(root, 2, 1, 25, 10)
#
# for data in data_loader:
#     print("ooooooooooooooooooooooooooooooo")
#     spt, qry, label=data
#     print(label[0])
