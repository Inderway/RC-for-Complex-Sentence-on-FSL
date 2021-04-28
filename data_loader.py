"""
data_loader
Author: Wei
Date: 2021/4/24
"""
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from transformers import BertTokenizer

class NYTDataset(data.Dataset):
    def __init__(self, data_dir, name, N):
        self.data_dir=data_dir
        self.N=N
        path=os.path.join(data_dir, name+'.json')
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.Data=json.load(open(path))
        self.classes=list(self.Data.keys())

        for i in range(24):
            print(self.classes[i])
            print(len(self.Data[self.classes[i]]))

        tokenzier = BertTokenizer.from_pretrained('bert-base-cased')

        target_classes = random.sample(self.classes, self.N)
        sentences = []
        sentences_of_first_class=[]
        idx_of_instances_in_first_class=[]
        sentences_of_second_class=[]
        idx_of_instances_in_second_class=[]
        for sent in self.Data[target_classes[0]]:
            sentences_of_first_class.append(sent['sentText'])
        for sent in self.Data[target_classes[1]]:
            sentences_of_second_class.append(sent['sentText'])
        idx_val_1=random.sample(list(enumerate(sentences_of_first_class)),12)
        idx_val_2 = random.sample(list(enumerate(sentences_of_second_class)), 13)
        for idx, val in idx_val_1:
            idx_of_instances_in_first_class.append(idx)
            sentences.append(val)
        for idx, val in idx_val_2:
            idx_of_instances_in_second_class.append(idx)
            sentences.append(val)



        # for c in target_classes:
        #     sents = []
        #     print(c)
        #     for sent in self.Data[c]:
        #         sents.append(sent['sentText'])
        #     if len(sentences)<12:
        #         sentences.extend(random.sample(sents,12))
        #     else:
        #         sentences.extend(random.sample(sents,13))
        max_len=0
        for idx in range(len(sentences)):
            sentences[idx]=tokenzier.convert_tokens_to_ids(tokenzier.tokenize(sentences[idx]))
            max_len=max(max_len,len(sentences[idx]))
        print(max_len)
        print(sentences)
        mask=[]
        entities=[]
        context=[]
        for i in range(len(sentences)):
            tmp_mask=len(sentences[i])*[1]
            if len(sentences[i])<max_len:
                padding=(max_len-len(sentences[i]))*[0]
                sentences[i].extend(padding)
                tmp_mask.extend(padding)
            mask.append(tmp_mask)

            c = target_classes[0] if i < 12 else target_classes[1]
            idx = idx_of_instances_in_first_class[i] if i < 12 else idx_of_instances_in_second_class[i-12]
            e_s = []
            for ent in self.Data[c][idx]['entityMentions']:
                e_s.append(ent['text'])
            ents=torch.zeros([len(e_s),max_len])
            for id, ent in enumerate(e_s):
                idxs=[]
                token=tokenzier.convert_tokens_to_ids(tokenzier.tokenize(ent))
                for j in range(len(sentences[i])):
                    if sentences[i][j: j+len(token)]==token:
                        idxs.append((j, j+len(token)))
                for start, end in idxs:
                    ents[id][start:end]=1
            entities.append(ents)
            ctxt=torch.zeros(len(ents), len(ents), max_len)
            for id, ent in enumerate(ents):
                for j in range(len(ents)):
                    if j==id:
                        continue






        #print(sentences)
        #print(mask)
        #print('--------------------------')
        #print(entities)





        # sentence=self.json_data[self.classes[0]][0]['sentText']
        # print(sentence)
        # tokenzier = BertTokenizer.from_pretrained('bert-base-cased')
        # tok=tokenzier.tokenize(sentence)
        # input_ids = tokenzier.convert_tokens_to_ids(tok)
        # print(tok)
        # print(input_ids)


    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        sentences=[]
        for c in target_classes:
            for sentence in self.Data[c]:
                sentences.append[sentence['sentText']]


    def __len__(self):
        return len(self.Data)

data_dir='data'
name='dict'
NYTDataset(data_dir,name, 2)




