"""
The main model of FSMRE
Author: Tong
Time: 11-04-2021
Remark: Wei
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np


class FSMRE(nn.Module):
    """
    few-shot multi-relation extraction
    """

    def __init__(self, encoder, aggregator, propagator, hidden_dim=100, proto_dim=200, support_shot=1,
                 query_shot=1, max_length=100) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(FSMRE, self).__init__()
        # the number of support instances inside a task
        self.support_shot = support_shot
        # the number of query instances inside a task
        self.query_shot = query_shot
        # the max length of sentences
        self.max_length = max_length

        self.hidden_dim = hidden_dim
        self.proto_dim = proto_dim

        # default: Bert encoder
        self.encoder = encoder
        # default: BiLSTM, simplified: average
        self.aggregator = aggregator

        self.propagator = propagator

        h_0 = torch.randn(2, 1, hidden_dim)
        c_0 = torch.randn(2, 1, hidden_dim)
        self.h0=h_0
        self.c0=c_0

        # # attention_layer
        # self.rel_aware_att_layer = nn.Sequential(
        #     # 300x100
        #     nn.Linear(self.hidden_dim + self.proto_dim, self.hidden_dim),
        #     nn.Sigmoid()
        # )

    def forward(self, support_set, query_set, labels):
        """
        generate prototype embedding from support set, and conduct prediction for query_set
        Args:
            support_set (tuple):
                [0]: instances, torch.Tensor, sentence_num * max_length
                [1]: mask, torch.Tensor, sentence_num * max_length
                [2]: entities, [torch.Tensor], sentence_num * entity_num * entity_mask
                [3]: context, [torch.Tensor], sentence_num * entity_num * entity_num * context_mask
                [4]: label, [torch.Tensor], sentence_num * entity_num * entity_num * label_num
            query_set (tuple):
                [0]: instances, torch.Tensor, sentence_num * max_length
                [1]: mask, torch.Tensor, sentence_num * max_length
                [2]: entities, [torch.Tensor], sentence_num * entity_num * entity_mask
                [3]: context, [torch.Tensor], sentence_num * entity_num * entity_num * context_mask
                [4]: label, [torch.Tensor], sentence_num * entity_num * entity_num
        Returns:
            prediction: [torch.Tensor], sentence_num * entity_num * entity_num * class_size
            query_set[4]: [torch.Tensor], sentence_num * entity_num * entity_num
        """
        # get prototype embedding for each class
        # size: class_size * self.prototype_size

        prototype, context_center= self._process_support(support_set, labels)
        prediction = self._process_query(prototype, query_set)


        return prediction

    def _process_support(self, support_set, labels):
        """
        generate prototype embedding for each class
        Args:
            support_set (tuple):
                [0]: instances, torch.Tensor, sentence_num * max_length
                [1]: mask, torch.Tensor, sentence_num * max_length
                [2]: entities, [torch.Tensor], sentence_num * entity_num * entity_mask
                [3]: context, [torch.Tensor], sentence_num * entity_num * entity_num * context_mask
                [4]: label, [torch.Tensor], sentence_num * entity_num * entity_num
        Returns:
            support_set
        """



        '''Step 0 & 1: encoding and propagation'''
        # label_num*instance_num*proto_dim
        batch_entities, batch_context = self.encoding_aggregation(support_set, labels)

        # '''Step 2: general propagation ''

        # batch_entities, batch_context = self.propagator(batch_entities, batch_context)

        '''Step 3: obtain prototype embedding'''
        # todo: get prototype from batch_entities
        prototype = [[] for i in range(len(labels))]
        context_center=[[] for i in range(len(labels))]
        for i, val in enumerate(batch_entities):
            prototype[i]=torch.tensor(np.mean(val,axis=0))
        for i, val in enumerate(batch_context):
            context_center[i]=torch.tensor(np.mean(val, axis=0))
        return prototype, context_center

    def _process_query(self, prototype, query_set):
        """
        generate predictions for query instances
        Args:
            prototype (torch.Tensor):
            query_set (tuple): refer to support_set
        Returns:
            predictions
        """
        '''Step 0 & 1: encoding and propagation'''
        batch_entities, batch_context = self._encode_aggregation(query_set)

        '''Step 2: general propagation '''
        batch_entities, batch_context = self.propagator(batch_entities, batch_context)

        '''Step 3: relation-aware propagation'''
        rel_att = self._relation_aware_attention(prototype, batch_context)
        # weight the context
        batch_context = rel_att * batch_context
        batch_entities, batch_context = self.propagator(batch_entities, batch_context)

        '''Step 4: prototype-based classification'''
        # todo: get prototype from batch_entities
        prediction = None

        return prediction

    def encoding_aggregation(self, input_set, labels):
        """
        general processing of support_set or query_set
        Args:
            input_set (tuple): support_set or query_set
        Returns:
        batch_entities, batch_contexts
        :param input_set:
        :param labels:
        """
        # output of encoder:
        # - 0. the last hidden state (batch_size, sequence_length, hidden_size)
        # - 1. the pooler_output of the classification token (batch_size, hidden_size)
        # - 2. the hidden_states of the outputs of the model at each layer and the initial embedding outputs
        #    (batch_size, sequence_length, hidden_size)

        '''Step 0: encoding '''
        # [-1] for the last layer representation -> size: sentence_num * max_length * h_dim(768)
        # get the encodings of the tokens in the sentences
        # sentence_num*max_len*768
        encodings = self.encoder(input_set[0], input_set[1])[2][-1]

        '''Step 1 - 1: entity aggregation'''
        # sequencial_processing: process entity
        label_num=len(labels)
        # [ [] entity_pair_num ,[],[]  ] label_num
        batch_entities = [[] for i in range(label_num)]
        batch_context=batch_entities
        # input_set[4]:sentence_num*entity_num*entity_num*label_num

        entity_embedding_dic={}
        context_embedding_dic={}

        for sentence_id, sentence_label in enumerate(input_set[4]):
            for entity_id, entity_1 in enumerate(sentence_label):
                for entity_id_2, pair_label in enumerate(entity_1):
                    if entity_id==entity_id_2:
                        continue
                    for label_id in range(label_num):
                        if pair_label[label_id]==1:
                            if (sentence_id, entity_id) not in entity_embedding_dic:
                                entity=[]
                                # input_set[2]: sentence_num*entity_num*max_len
                                for j, val in enumerate(input_set[2][sentence_id][entity_id]):
                                    if val == 0 and len(entity) != 0:
                                        break
                                    else:
                                        if val == 0:
                                            continue
                                        else:
                                            entity.append(list(encodings[sentence_id][j]))
                                entity = torch.tensor(entity)
                                entity = torch.unsqueeze(entity, 1)
                                output, (hn, cn)=self.aggregator(entity, (self.h0, self.c0))
                                embedding=(hn[0]+hn[1])/2.0
                                entity_embedding_dic[(sentence_id, entity_id)]=embedding
                            if (sentence_id, entity_id_2) not in entity_embedding_dic:
                                entity = []
                                # input_set[2]: sentence_num*entity_num*max_len
                                for j, val in enumerate(input_set[2][sentence_id][entity_id_2]):
                                    if val == 0 and len(entity) != 0:
                                        break
                                    else:
                                        if val == 0:
                                            continue
                                        else:
                                            entity.append(list(encodings[sentence_id][j]))
                                entity = torch.tensor(entity)
                                entity = torch.unsqueeze(entity, 1)
                                output, (hn, cn) = self.aggregator(entity, (self.h0, self.c0))
                                embedding = (hn[0] + hn[1]) / 2.0
                                entity_embedding_dic[(sentence_id, entity_id_2)] = embedding
                            if (sentence_id, entity_id, entity_id_2) not in context_embedding_dic:
                                context=[]
                                for j, val in enumerate(input_set[3][sentence_id][entity_id][entity_id_2]):
                                    if val == 0 and len(entity) != 0:
                                        break
                                    else:
                                        if val == 0:
                                            continue
                                        else:
                                            context.append(list(encodings[sentence_id][j]))
                                context=torch.tensor(context)
                                context=torch.unsqueeze(context, 1)
                                output, (hn, cn) = self.aggregator(entity, (self.h0, self.c0))
                                embedding = (hn[0] + hn[1]) / 2.0
                                context_embedding_dic[(sentence_id, entity_id, entity_id_2)] = embedding
                                context_embedding_dic[(sentence_id, entity_id_2, entity_id)] = embedding
                            batch_entities[label_id].append(torch.cat(entity_embedding_dic[(sentence_id, entity_id)],
                                                                      entity_embedding_dic[(sentence_id, entity_id_2)]))
                            batch_context[label_id].append(context_embedding_dic[(sentence_id, entity_id, entity_id_2)])
        return batch_entities, batch_context





    def _aggregate_entity(self, sentence_encodings, entity_mask):
        """
        generate entity encoding from sentence encodings.
        Args:
            sentence_encodings (torch.Tensor): sentence encodings
            entity_mask (torch.Tensor): context_mask
        Returns:
            node-weight (entity embedding) for relation graph
        """
        return self.aggregator(sentence_encodings, entity_mask)

    def _aggregate_context(self, sentence_encodings, context_mask):
        """
        generate pair-wise context encodings from sentence encodings.
        Args:
            sentence_encodings (torch.Tensor): sentence encodings
            context_mask (torch.Tensor): context_mask
        Returns:
            edge-weight (context embedding) for relation graph
        """
        return self.aggregator(sentence_encodings, context_mask)

    def _relation_aware_attention(self, prototype, weight):
        """
        calculate attention weight for relation-aware propagation
        Args:
            prototype (torch.Tensor): prototype embedding
            weight (torch.Tensor): edge-weight (context embedding) for relation graph
        Returns:
            attention weight
        """
        # todo: expand prototype
        return self.rel_aware_att_layer()