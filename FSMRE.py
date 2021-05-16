"""
The main model of FSMRE
Author: Tong
Time: 11-04-2021
Remark: Wei
"""

import torch
import torch.nn as nn
import numpy as np
from math import exp


def euclid_distance(x, y):
    return torch.pow(x - y, 2).sum().float().item()


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
        self.h0 = h_0
        self.c0 = c_0

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
            :param labels:
        """
        # get prototype embedding for each class
        # size: class_size * self.prototype_size

        prototype, context_center, instances_count = self._process_support(support_set, labels)
        prediction = self._process_query(prototype, context_center, query_set, labels, instances_count)
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
        instances_count = []
        for label_instances in batch_entities:
            instances_count.append(len(label_instances))
        # '''Step 2: general propagation ''

        # batch_entities, batch_context = self.propagator(batch_entities, batch_context)

        '''Step 3: obtain prototype embedding'''
        # todo: get prototype from batch_entities
        prototype = []
        context_center = []
        for i, val in enumerate(batch_entities):
            mean_val=torch.mean(val, dim=0)
            prototype.append(torch.empty_like(mean_val).copy_(mean_val))
        for i, val in enumerate(batch_context):
            mean_val = torch.mean(val, dim=0)
            context_center.append(torch.empty_like(mean_val).copy_(mean_val))
        return prototype, context_center, instances_count

    def _process_query(self, prototype, context_center, query_set, labels, instances_count):
        """
        generate predictions for query instances
        Args:
            prototype (torch.Tensor):
            query_set (tuple): refer to support_set
        Returns:
            predictions
        """
        instance_num = 0
        for instance_c in instances_count:
            instance_num += instance_c
        '''Step 0 & 1: encoding and propagation'''
        entitity_dic, context_dic = self.encoding_aggregation_query(query_set, labels)

        '''Step 2: general propagation '''

        prediction = [
            [[[[0 for n in range(len(labels))] for m in range(len(labels))] for k in range(len(query_set[4][i]))] for j
             in range(len(query_set[4][i]))] for i in range(len(query_set[4]))]
        for i in range(len(query_set[0])):
            x = []
            entity_num = len(query_set[2][i])
            edge_index = [[], []]
            for j in range(entity_num):
                x.append(entitity_dic[(i, j)])
                for delta in range(1, entity_num - j):
                    edge_index[0].append(j)
                    edge_index[0].append(j + delta)
                    edge_index[1].append(j + delta)
                    edge_index[1].append(j)
            x = torch.cat(x, 0)
            edge_index = torch.tensor(edge_index)

            # each relation
            for k in range(len(labels)):
                edge_weight = []
                for j in range(0, len(edge_index[0]), 2):
                    edge_weight.append(1.0 / euclid_distance(
                        context_dic[(i, j, edge_index[1][j].item())],
                        context_center[k]
                    ))
                edge_weight = torch.tensor(edge_weight)
                out = self.propagator(x, edge_index, edge_weight)
                for j in range(len(edge_index[0])):
                    pred_embedding = torch.cat((out[j], out[edge_index[1][j].item()]),dim=0)
                    for n in range(len(labels)):
                        sum = 0.0
                        for l in range(len(labels)):
                            if l == n:
                                continue
                            # FixMe limit function
                            sum += exp(
                                euclid_distance(pred_embedding, prototype[l]) + float(instances_count[l]) / float(
                                    instance_num - instances_count[l]))
                        numerator = exp(
                            euclid_distance(pred_embedding, prototype[n]) + float(instances_count[n]) / float(
                                instance_num - instances_count[n]))
                        prediction[i][j][edge_index[1][j]][k][n] = numerator / (numerator + sum)
        for i in range(len(prediction)):
            prediction[i] = torch.tensor(prediction[i])

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
        label_num = len(labels)
        # [ [] entity_pair_num ,[],[]  ] label_num
        batch_entities = [[] for i in range(label_num)]
        batch_context = [[] for i in range(label_num)]
        # input_set[4]:sentence_num*entity_num*entity_num*label_num

        entity_embedding_dic = {}
        context_embedding_dic = {}

        for sentence_id, sentence_label in enumerate(input_set[4]):
            for entity_id, entity_1 in enumerate(sentence_label):
                for entity_id_2, pair_label in enumerate(entity_1):
                    if entity_id == entity_id_2:
                        continue
                    for label_id in range(label_num):
                        if pair_label[label_id] == 1:
                            if (sentence_id, entity_id) not in entity_embedding_dic:
                                entity = []
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
                                output, (hn, cn) = self.aggregator(entity, (self.h0, self.c0))
                                embedding = (hn[0] + hn[1]) / 2.0
                                entity_embedding_dic[(sentence_id, entity_id)] = embedding
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
                                context = []
                                for j, val in enumerate(input_set[3][sentence_id][entity_id][entity_id_2]):
                                    if val == 0 and len(context) != 0:
                                        break
                                    else:
                                        if val == 0:
                                            continue
                                        else:
                                            context.append(list(encodings[sentence_id][j]))
                                if len(context) == 0:
                                    context.append(list(torch.zeros(self.hidden_dim)))
                                context = torch.tensor(context)
                                context = torch.unsqueeze(context, 1)
                                output, (hn, cn) = self.aggregator(context, (self.h0, self.c0))
                                embedding = (hn[0] + hn[1]) / 2.0
                                context_embedding_dic[(sentence_id, entity_id, entity_id_2)] = embedding
                                context_embedding_dic[(sentence_id, entity_id_2, entity_id)] = embedding
                            batch_entities[label_id].append(
                                torch.cat((entity_embedding_dic[(sentence_id, entity_id)],
                                                entity_embedding_dic[(sentence_id, entity_id_2)]), 1))
                            batch_context[label_id].append(
                                context_embedding_dic[(sentence_id, entity_id, entity_id_2)])

        for i in range(label_num):
            batch_entities[i]=torch.cat(batch_entities[i],0)
            batch_context[i]=torch.cat(batch_context[i],0)
        return batch_entities, batch_context

    def encoding_aggregation_query(self, input_set, labels):
        encodings = self.encoder(input_set[0], input_set[1])[2][-1]
        label_num = len(labels)
        entity_embedding_dic = {}
        context_embedding_dic = {}
        for sentence_id, sentence_label in enumerate(input_set[4]):
            for entity_id, entity_1 in enumerate(sentence_label):
                for entity_id_2, pair_label in enumerate(entity_1):
                    if entity_id == entity_id_2:
                        continue
                    for label_id in range(label_num):
                        if (sentence_id, entity_id) not in entity_embedding_dic:
                            entity = []
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
                            output, (hn, cn) = self.aggregator(entity, (self.h0, self.c0))
                            embedding = (hn[0] + hn[1]) / 2.0
                            entity_embedding_dic[(sentence_id, entity_id)] = embedding
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
                            context = []
                            for j, val in enumerate(input_set[3][sentence_id][entity_id][entity_id_2]):
                                if val == 0 and len(context) != 0:
                                    break
                                else:
                                    if val == 0:
                                        continue
                                    else:
                                        context.append(list(encodings[sentence_id][j]))
                            if len(context) == 0:
                                context.append(list(torch.zeros(self.hidden_dim)))
                            context = torch.tensor(context)
                            context = torch.unsqueeze(context, 1)
                            output, (hn, cn) = self.aggregator(context, (self.h0, self.c0))
                            embedding = (hn[0] + hn[1]) / 2.0
                            context_embedding_dic[(sentence_id, entity_id, entity_id_2)] = embedding
                            context_embedding_dic[(sentence_id, entity_id_2, entity_id)] = embedding
        return entity_embedding_dic, context_embedding_dic