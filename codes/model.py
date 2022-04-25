#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.distributions import Uniform
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from dataloader import *
import random
import pickle
import math
try:
    from apex import amp
except:
    print("apex not installed")
import os
from collections import defaultdict
import sklearn
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(
    name_query_dict.keys())  # ['1p', '2p',


# '3p', '2i', '3i',
# 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF',
# '2u-DM', 'up-DNF', 'up-DM']

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, particles):

        # [batch_size, num_particles, embedding_size]
        K = self.query(particles)
        V = self.query(particles)
        Q = self.query(particles)

        # [batch_size, num_particles, num_particles]
        attention_scores = torch.matmul(Q, K.permute(0,2,1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # [batch_size, num_particles, num_particles]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        # [batch_size, num_particles, embedding_size]
        attention_output = torch.matmul(attention_probs, V)

        return attention_output

class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """
    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)


        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))


class ParticleCrusher(nn.Module):

    def __init__(self, embedding_size, num_particles):
        super(ParticleCrusher, self).__init__()

        # self.noise_layer = nn.Linear(embedding_size, embedding_size)
        self.num_particles = num_particles

        self.off_sets = nn.Parameter(torch.zeros([1, num_particles, embedding_size]), requires_grad=True)
        # self.layer_norm = LayerNorm(embedding_size)



    def forward(self, batch_of_embeddings):
        # shape of batch_of_embeddings: [batch_size, embedding_size]
        # the return is a tuple ([batch_size, embedding_size, num_particles], [batch_size, num_particles])
        # The first return is the batch of particles for each entity, the second is the weights of the particles
        # Use gaussian kernel to do this

        batch_size, embedding_size = batch_of_embeddings.shape

        # [batch_size, num_particles, embedding_size]
        expanded_batch_of_embeddings = batch_of_embeddings.reshape(batch_size, -1, embedding_size) + self.off_sets

        return expanded_batch_of_embeddings


class Projection(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(Projection, self).__init__()

        self.layer_norm_1 = LayerNorm(embedding_size)
        self.layer_norm_2 = LayerNorm(embedding_size)
        # self.layer_norm_3 = LayerNorm(embedding_size)
        # self.layer_norm_4 = LayerNorm(embedding_size)

        self.embedding_size = embedding_size

        self.dropout = dropout

        self.self_attn = SelfAttention(embedding_size)

        # self.ffn = FFN(embedding_size, dropout)
        # self.ffn2 = FFN(embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.Wz = nn.Linear(embedding_size, embedding_size)
        self.Uz = nn.Linear(embedding_size, embedding_size)

        self.Wr = nn.Linear(embedding_size, embedding_size)
        self.Ur = nn.Linear(embedding_size, embedding_size)

        self.Wh = nn.Linear(embedding_size, embedding_size)
        self.Uh = nn.Linear(embedding_size, embedding_size)


    def forward(self, particles, relation_embedding):
        """

        :param particles: [batch_size, num_particles, embedding_size]
        :param relation_embedding: [batch_size, embedding_size * 2]

        :return: [batch_size, num_particles, embedding_size]
        """

        #  [batch_size, 1, embedding_size]
        relation_transition = torch.unsqueeze(relation_embedding, 1)

        #  [batch_size, num_particles, embedding_size]
        projected_particles = particles

        z = self.sigmoid(self.Wz(self.dropout(relation_transition)) + self.Uz(self.dropout(projected_particles)))
        r = self.sigmoid(self.Wr(self.dropout(relation_transition)) + self.Ur(self.dropout(projected_particles)))

        h_hat = self.tanh(self.Wh(self.dropout(relation_transition)) + self.Uh(self.dropout(projected_particles * r)))

        h = (1 - z) * particles + z * h_hat

        projected_particles = h
        projected_particles = self.layer_norm_1(projected_particles)

        projected_particles = self.self_attn(self.dropout(projected_particles))
        projected_particles = self.layer_norm_2(projected_particles)

        # projected_particles = self.ffn(projected_particles)  + projected_particles
        # projected_particles = self.layer_norm_3(projected_particles)

        return projected_particles

class HigherOrderProjection(nn.Module):
    def __init__(self, hidden_size, dropout, op_projection):
        super(HigherOrderProjection, self).__init__()

        self.attn = SelfAttention(hidden_size)

        self.op_projection = op_projection
        self.dropout = dropout

        self.ffn = FFN(hidden_size, dropout)

        self.layer_norm_ffn = LayerNorm(hidden_size)
        self.layer_norm_attn = LayerNorm(hidden_size)

    def forward(self, particles, relation_embedding):

        particles = self.attn(particles)
        particles = self.layer_norm_attn(particles)

        particles = self.ffn(particles) + particles
        particles = self.layer_norm_ffn(particles)

        particles = self.op_projection(particles, relation_embedding)

        return particles



class Complement(nn.Module):
    def __init__(self, embedding_size, num_particles, dropout):
        super(Complement, self).__init__()

        self.complement_layer = SelfAttention(embedding_size)

        self.dropout = dropout

        self.num_particles = num_particles

        self.ffn = FFN(embedding_size, dropout)

        self.layer_norm_1 = LayerNorm(embedding_size)
        self.layer_norm_2 = LayerNorm(embedding_size)

        self.layer_norm_3 = LayerNorm(embedding_size)
        self.layer_norm_4 = LayerNorm(embedding_size)


    def forward(self, particles):
        """
        :param particles: [batch_size, num_particles, embedding_size]

        :return: [batch_size, num_particles, embedding_size], [batch_size, num_particles]
        """

        batch_size, num_particles, embedding_size = particles.shape


        # [batch_size, num_particles, embedding_size]
        new_particles = particles

        new_particles = self.complement_layer(self.dropout(new_particles))
        new_particles = self.layer_norm_1(new_particles)
        new_particles = self.ffn(new_particles) + new_particles
        new_particles = self.layer_norm_2(new_particles)

        new_particles = self.complement_layer(self.dropout(new_particles))
        new_particles = self.layer_norm_3(new_particles)
        new_particles = self.ffn(new_particles) + new_particles
        new_particles = self.layer_norm_4(new_particles)
        
        return new_particles


class Intersection(nn.Module):

    def __init__(self, embedding_size, num_particles, dropout):
        super(Intersection, self).__init__()

        self.intersection_layer = SelfAttention(embedding_size)

        self.dropout = dropout

        self.num_particles = num_particles

        self.layer_norm_1 = LayerNorm(embedding_size)
        self.layer_norm_2 = LayerNorm(embedding_size)

        self.layer_norm_3 = LayerNorm(embedding_size)
        self.layer_norm_4 = LayerNorm(embedding_size)

        self.ffn = FFN(embedding_size, dropout)


    def forward(self, particles_sets):
        """
        :param particles_sets: [batch_size, num_sets, num_particles, embedding_size]
        :param weights_sets: [batch_size, num_sets, num_particles]
        :return: [batch_size, num_particles, embedding_size]
        """

        batch_size, num_sets, num_particles, embedding_size = particles_sets.shape


        # [batch_size, num_sets * num_particles, embedding_size]
        flatten_particles = particles_sets.view(batch_size, -1, embedding_size)

        # [batch_size, num_sets * num_particles, embedding_size]
        flatten_particles = self.intersection_layer(self.dropout(flatten_particles))
        flatten_particles = self.layer_norm_1(flatten_particles)

        flatten_particles = self.ffn(flatten_particles) + flatten_particles
        flatten_particles = self.layer_norm_2(flatten_particles)


        flatten_particles = self.intersection_layer(self.dropout(flatten_particles))
        flatten_particles = self.layer_norm_3(flatten_particles)

        flatten_particles = self.ffn(flatten_particles) + flatten_particles
        flatten_particles = self.layer_norm_4(flatten_particles)



        particles = flatten_particles[:, num_sets * torch.arange(self.num_particles)]

        return particles




class Query2Particles(nn.Module):
    def __init__(self, nentity, nrelation,
                 entity_space_dim,
                 num_particles= 2,
                 dropout_rate=0.2,
                 label_smoothing=0.1):
        super(Query2Particles, self).__init__()

        # entity_emb_layer, num_ent_embeddings, ent_embedding_dim = create_emb_layer(entity_weights, fixed_embeddings)
        # relation_emb_layer, num_rel_embeddings, rel_embedding_dim = create_emb_layer(relation_weights, fixed_embeddings)
        #
        # assert ent_embedding_dim == rel_embedding_dim
        # assert nentity == num_ent_embeddings
        # assert nrelation == num_rel_embeddings

        self.label_smoothing = label_smoothing

        hidden_dim = entity_space_dim
        set_hidden_dim = entity_space_dim

        self.nentity = nentity
        self.nrelation = nrelation
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        # Number of particles used in the data
        self.num_particles = num_particles


        self.dropout_rate = dropout_rate


        self.dropout = nn.Dropout(dropout_rate)

        # Initialize the entity embeddings and relation transition embeddings with uniform gamma
        self.entity_embedding = nn.Embedding(nentity, hidden_dim)
        self.relation_transition_embedding = nn.Embedding(nrelation, hidden_dim)

        # use the particles to do the assignments
        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(hidden_dim,
                                 nentity,
                                 bias=True)
        self.decoder.weight = embedding_weights


        # The model support the operation of projection, intersection, union, and complement
        self.op_to_particles = ParticleCrusher(hidden_dim, num_particles)
        self.op_projection = Projection(hidden_dim, self.dropout)
        self.op_intersection = Intersection(hidden_dim, num_particles, self.dropout)
        self.op_complement = Complement(hidden_dim, num_particles,  self.dropout)

        self.op_higher_projection = HigherOrderProjection(hidden_dim, self.dropout, self.op_projection)

    def forward(self, batch_queries, qtype, inverted=False):
        """
        :param positive_sample: [batch_size]
        :param negative_sample: [batch_size, num_negative_sample]
        :param batch_queries: [batch_size, query_length]
        :param qtype: str

        :return: positive_likelihoods, negative_likelihoods,  particles, weights
        positive_likelihoods: [batch_size, 1]
        negative_likelihoods: [batch_size, num_negative_sample]

        """

        if qtype == "1p":

            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # Forward propagation
            particles= self.op_to_particles(entity_1)

            particles= self.op_projection(particles, relation_transition_1)

        elif qtype == "2u-arg":
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            batch_size, embedding_size = relation_transition_1.shape

            permutation_index = torch.randperm(batch_size)

            # Forward propagation
            # [batch_size, num_particles, embedding_size]
            particles = self.op_to_particles(entity_1)

            permuted_particles = particles[permutation_index]
            permuted_relations = relation_transition_1[permutation_index]

            # [batch_size, num_particles, embedding_size]
            particles = self.op_projection(particles, relation_transition_1)
            permuted_particles = self.op_projection(permuted_particles, permuted_relations)

            # [batch_size, 2 * num_particles, embedding_size]
            particles = torch.cat([particles, permuted_particles], dim=1)



        elif qtype == "2p":
            # positive sample is of shape [batch_size, 4], [entity1_id, rel_id1, rel_id2, answer_id]
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 2])

            # Forward propagation
            particles = self.op_to_particles(entity_1)

            particles = self.op_projection(particles,
                                                    relation_transition_1)

            particles = self.op_higher_projection(particles,
                                                    relation_transition_2)

        elif qtype == "3p":

            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_3 = self.relation_transition_embedding(batch_queries[:, 3])

            # Forward propagation
            particles = self.op_to_particles(entity_1)

            particles = self.op_projection(particles, relation_transition_1)

            particles = self.op_higher_projection(particles, relation_transition_2)

            particles = self.op_higher_projection(particles, relation_transition_3)


        elif qtype == "2i":
            # [batch_size, embedding_size]
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 3])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1)


            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)


            # [batch_size, embedding_size, num_particles] ->
            # [ batch_size, num_sets, embedding_size, num_particles]
            sets_of_particles = torch.stack([particles_1, particles_2], dim=1)


            particles = self.op_intersection(sets_of_particles)


        elif qtype == "3i":
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            entity_3 = self.entity_embedding(batch_queries[:, 4])

            # [batch_size, embedding_size]
            relation_transition_3 = self.relation_transition_embedding(batch_queries[:, 5])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)

            particles_3 = self.op_to_particles(entity_3)

            particles_3 = self.op_projection(particles_3, relation_transition_3)


            sets_of_particles = torch.stack([particles_1, particles_2, particles_3], dim=1)

            particles = self.op_intersection(sets_of_particles)


        elif qtype == "2in":

            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 3])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)

            # complement operation here
            particles_2 = self.op_complement(particles_2)


            sets_of_particles = torch.stack([particles_1, particles_2], dim=1)

            particles = self.op_intersection(sets_of_particles)


        elif qtype == "3in":

            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            entity_3 = self.entity_embedding(batch_queries[:, 4])

            # [batch_size, embedding_size]
            relation_transition_3 = self.relation_transition_embedding(batch_queries[:, 5])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)

            particles_3 = self.op_to_particles(entity_3)

            particles_3 = self.op_projection(particles_3, relation_transition_3)

            # Only the third projection is taken complement
            particles_3 = self.op_complement(particles_3)

            sets_of_particles = torch.stack([particles_1, particles_2, particles_3], dim=1)

            particles = self.op_intersection(sets_of_particles)


        elif qtype == "inp":
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_1_2 = self.relation_transition_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 5])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1_1)


            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_1_2)

            # complement before the intersection
            particles_2 = self.op_complement(particles_2)

            sets_of_particles = torch.stack([particles_1, particles_2], dim=1)

            particles = self.op_intersection(sets_of_particles)

            particles = self.op_higher_projection(particles, relation_transition_2)


        elif qtype == "pin":
            # [batch_size, embedding_size]
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            relation_transition_1_2 = self.relation_transition_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 4])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1_1)

            particles_1 = self.op_higher_projection(particles_1, relation_transition_1_2)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)

            particles_2 = self.op_complement(particles_2)

            sets_of_particles = torch.stack([particles_1, particles_2], dim=1)

            particles = self.op_intersection(sets_of_particles)

        elif qtype == "pni":
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            relation_transition_1_2 = self.relation_transition_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 4])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 5])

            particles_1 = self.op_to_particles(entity_1)

            particles_1= self.op_projection(particles_1, relation_transition_1_1)

            particles_1 = self.op_higher_projection(particles_1, relation_transition_1_2)

            particles_1 = self.op_complement(particles_1)

            particles_2 = self.op_to_particles(entity_2)


            particles_2 = self.op_projection(particles_2, relation_transition_2)

            sets_of_particles = torch.stack([particles_1, particles_2], dim=1)

            particles = self.op_intersection(sets_of_particles)


        # the rest of data types are not trained before

        elif qtype == "ip":
            # [batch_size, embedding_size]
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_1_2 = self.relation_transition_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 4])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1_1)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_1_2)

            sets_of_particles = torch.stack([particles_1, particles_2], dim=1)

            particles = self.op_intersection(sets_of_particles)

            particles = self.op_higher_projection(particles, relation_transition_2)


        elif qtype == "pi":
            # [batch_size, embedding_size]
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            relation_transition_1_2 = self.relation_transition_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 4])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1_1)

            particles_1 = self.op_higher_projection(particles_1, relation_transition_1_2)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)

            sets_of_particles = torch.stack([particles_1, particles_2], dim=1)

            particles = self.op_intersection(sets_of_particles)


        elif qtype == "2u-DNF":

            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 3])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)

            particles = torch.cat([particles_1, particles_2], dim=1)

        elif qtype == "up-DNF":
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 2])

            # [batch_size, embedding_size]
            relation_transition_1_2 = self.relation_transition_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 5])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1_1)

            particles_1 = self.op_higher_projection(particles_1, relation_transition_2)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_1_2)

            particles_2 = self.op_higher_projection(particles_2, relation_transition_2)

            particles = torch.cat([particles_1, particles_2], dim=1)

        elif qtype == "2u-DM":
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 4])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_2)

            particles = torch.cat([particles_1, particles_2], dim=1)

        elif qtype == "up-DM":
            entity_1 = self.entity_embedding(batch_queries[:, 0])

            # [batch_size, embedding_size]
            relation_transition_1_1 = self.relation_transition_embedding(batch_queries[:, 1])

            # [batch_size, embedding_size]
            entity_2 = self.entity_embedding(batch_queries[:, 3])

            # [batch_size, embedding_size]
            relation_transition_1_2 = self.relation_transition_embedding(batch_queries[:, 4])

            # [batch_size, embedding_size]
            relation_transition_2 = self.relation_transition_embedding(batch_queries[:, 7])

            particles_1 = self.op_to_particles(entity_1)

            particles_1 = self.op_projection(particles_1, relation_transition_1_1)

            particles_1 = self.op_higher_projection(particles_1, relation_transition_2)

            particles_2 = self.op_to_particles(entity_2)

            particles_2 = self.op_projection(particles_2, relation_transition_1_2)

            particles_2 = self.op_higher_projection(particles_2, relation_transition_2)

            particles = torch.cat([particles_1, particles_2], dim=1)

        else:
            raise ValueError('query type %s not supported' % qtype)

        # we will use the cross entropy loss here. We will no longer use the positive or negative sampling
        # we will first try to fit the model as good as possible.

        """ 
        We will add the complement layer to the final particles if we have the inverted argument 
        set to be true. This operation is only valid during training the queries without negation
        [1p 2p 3p 2i 3i]
        """
        if inverted:
            assert (qtype in ["1p", "2p", "3p", "2i", "3i"])
            particles = self.op_complement(particles)



        # Here we use the argmax pooling to the particles to obtain the correct scores for each particle.

        # [batch_size, num_particles, num_entities]
        all_prediction_scores = self.decoder(particles)

        # [batch_size, num_particles, num_entities]
        # attention_score = nn.Softmax(dim=1)(all_prediction_scores / math.sqrt(self.entity_dim))

        # [batch_size, num_entities]
        prediction_scores, _ =  all_prediction_scores.max(dim=1)


        return prediction_scores


    @staticmethod
    def train_step(model, iter, optimizer, use_apex):
        model.train()
        optimizer.zero_grad()
        positive_sample, _, subsampling_weight, batch_queries, query_structures = next(iter)
        positive_sample = torch.tensor(positive_sample).cuda()
        batch_queries = torch.tensor(batch_queries).cuda()

        qtype = query_name_dict[query_structures[0]]

        _p = random.random()
        if qtype == "1p":
            if  _p < 0.3333:
                qtype = "2u-arg"

        try:
            label_smoothing = model.label_smoothing
        except:
            label_smoothing = model.module.label_smoothing

        prediction_scores = model(batch_queries, qtype)
        # print(prediction_scores.shape)
        # print(positive_sample.shape)

        loss_fct = LabelSmoothingLoss(smoothing=label_smoothing, reduction='none')
        masked_lm_loss = loss_fct(prediction_scores, positive_sample.view(-1))

        # print(masked_lm_loss.shape)
        loss = (masked_lm_loss).mean()

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                optimizer.step()
        else:
            loss.backward()
            optimizer.step()

        log = {
            'qtype': qtype,
            'loss': loss.item(),
        }

        return log

    @staticmethod
    def train_inv_step(model, iter, optimizer, use_apex):
        model.train()
        optimizer.zero_grad()
        positive_sample, _, subsampling_weight, batch_queries, query_structures = next(iter)

        positive_sample = torch.tensor(positive_sample).cuda()
        # positive_sample = positive_sample.cuda()
        # negative_sample = torch.tensor(negative_sample).cuda()
        batch_queries = torch.tensor(batch_queries).cuda()
        # subsampling_weight = torch.tensor(subsampling_weight).cuda()

        qtype = query_name_dict[query_structures[0]]

        prediction_scores = - model(batch_queries, qtype, inverted=True)
        try:
            label_smoothing = model.label_smoothing
        except:
            label_smoothing = model.module.label_smoothing

        loss_fct = LabelSmoothingLoss(smoothing=label_smoothing, reduction='none')
        masked_lm_loss = loss_fct(prediction_scores, positive_sample.view(-1))

        loss = (masked_lm_loss).mean()

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                optimizer.step()
        else:
            loss.backward()
            optimizer.step()

        log = {
            'qtype': qtype + "_inv",
            'loss': loss.item(),
        }

        return log

    @staticmethod
    def evaluate_step(model, negative_sample, queries, queries_unflatten, query_structures, easy_answers, hard_answers):
        model.eval()

        # here are only do the batch_size 1 case for evaluation


        queries = torch.tensor(queries).cuda()

        # We force all query structure in a single batch is the same
        qtype = query_name_dict[query_structures[0]]

        # print(positive_sample_placeholder.shape)
        # print(negative_sample.shape)

        # These are the two sets of answers
        easy_answer = easy_answers[queries_unflatten[0]]
        hard_answer = hard_answers[queries_unflatten[0]]

        all_answer = easy_answer.union(hard_answer)

        if len(list(hard_answer)) == 0:
            logs = {
                "mr": 30000,
                "mrr": 0,
                "hit_at_1": 0,
                "hit_at_2": 0,
                "hit_at_3": 0,
                "hit_at_5": 0,
                "hit_at_10": 0,
                "num_samples": 0
            }
            return logs


        hard_answer_ids = torch.tensor(list(hard_answer))
        easy_answer_ids = torch.tensor(list(easy_answer))
        all_answer_ids = torch.tensor(list(all_answer))


        prediction_scores = model(queries, qtype)

        # [nentities]
        prediction_scores = prediction_scores.squeeze()
        original_scores = prediction_scores.clone()

        # [nentities]
        not_answer_scores = prediction_scores
        not_answer_scores[all_answer_ids] = - 10000000

        # [1, nentities]
        not_answer_scores = not_answer_scores.unsqueeze(0)
        # print(not_answer_distance)

        # [num_hard_answers]
        hard_answer_scores = original_scores[hard_answer_ids]

        # [num_hard_answers, 1]
        hard_answer_scores = hard_answer_scores.unsqueeze(-1)


        answer_is_smaller_matrix = ((hard_answer_scores - not_answer_scores) < 0)

        hard_answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

        rankings = hard_answer_rankings.float()

        mr = torch.mean(rankings).cpu().numpy()
        mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy()
        hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy()
        hit_at_2 = torch.mean((rankings < 2.5).double()).cpu().numpy()
        hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy()
        hit_at_5 = torch.mean((rankings < 5.5).double()).cpu().numpy()
        hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy()

        # The best scores are implemented and compared with EmQL paper:
        # https://github.com/google-research/language/blob/master/language/emql/eval.py line 132
        # Which is the traditional definition of Hit@N
        mr_best = torch.min(rankings).cpu().numpy()
        mrr_best = torch.max(torch.reciprocal(rankings)).cpu().numpy()
        hit_at_1_best = torch.max((rankings < 1.5).double()).cpu().numpy()
        hit_at_2_best = torch.max((rankings < 2.5).double()).cpu().numpy()
        hit_at_3_best = torch.max((rankings < 3.5).double()).cpu().numpy()
        hit_at_5_best = torch.max((rankings < 5.5).double()).cpu().numpy()
        hit_at_10_best = torch.max((rankings < 10.5).double()).cpu().numpy()

        num_samples = len(all_answer)

        logs_new = {
            "mr": mr,
            "mrr": mrr,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_10": hit_at_10,
            "num_samples": 1.0,
            "new_num_samples": num_samples,
        }

        logs_tradition = {
            "mr": mr_best,
            "mrr": mrr_best,
            "hit_at_1": hit_at_1_best,
            "hit_at_3": hit_at_3_best,
            "hit_at_10": hit_at_10_best,
            "num_samples": 1.0,
            "new_num_samples": num_samples,
        }

        return logs_new, logs_tradition


def test_dataloading_new(data_path="../data/FB15k-237-betae"):
    print("start open")
    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    print("open finished")
    negative_sample_size = 1
    batch_size = 512
    cpu_num = 1
    test_batch_size = 1
    embedding_size = 400
    num_p = 2
    # print("Load pre_trained embeddings")
    # ent_embedding_path = "../data/transe_batae_fb237_ent.npy"
    # rel_embedding_path = "../data/transe_batae_fb237_rel.npy"

    # with open(ent_embedding_path, 'rb') as f:
    # ent_embedding = np.load(f)
    #
    # with open(rel_embedding_path, 'rb') as f:
    # rel_embedding = np.load(f)

    # ent_embedding = np.load(ent_embedding_path, 'r')
    # rel_embedding = np.load(rel_embedding_path, 'r')

    print("Cuda start")
    model = Query2Particles(nentity, nrelation, embedding_size, num_particles=num_p)

    print("create model")
    model.cuda()

    # if torch.cuda.device_count() > 1:
    #     # print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    no_decay = ['bias', 'layer_norm', 'embedding', "off_sets"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.Adam(
        optimizer_grouped_parameters,
        lr=0.0001,
    )
    print("to Cuda finished")

    use_apex = False

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    def load_data():
        '''
        Load all queries
        '''
        print("loading data")
        train_queries = pickle.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
        train_answers = pickle.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
        valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
        test_queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))

        return train_queries, train_answers, valid_queries, \
               valid_hard_answers, valid_easy_answers, test_queries, \
               test_hard_answers, test_easy_answers

    def load_partial_data():
        '''
        Load all queries
        '''
        print("loading data")
        train_queries = pickle.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
        train_answers = pickle.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
        valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
        test_queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))

        return train_queries, train_answers, valid_queries, \
               valid_hard_answers, valid_easy_answers, test_queries, \
               test_hard_answers, test_easy_answers

    train_queries, train_answers, valid_queries, valid_hard_answers, \
    valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data()

    for query_structure in train_queries:
        print(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))

    # for query_structure in train_queries:
    #     print(query_name_dict[query_structure] + ": " + str(len(train_answers[query_structure])))

    # print(train_queries.keys())
    # print(train_answers.keys())
    train_iterators = {}

    print("====== Create Training Iterators  ======")
    model.train()
    for query_structure in train_queries:
        tmp_queries = list(train_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(tmp_queries, nentity, nrelation, negative_sample_size, train_answers),
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        train_iterators[query_name_dict[query_structure]] = new_iterator

        print(query_name_dict[query_structure])

        # positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(new_iterator)
        # print(positive_sample)
        # print(negative_sample)
        # print(subsampling_weight)
        # print(batch_queries)
        # print(query_structure)
        #
        #




        log_of_the_step = model.train_step(model, new_iterator, optimizer, use_apex)
        print(log_of_the_step)

        if query_name_dict[query_structure] in ["1p", "2p", "3p", "2i", "3i"]:
            log_of_the_step = model.train_inv_step(model, new_iterator, optimizer, use_apex)
            print(log_of_the_step)


        print("======")

    print("====== Create Validation Dataloader ======")

    for query_structure in valid_queries:
        logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))

        tmp_queries = list(valid_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        valid_dataloader = DataLoader(
            TestDataset(
                tmp_queries,
                nentity,
                nrelation,
            ),
            batch_size=test_batch_size,
            num_workers=cpu_num,
            collate_fn=TestDataset.collate_fn
        )
        model.eval()
        for negative_sample, queries, queries_unflatten, query_structures in valid_dataloader:
            print(query_name_dict[query_structure])
            # print(len(valid_dataloader))
            # print(len(negative_sample))
            # print(queries)
            # print(queries_unflatten)
            # print(query_structures)

            # The keys for the answer sets are

            # print("valid_easy_answers")
            # for query in queries_unflatten:
            #     print("size easy", len(valid_easy_answers[query]))
            #
            # print("valid_hard_answers")
            # for query in queries_unflatten:
            #     print("size hard", len(valid_hard_answers[query]))

            log_of_the_step = model.evaluate_step(model,
                                                        negative_sample,
                                                        queries,
                                                        queries_unflatten,
                                                        query_structures,
                                                        valid_easy_answers,
                                                        valid_hard_answers)

            print(log_of_the_step)

            print("======")
            break

def test_entailment_loading(data_path="../data/FB15k-237-q2p"):
    print("start open")
    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    print("open finished")
    negative_sample_size = 1
    batch_size = 512
    cpu_num = 1
    test_batch_size = 1
    embedding_size = 400
    num_p = 2
    # print("Load pre_trained embeddings")
    # ent_embedding_path = "../data/transe_batae_fb237_ent.npy"
    # rel_embedding_path = "../data/transe_batae_fb237_rel.npy"

    # with open(ent_embedding_path, 'rb') as f:
    # ent_embedding = np.load(f)
    #
    # with open(rel_embedding_path, 'rb') as f:
    # rel_embedding = np.load(f)

    # ent_embedding = np.load(ent_embedding_path, 'r')
    # rel_embedding = np.load(rel_embedding_path, 'r')

    print("Cuda start")
    model = Query2Particles(nentity, nrelation, embedding_size, num_particles=num_p)

    print("create model")
    model.cuda()

    # if torch.cuda.device_count() > 1:
    #     # print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    no_decay = ['bias', 'layer_norm', 'embedding', "off_sets"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.Adam(
        optimizer_grouped_parameters,
        lr=0.0001,
    )
    print("to Cuda finished")

    use_apex = False

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    def load_data():
        '''
        Load all queries
        '''
        print("loading data")
        train_queries = pickle.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
        train_answers = pickle.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
        valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
        test_queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))

        return train_queries, train_answers, valid_queries, \
               valid_hard_answers, valid_easy_answers, test_queries, \
               test_hard_answers, test_easy_answers

    def load_partial_data():
        '''
        Load all queries
        '''
        print("loading data")
        train_queries = pickle.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
        train_answers = pickle.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
        valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
        test_queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))

        return train_queries, train_answers, valid_queries, \
               valid_hard_answers, valid_easy_answers, test_queries, \
               test_hard_answers, test_easy_answers

    train_queries, train_answers, valid_queries, valid_hard_answers, \
    valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data()

    for query_structure in train_queries:
        print(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))

    for query_structure in valid_queries:
        print(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))

    for query_structure in test_queries:
        print(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))

    # for query_structure in train_queries:
    #     print(query_name_dict[query_structure] + ": " + str(len(train_answers[query_structure])))

    print("====== Create Training Iterators  ======")
    train_iterators = {}
    for query_structure in query_name_dict.keys():
        if "n" in query_name_dict[query_structure] or "DM" in query_name_dict[query_structure]:
            continue
        if query_structure in train_queries:
            tmp_train_queries = list(train_queries[query_structure])
        else:
            tmp_train_queries = []

        if query_structure in valid_queries:
            tmp_valid_queries = list(valid_queries[query_structure])
        else:
            tmp_valid_queries = []

        if query_structure in test_queries:
            tmp_test_queries = list(test_queries[query_structure])
        else:
            tmp_test_queries = []

        print(len(tmp_train_queries))
        print(len(tmp_valid_queries))
        print(len(tmp_test_queries))
        tmp_queries = tmp_train_queries + tmp_valid_queries + tmp_test_queries
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        print(len(tmp_queries))

        all_answers = train_answers

        for other_answers in [valid_easy_answers, valid_hard_answers, test_easy_answers, test_hard_answers]:
            for query, answers in other_answers.items():
                if query in all_answers:
                    all_answers[query] = all_answers[query].union(answers)
                else:
                    all_answers[query] = answers

        new_iterator = SingledirectionalOneShotIterator(DataLoader(
                    TrainDataset(tmp_queries, nentity, nrelation, negative_sample_size, all_answers),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=cpu_num,
                    collate_fn=TrainDataset.collate_fn
                ))
        train_iterators[query_name_dict[query_structure]] = new_iterator
        log_of_the_step = model.train_step(model, new_iterator, optimizer, use_apex)
        print(log_of_the_step)




    # print(train_queries.keys())
    # print(train_answers.keys())
    # train_iterators = {}
    #
    # print("====== Create Training Iterators  ======")
    # model.train()
    # for query_structure in train_queries:
    #     tmp_queries = list(train_queries[query_structure])
    #     tmp_queries = [(query, query_structure) for query in tmp_queries]
    #     new_iterator = SingledirectionalOneShotIterator(DataLoader(
    #         TrainDataset(tmp_queries, nentity, nrelation, negative_sample_size, train_answers),
    #         batch_size=batch_size,
    #         shuffle=True,
    #         num_workers=cpu_num,
    #         collate_fn=TrainDataset.collate_fn
    #     ))
    #     train_iterators[query_name_dict[query_structure]] = new_iterator
    #
    #
    #
    #
    #     print(query_name_dict[query_structure])
    #
    #
    #
    #
    #
    #
    #     log_of_the_step = model.train_step(model, new_iterator, optimizer, use_apex)
    #     print(log_of_the_step)
    #
    #     if query_name_dict[query_structure] in ["1p", "2p", "3p", "2i", "3i"]:
    #         log_of_the_step = model.train_inv_step(model, new_iterator, optimizer, use_apex)
    #         print(log_of_the_step)
    #
    #
    #     print("======")

    print("====== Create Validation Dataloader ======")

    for query_structure in valid_queries:
        logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))

        tmp_queries = list(valid_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        valid_dataloader = DataLoader(
            TestDataset(
                tmp_queries,
                nentity,
                nrelation,
            ),
            batch_size=test_batch_size,
            num_workers=cpu_num,
            collate_fn=TestDataset.collate_fn
        )
        model.eval()
        for negative_sample, queries, queries_unflatten, query_structures in valid_dataloader:
            print(query_name_dict[query_structure])
            # print(len(valid_dataloader))
            # print(len(negative_sample))
            # print(queries)
            # print(queries_unflatten)
            # print(query_structures)

            # The keys for the answer sets are

            # print("valid_easy_answers")
            # for query in queries_unflatten:
            #     print("size easy", len(valid_easy_answers[query]))
            #
            # print("valid_hard_answers")
            # for query in queries_unflatten:
            #     print("size hard", len(valid_hard_answers[query]))

            log_of_the_step = model.evaluate_step(model,
                                                        negative_sample,
                                                        queries,
                                                        queries_unflatten,
                                                        query_structures,
                                                        valid_easy_answers,
                                                        valid_hard_answers)

            print(log_of_the_step)

            print("======")
            break

if __name__ == "__main__":
    # test_dataloading_new()
    # test_dataloading_new("../data/FB15k-237-q2b")
    test_entailment_loading("../data/FB15k-237-q2b")


