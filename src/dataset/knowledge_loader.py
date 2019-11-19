# -*- coding: utf-8 -*-
import numpy as np
import json

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
NONE_ID = 0
START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']
START_KB = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T']


class KnowledgeLoader(object):
    """
    Knowledge loader with simpler functions
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_vocab(self, vocab_size, embed_dim=300):
        print("Loading word vocabs...")
        with open('%s/vocab' % self.data_dir) as fr:
            raw_vocab = json.load(fr)
        vocab_list = START_VOCAB + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
        if len(vocab_list) > vocab_size:
            vocab_list = vocab_list[:vocab_size]
        print("%d words loaded." % len(vocab_list))

        print("Loading word vectors...")
        vectors = {}
        with open('%s/glove.840B.300d.txt' % self.data_dir) as f:
            for i, line in enumerate(f):
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ') + 1:]
                vectors[word] = vector
        embed = []
        for word in vocab_list:
            if word in vectors:
                vector = list(map(float, vectors[word].split()))
            else:
                vector = np.zeros(embed_dim, dtype=np.float32)
            embed.append(vector)
        embed = np.array(embed, dtype=np.float32)
        return vocab_list, embed

    def load_entity_relation(self, trans_dim=100):
        print("Loading entity vocabs...")
        entity_list = []
        with open('%s/entity.txt' % self.data_dir) as f:
            for i, line in enumerate(f):
                e = line.strip()
                entity_list.append(e)
        print("%d entities loaded." % len(entity_list))
        print("Loading relation vocabs...")
        relation_list = []
        with open('%s/relation.txt' % self.data_dir) as f:
            for i, line in enumerate(f):
                r = line.strip()
                relation_list.append(r)
        print("%d relations loaded." % len(relation_list))

        print("Loading entity vectors...")
        entity_embed = []
        with open('%s/entity_transE.txt' % self.data_dir) as f:
            for i, line in enumerate(f):
                s = line.strip().split('\t')
                entity_embed.append(list(map(float, s)))

        print("Loading relation vectors...")
        relation_embed = []
        with open('%s/relation_transE.txt' % self.data_dir) as f:
            for i, line in enumerate(f):
                s = line.strip().split('\t')
                relation_embed.append(s)
        pad_embed = np.zeros((4, trans_dim), dtype=np.float32)
        entity_relation_vocab = START_KB + entity_list + relation_list
        entity_relation_embed = np.array(entity_embed + relation_embed, dtype=np.float32)
        entity_relation_embed = np.concatenate([pad_embed, entity_relation_embed], axis=0)
        print("entity_relation_vocab:", len(entity_relation_vocab))
        print("entity_relation_embed:", entity_relation_embed.shape)
        return entity_relation_vocab, entity_relation_embed

    def load_csk_entities(self):
        csk_entity_list = []
        kb_entity_fp = "%s/csk_entities.txt" % self.data_dir
        print('Loading commonsense entities from [%s]' % kb_entity_fp)
        with open(kb_entity_fp, 'r') as fr:
            for idx, entity in enumerate(fr):
                csk_entity_list.append(entity.strip())
        print("%d csk_entities loaded." % len(csk_entity_list))
        return csk_entity_list

