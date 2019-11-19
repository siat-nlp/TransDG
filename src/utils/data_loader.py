# -*- coding: utf-8 -*-
import numpy as np
import json
import tensorflow as tf
import pickle

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
NONE_ID = 0
START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']


def load_data(data_dir, is_train=False):
    data_train, data_dev, data_test = [], [], []
    if is_train:
        with open('%s/train.txt' % data_dir) as f:
            for idx, line in enumerate(f):
                # ========================================================
                data_train.append(json.loads(line))
                if idx % 100000 == 0 and idx > 0:
                    print('read train file line %d' % idx)
                #if idx < 200000:
                #    data_train.append(json.loads(line))
                # ========================================================
        with open('%s/valid.txt' % data_dir) as f:
            for line in f:
                data_dev.append(json.loads(line))

    else:
        with open('%s/test.txt' % data_dir) as f:
            for line in f:
                data_test.append(json.loads(line))
    if is_train:
        return data_train, data_dev
    else:
        return data_test


def load_trans_data(dir, is_train=False):
    data_trans_train = {}
    data_trans_valid = {}
    data_trans_test = {}

    if is_train:
        # TODO：revise to automatically check
        # ========================================================
        train_list = list(range(0, 34))
        #train_list = list(range(0, 2))    # we only get 200000 data
        # ========================================================
        train_list = [str(id) for id in train_list]
        sent_repr = []
        rm_final_feats = []
        for idx in train_list:
            with open('%s/data_trans_train_%s.picke' % (dir, idx), 'rb') as fr:
                print("read data_trans_tran_%s.pickle" % idx)
                trans_bucket = pickle.load(fr)
                sent_bucket = trans_bucket['sent_repr'].tolist()
                rm_feat_bucket = trans_bucket['rm_final_feats'].tolist()
                for sent, rm_feat in zip(sent_bucket, rm_feat_bucket):
                    sent_repr.append(sent)
                    rm_final_feats.append(rm_feat)

        data_trans_train = {'sent_repr': np.array(sent_repr),
                            'rm_final_feats': np.array(rm_final_feats)}
        print("trans_sent_repr:", data_trans_train['sent_repr'].shape)
        print("trans_rm_feats:", data_trans_train['rm_final_feats'].shape)

        with open('%s/data_trans_valid.picke' % dir, 'rb') as fr:
            data_trans_valid = pickle.load(fr)

    else:
        with open('%s/data_trans_test.picke' % dir, 'rb') as fr:
            data_trans_test = pickle.load(fr)
    if is_train:
        return data_trans_train, data_trans_valid
    else:
        return data_trans_test


def load_vocab(dir, vocab_size, embed_units):
    print("loading word vocabs...")
    with open('%s/vocab' % dir) as fr:
        raw_vocab = json.load(fr)

    vocab_list = START_VOCAB + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
    if len(vocab_list) > vocab_size:
        vocab_list = vocab_list[:vocab_size]

    print("loading entity list...")
    entity_list = []
    with open('%s/csk_entity.txt' % dir) as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)

    print("loading word vectors...")
    vectors = {}
    with open('%s/glove.840B.300d.txt' % dir) as f:
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
            vector = np.zeros(embed_units, dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)

    entity_embed = []
    '''
    for entity_word in entity_list:
        if entity_word in vectors:
            vector = list(map(float, vectors[entity_word].split()))
            entity_embed.append(vector)
    entity_embed = np.array(entity_embed, dtype=np.float32)
    '''
    return vocab_list, embed, entity_embed


def pad_batched_data(batched_data):
    batched_post_tokens = [item['post'].split() for item in batched_data]
    batched_res_tokens = [item['response'].split() for item in batched_data]

    encoder_len = max([len(p) for p in batched_post_tokens]) + 1
    decoder_len = max([len(r) for r in batched_res_tokens]) + 1
    posts, responses, posts_length, responses_length = [], [], [], []
    for token_list in batched_post_tokens:
        posts.append(_padding(token_list, encoder_len))
        posts_length.append(len(token_list) + 1)
    for token_list in batched_res_tokens:
        responses.append(_padding(token_list, decoder_len))
        responses_length.append(len(token_list) + 1)

    batched_corrs = [item['corr_responses'] for item in batched_data]
    corr_responses = []
    for corrs in batched_corrs:
        response_k = []
        for res in corrs:
            tokens = res.split()
            token_pad = _pad_corr_res(tokens, decoder_len)
            response_k.append(token_pad)
        corr_responses.append(response_k)

    paded_data = {'posts': np.array(posts),
                  'responses': np.array(responses),
                  'posts_length': posts_length,
                  'responses_length': responses_length,
                  'corr_responses': np.array(corr_responses)}
    return paded_data


def _pad_corr_res(s, max_len):
    if len(s) >= max_len - 1:
        sentence = s[:max_len - 1] + ['_EOS']
    else:
        sentence = s + ['_EOS'] + ['_PAD'] * (max_len - len(s) - 1)
    return sentence


def _padding(sent, l):
    return sent + ['_EOS'] + ['_PAD'] * (l - len(sent) - 1)
