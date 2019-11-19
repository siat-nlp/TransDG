# -*- coding: utf-8 -*-
import numpy as np
import pickle
from .log_util import LogInfo


class WordEmbeddingUtil(object):

    def __init__(self, emb_dir, dim_emb):
        self.word_dict_fp = '%s/word_emb.indices' % emb_dir
        self.word_emb_mat_fp = '%s/word_emb.glove_%d.npy' % (emb_dir, dim_emb)
        self.mid_dict_fp = '%s/mid_emb.indices' % emb_dir
        self.mid_emb_mat_fp = '%s/mid_emb.glove_%d.npy' % (emb_dir, dim_emb)
        self.dep_name_fp = '%s/dep_names.txt' % emb_dir

        self.dim_emb = dim_emb
        self.word_idx_dict = None
        self.word_emb_matrix = None
        self.n_words = None
        self.mid_idx_dict = None
        self.mid_emb_matrix = None
        self.n_mids = None
        self.dep_name_dict = {}

        self.load_word_indices()
        self.load_mid_indices()
        self.load_dep_names()

    def load_word_indices(self):
        if self.word_idx_dict is None:
            LogInfo.logs('Loading <word, idx> pairs from [%s] ... ', self.word_dict_fp)
            with open(self.word_dict_fp, 'rb') as br:
                self.word_idx_dict = pickle.load(br)
            LogInfo.logs('%d <word, idx> loaded.', len(self.word_idx_dict))
            self.n_words = len(self.word_idx_dict)
    
    def load_mid_indices(self):
        if self.mid_idx_dict is None:
            LogInfo.logs('Loading <mid, idx> pairs from [%s] ... ', self.mid_dict_fp)
            with open(self.mid_dict_fp, 'rb') as br:
                self.mid_idx_dict = pickle.load(br)
            LogInfo.logs('%d <mid, idx> loaded.', len(self.mid_idx_dict))
            self.n_mids = len(self.mid_idx_dict)

    def load_word_embeddings(self):
        if self.word_emb_matrix is None:
            LogInfo.logs('Loading word embeddings for [%s] ...', self.word_emb_mat_fp)
            self.word_emb_matrix = np.load(self.word_emb_mat_fp)
            LogInfo.logs('%s word embedding loaded.', self.word_emb_matrix.shape)
            assert self.word_emb_matrix.shape == (self.n_words, self.dim_emb)

    def load_mid_embeddings(self):
        if self.mid_emb_matrix is None:
            LogInfo.logs('Loading mid embeddings for [%s] ...', self.mid_emb_mat_fp)
            self.mid_emb_matrix = np.load(self.mid_emb_mat_fp)
            LogInfo.logs('%s mid embedding loaded.', self.mid_emb_matrix.shape)
            assert self.mid_emb_matrix.shape == (self.n_mids, self.dim_emb)

    def load_dep_names(self):
        with open(self.dep_name_fp, 'r') as br:
            for line in br.readlines():
                dep, name = line.strip().split('\t')
                self.dep_name_dict[dep] = name
        LogInfo.logs('%d dependency name loaded.', len(self.dep_name_dict))

    def get_phrase_emb(self, phrase):
        """
        Calculate the embedding of a new phrase, by averaging the embeddings of all observed words
        """
        self.load_word_embeddings()
        if phrase == '':
            return None
        spt = phrase.split(' ')
        idx_list = []
        for wd in spt:
            if wd in self.word_idx_dict:
                idx_list.append(self.word_idx_dict[wd])
        if len(idx_list) == 0:
            return None
        emb = np.mean(self.word_emb_matrix[idx_list], axis=0)  # (n_words, dim_emb) ==> (dim_emb, )
        emb = emb / np.linalg.norm(emb)  # normalization
        return emb

    def produce_active_word_embedding(self, active_word_dict, dep_simulate=False):
        active_size = len(active_word_dict)
        word_emb_matrix = np.random.uniform(low=-0.1, high=0.1,
                                            size=(active_size, self.dim_emb)).astype('float32')
        self.load_word_indices()
        self.load_word_embeddings()
        for tok, active_idx in active_word_dict.items():
            if tok in self.word_idx_dict:
                local_idx = self.word_idx_dict[tok]
                word_emb_matrix[active_idx] = self.word_emb_matrix[local_idx]
            elif dep_simulate:       # try to simulate the embedding of dependency label
                dep_tok = tok.replace('!', '')
                if dep_tok in self.dep_name_dict:       # this is a known dependency label
                    dep_name_spt = self.dep_name_dict[dep_tok].split(' ')
                    filt_dep_name_spt = list(filter(lambda x: x in self.word_idx_dict, dep_name_spt))
                    if len(filt_dep_name_spt) == 0:
                        continue
                    # Now simulate dependency embedding by its name
                    simu_emb = np.zeros((self.dim_emb,), dtype='float32')
                    for term in filt_dep_name_spt:
                        term_idx = self.word_idx_dict[term]
                        simu_emb += self.word_emb_matrix[term_idx]
                    simu_emb /= len(filt_dep_name_spt)
                    if not tok.startswith('!'):
                        word_emb_matrix[active_idx] = simu_emb
                    else:       # reversed label
                        word_emb_matrix[active_idx] = -1. * simu_emb

        LogInfo.logs('dependency simulate = %s.', dep_simulate)
        LogInfo.logs('%s active word embedding created.', word_emb_matrix.shape)
        return word_emb_matrix
    
    def produce_active_mid_embedding(self, active_mid_dict):
        active_size = len(active_mid_dict)
        mid_emb_matrix = np.random.uniform(low=-0.1, high=0.1,
                                           size=(active_size, self.dim_emb)).astype('float32')
        self.load_mid_indices()
        self.load_mid_embeddings()
        for tok, active_idx in active_mid_dict.items():
            if tok in self.mid_idx_dict:
                local_idx = self.mid_idx_dict[tok]
                mid_emb_matrix[active_idx] = self.mid_emb_matrix[local_idx]
        LogInfo.logs('%s active mid embedding created.', mid_emb_matrix.shape)
        return mid_emb_matrix
