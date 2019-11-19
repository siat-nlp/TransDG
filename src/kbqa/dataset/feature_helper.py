# -*- coding: utf-8 -*-
"""
Given the whole dataset and one schema, generate different part of inputs
"""
import numpy as np
from ..utils.dependency_util import DependencyUtil


class FeatureHelper:
    def __init__(self, active_dicts, qa_list, freebase_helper,
                 path_max_size=3, qw_max_len=20, pw_max_len=8, pseq_max_len=3):
        self.word_idx_dict = active_dicts['word']
        self.path_idx_dict = active_dicts['path']
        self.mid_idx_dict = active_dicts['mid']
        self.qa_list = qa_list
        self.freebase_helper = freebase_helper
        self.path_max_size = path_max_size
        self.qw_max_len = qw_max_len
        self.pw_max_len = pw_max_len
        self.pseq_max_len = pseq_max_len

        self.dep_util = DependencyUtil()

    def generate_qw_feat(self, sc):
        q_idx = sc.q_idx
        qa = self.qa_list[q_idx]
        lower_tok_list = [tok.lower() for tok in qa['tokens']]

        for raw_path in sc.raw_paths:
            category, gl_data, _ = raw_path
            if gl_data.category == 'Entity' and gl_data.end <= len(lower_tok_list):
                for idx in range(gl_data.start, gl_data.end - 1):
                    lower_tok_list[idx] = ''
                lower_tok_list[gl_data.end - 1] = '<E>'
            elif gl_data.category == 'Tm' and gl_data.end <= len(lower_tok_list):
                for idx in range(gl_data.start, gl_data.end-1):
                    lower_tok_list[idx] = ''
                lower_tok_list[gl_data.end-1] = '<Tm>'

        ph_lower_tok_list = list(filter(lambda x: x != '', lower_tok_list))
        ph_qw_idx_seq = [self.word_idx_dict.get(token, 2) for token in ph_lower_tok_list]

        ph_len = min(self.qw_max_len, len(ph_lower_tok_list))
        qw_input = np.zeros((self.path_max_size, self.qw_max_len), dtype='int32')
        qw_len = np.zeros((self.path_max_size,), dtype='int32')
        for path_idx in range(self.path_max_size):
            qw_input[path_idx, :ph_len] = ph_qw_idx_seq[:ph_len]
            qw_len[path_idx] = ph_len
        return qw_input, qw_len

    def generate_dep_feat(self, sc):
        q_idx = sc.q_idx
        qa = self.qa_list[q_idx]
        tok_list = [tok.lower() for tok in qa['tokens']]
        linkings = [raw_path[1] for raw_path in sc.raw_paths]

        dep_path_tok_lists = self.dep_util.context_pattern(tok_list=tok_list, linkings=linkings)

        dep_input = np.zeros((self.path_max_size, self.qw_max_len), dtype='int32')
        dep_len = np.zeros((self.path_max_size,), dtype='int32')
        for path_idx, local_dep_seq in enumerate(dep_path_tok_lists):
            local_len = min(self.qw_max_len, len(local_dep_seq))
            local_dep_idx_seq = [self.word_idx_dict.get(token, 2) for token in local_dep_seq]
            dep_input[path_idx, :local_len] = local_dep_idx_seq[:local_len]
            dep_len[path_idx] = local_len
        return dep_input, dep_len

    def generate_whole_path_feat(self, sc):
        path_size = len(sc.raw_paths)
        path_ids = np.zeros((self.path_max_size,), dtype='int32')
        pw_input = np.zeros((self.path_max_size, self.pw_max_len), dtype='int32')
        pw_len = np.zeros((self.path_max_size,), dtype='int32')
        pseq_ids = np.zeros((self.path_max_size, self.pseq_max_len), dtype='int32')
        pseq_len = np.zeros((self.path_max_size,), dtype='int32')
        sc.path_words_list = []

        for path_idx, (raw_path, mid_seq) in enumerate(zip(sc.raw_paths, sc.path_list)):
            path_cate = raw_path[0]
            path_str = '%s|%s' % (path_cate, '\t'.join(mid_seq))
            path_ids[path_idx] = self.path_idx_dict.get(path_str, 2)

            local_words = []  # collect path words
            pseq_len[path_idx] = len(mid_seq)
            for mid_pos, mid in enumerate(mid_seq):
                pseq_ids[path_idx, mid_pos] = self.mid_idx_dict.get(mid, 2)
                p_name = self.freebase_helper.get_item_name(mid)
                if p_name != '':
                    spt = p_name.split(' ')
                    local_words += spt
            sc.path_words_list.append(local_words)
            local_pw_idx_seq = [self.word_idx_dict.get(token, 2) for token in local_words]
            local_pw_len = min(self.pw_max_len, len(local_pw_idx_seq))
            pw_input[path_idx, :local_pw_len] = local_pw_idx_seq[:local_pw_len]
            pw_len[path_idx] = local_pw_len

        return path_size, path_ids, pw_input, pw_len, pseq_ids, pseq_len
