# -*- coding: utf-8 -*-
import numpy as np
from ..utils.log_util import LogInfo


class BaseDataLoader(object):
    """
    Data loader with simpler functions
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch_np_data_list = None
        self.batch_real_size_list = None
        self.n_batch = None  # number of batches
        self.n_rows = None

    def __len__(self):
        return self.n_rows

    def get_batch(self, batch_idx):
        local_data = self.batch_np_data_list[batch_idx]
        local_size = self.batch_real_size_list[batch_idx]
        return local_data, local_size

    def prepare_np_input_list(self, global_input_dict, n_rows):
        """
        :param global_input_dict:   <input_name, [np_value of each data point]>
        :param n_rows: total number of data
        """
        self.batch_np_data_list = []
        self.batch_real_size_list = []
        self.n_rows = remain_rows = n_rows
        while remain_rows > 0:
            self.batch_np_data_list.append({})
            active_size = min(remain_rows, self.batch_size)
            self.batch_real_size_list.append(active_size)
            remain_rows -= active_size
        self.n_batch = len(self.batch_real_size_list)
        LogInfo.logs('n_rows = %d, batch_size = %d, n_batch = %d.',
                     self.n_rows, self.batch_size, self.n_batch)

        for key, input_data_list in global_input_dict.items():
            if len(input_data_list) != n_rows:
                # len == n_rows: input values for all data points are present
                # len == 0: direct input is not available (generated in post-process)
                # Not allowed if input values are partially provided
                LogInfo.logs('Warning: len(%s) = %d, mismatch with n_row = %d.',
                             key, len(input_data_list), n_rows)
                continue
            for batch_idx in range(self.n_batch):
                st_idx = batch_idx * self.batch_size
                ed_idx = st_idx + self.batch_size
                local_batch_input = input_data_list[st_idx: ed_idx]
                np_arr = np.array(local_batch_input)
                self.batch_np_data_list[batch_idx][key] = np_arr


class SchemaEvalDataLoader(BaseDataLoader):

    def __init__(self, q_evals_dict, mode, batch_size, shuffle=False):
        """
        :param q_evals_dict: <q, [sc]>
        :param mode: train / valid / test, just for display
        :param batch_size:
        :param shuffle: shuffle all candidate schemas or not
        """
        BaseDataLoader.__init__(self, batch_size=batch_size)

        self.mode = mode
        self.eval_sc_tup_list = []      # [(q_idx, sc)], used for tracing the original feed data
        self.total_questions = len(q_evals_dict)
        for q_idx, eval_list in q_evals_dict.items():
            for sc in eval_list:
                self.eval_sc_tup_list.append((q_idx, sc))
        n_rows = len(self.eval_sc_tup_list)

        self.eval_sc_tup_list.sort(key=lambda _tup: _tup[0])  # just sort by q_idx
        if shuffle:
            np.random.shuffle(self.eval_sc_tup_list)

        global_input_dict = {}
        for q_idx, sc in self.eval_sc_tup_list:
            sc_np_dict = sc.input_np_dict       # Already generated features
            for k, v in sc_np_dict.items():
                global_input_dict.setdefault(k, []).append(v)
        LogInfo.logs('%d total questions, %d schemas saved in dataloader [%s].',
                     self.total_questions, n_rows, mode)

        self.prepare_np_input_list(global_input_dict=global_input_dict, n_rows=n_rows)


class SchemaOptmDataLoader(BaseDataLoader):

    def __init__(self, q_optm_pairs_dict, mode, batch_size, shuffle=False):
        """
        :param q_optm_pairs_dict: <q, [(sc+, sc-)]>
        :param mode: train / valid / test, just for display
        :param batch_size:
        :param shuffle: shuffle all candidate schemas or not
        """
        BaseDataLoader.__init__(self, batch_size=batch_size)

        self.mode = mode
        self.optm_pair_tup_list = []      # [(q_idx, sc+, sc-)], used for tracing the original feed data
        self.total_questions = len(q_optm_pairs_dict)
        for q_idx, optm_pairs in q_optm_pairs_dict.items():
            for pos_sc, neg_sc in optm_pairs:
                self.optm_pair_tup_list.append((q_idx, pos_sc, neg_sc))
        n_pairs = len(self.optm_pair_tup_list)

        self.optm_pair_tup_list.sort(key=lambda _tup: _tup[0])  # just sort by q_idx
        if shuffle:
            np.random.shuffle(self.optm_pair_tup_list)

        global_input_dict = {}
        for q_idx, pos_sc, neg_sc in self.optm_pair_tup_list:
            for sc in (pos_sc, neg_sc):
                sc_np_dict = sc.input_np_dict       # Already generated features
                for k, v in sc_np_dict.items():
                    global_input_dict.setdefault(k, []).append(v)
        LogInfo.logs('%d <pos, neg> pairs saved in dataloader [%s].', n_pairs, mode)

        self.prepare_np_input_list(global_input_dict=global_input_dict, n_rows=2*n_pairs)
