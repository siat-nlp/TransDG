# -*- coding: utf-8 -*-
import numpy as np
import pickle
import time


class DataLoader(object):
    """
    Data loader with simpler functions
    """

    def __init__(self, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_data_list = []
        self.batch_size_list = []
        self.n_batch = None  # number of batches
        self.n_rows = None   # number of samples

    def __len__(self):
        return self.n_rows

    def get_batch(self, batch_idx):
        local_data = self.batch_data_list[batch_idx]
        local_size = self.batch_size_list[batch_idx]
        return local_data, local_size

    def prepare_input_list(self, global_input_dict, n_rows):
        """
        :param global_input_dict: <input_name, [np_value of each data point]>
        :param n_rows: total number of data
        """
        if self.shuffle:
            np.random.shuffle(global_input_dict)

        self.n_rows = remain_rows = n_rows
        while remain_rows > 0:
            self.batch_data_list.append({})
            active_size = min(remain_rows, self.batch_size)
            self.batch_size_list.append(active_size)
            remain_rows -= active_size
        self.n_batch = len(self.batch_size_list)
        #print('n_rows = %d, batch_size = %d, n_batch = %d.' % (self.n_rows, self.batch_size, self.n_batch))

        for key, input_data_list in global_input_dict.items():
            if len(input_data_list) != n_rows:
                print('Warning: len(%s) = %d, mismatch with n_row = %d.' % (key, len(input_data_list), n_rows))
                continue
            for batch_idx in range(self.n_batch):
                st_idx = batch_idx * self.batch_size
                ed_idx = st_idx + self.batch_size
                local_batch_input = input_data_list[st_idx: ed_idx]
                self.batch_data_list[batch_idx][key] = np.array(local_batch_input)

    def load_data(self, file_path):
        print("Loading data from [%s]" % file_path)
        t0 = time.time()
        with open(file_path, 'rb') as br:
            q_dict = pickle.load(br)
        n_rows = len(q_dict)

        global_input_dict = {}
        print("Loaded %d data. [%.3fs]" % (n_rows, time.time()-t0))
        for q_idx, input_np_dict in q_dict.items():
            for k, v in input_np_dict.items():
                global_input_dict.setdefault(k, []).append(v)
        self.prepare_input_list(global_input_dict=global_input_dict, n_rows=n_rows)

    def feed_by_data(self, q_dict):
        n_rows = len(q_dict)
        global_input_dict = {}
        for q_idx, input_np_dict in q_dict.items():
            for k, v in input_np_dict.items():
                global_input_dict.setdefault(k, []).append(v)
        self.prepare_input_list(global_input_dict=global_input_dict, n_rows=n_rows)
