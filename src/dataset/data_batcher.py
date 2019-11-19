# -*- coding: utf-8 -*-
import numpy as np
import pickle
import time
from queue import Queue
from threading import Thread
from .data_loader import DataLoader

CHUNK_NUM = 20


class DataBatcher(object):
    """
        Data batcher with queue for loading big dataset
    """

    def __init__(self, data_dir, file_list, batch_size, num_epoch, shuffle=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.shuffle = shuffle

        self.cur_epoch = 0
        self.loader_queue = Queue(maxsize=CHUNK_NUM)
        self.loader_queue_size = 0
        self.batch_iter = self.batch_generator()
        self.input_gen = self.loader_generator()

        # Start the threads that load the queues
        self.loader_q_thread = Thread(target=self.fill_loader_queue)
        self.loader_q_thread.setDaemon(True)
        self.loader_q_thread.start()

        # Start a thread that watches the other threads and restarts them if they're dead
        self.watch_thread = Thread(target=self.monitor_threads)
        self.watch_thread.setDaemon(True)
        self.watch_thread.start()

    def get_batch(self):
        try:
            batch_data, local_size = next(self.batch_iter)
        except StopIteration:
            batch_data = None
            local_size = 0
        return batch_data, local_size

    def get_epoch(self):
        return self.cur_epoch

    def full(self):
        if self.loader_queue_size == CHUNK_NUM:
            return True
        else:
            return False

    def batch_generator(self):
        while self.loader_queue_size > 0:
            data_loader = self.loader_queue.get()
            n_batch = data_loader.n_batch
            self.loader_queue_size -= 1
            for batch_idx in range(n_batch):
                batch_data, local_size = data_loader.get_batch(batch_idx=batch_idx)
                yield batch_data, local_size

    def loader_generator(self):
        for epoch in range(self.num_epoch):
            self.cur_epoch = epoch
            if self.shuffle:
                np.random.shuffle(self.file_list)
            for idx, f in enumerate(self.file_list):
                reader = open("%s/%s" % (self.data_dir, f), 'br')
                q_dict = pickle.load(reader)
                data_loader = DataLoader(batch_size=self.batch_size)
                data_loader.feed_by_data(q_dict)
                yield data_loader

    def fill_loader_queue(self):
        while True:
            if self.loader_queue_size <= CHUNK_NUM:
                try:
                    data_loader = next(self.input_gen)
                    self.loader_queue.put(data_loader)
                    self.loader_queue_size += 1
                except StopIteration:
                    break

    def monitor_threads(self):
        """Watch loader queue thread and restart if dead."""
        while True:
            time.sleep(60)
            if not self.loader_q_thread.is_alive():  # if the thread is dead
                print('Found loader queue thread dead. Restarting.')
                new_t = Thread(target=self.fill_loader_queue)
                self.loader_q_thread = new_t
                new_t.daemon = True
                new_t.start()
