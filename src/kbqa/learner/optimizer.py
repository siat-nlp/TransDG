# -*- coding: utf-8 -*-
"""
Wrapper for training different kernels
"""
from ..utils.log_util import LogInfo


class Optimizer:

    def __init__(self, model, sess, ob_batch_num=100):
        self.model = model
        self.sess = sess
        self.ob_batch_num = ob_batch_num
        self.scan_data = 0
        self.scan_batch = 0
        self.ret_loss = 0
        self.tb_point = 0 

    def _reset_optm_info(self):
        self.scan_data = self.scan_batch = 0
        self.ret_loss = 0.

    def _optimize(self, optm_data_loader, batch_idx):
        local_data, local_size = optm_data_loader.get_batch(batch_idx=batch_idx)

        _, local_loss, summary = self.model.train_batch(self.sess, local_data)
        local_loss = float(local_loss)
        self.ret_loss = 1.0 * (self.ret_loss * self.scan_data + local_loss * local_size) / (self.scan_data + local_size)
        self.scan_data += local_size
        self.scan_batch += 1
        self.tb_point += 1
        if self.scan_batch % self.ob_batch_num == 0:
            LogInfo.logs('[optm-%s-%d/%d] cur_batch_loss = %.6f, avg_loss = %.6f, scanned = %d/%d',
                         optm_data_loader.mode,
                         self.scan_batch,
                         optm_data_loader.n_batch,
                         local_loss,
                         self.ret_loss,
                         self.scan_data,
                         len(optm_data_loader))

    def optimize_all(self, optm_data_loader):
        self._reset_optm_info()
        for batch_idx in range(optm_data_loader.n_batch):
            self._optimize(optm_data_loader=optm_data_loader, batch_idx=batch_idx)
