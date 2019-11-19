# -*- coding: utf-8 -*-
"""
Wrapper for training
"""
import os
import numpy as np


class Trainer:

    def __init__(self, model, sess, trans_model, trans_sess, saver, best_saver, log_dir, save_per_step=1000):
        self.model = model
        self.sess = sess
        self.trans_model = trans_model
        self.trans_sess = trans_sess
        self.saver = saver
        self.best_saver = best_saver
        self.model_dir = "%s/models" % log_dir
        self.best_model_dir = "%s/models/best_model" % log_dir
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)
        self.save_per_step = save_per_step

        self.scan_data = 0
        self.scan_batch = 0
        self.best_eval_loss = 1e5

    def _reset_optm_info(self):
        self.scan_data = 0
        self.scan_batch = 0

    def _optimize(self, batch_data, eval_loader, queue_size):
        if self.trans_model is not None:
            # transfer representation
            trans_repr = self.trans_model.transfer_encode(self.trans_sess, batch_data)
        else:
            trans_repr = None

        # model training
        ppx_loss, local_loss = self.model.train_batch(self.sess, batch_data, trans_repr)

        avg_ppx = float(np.mean(ppx_loss))
        avg_loss = float(np.mean(local_loss))

        if self.model.global_step.eval(session=self.sess) % 100 == 0:
            print("[queue=%d scan=%d] global step: %d loss: %.3f ppx_loss: %.3f perplexity: %.3f" % (
                queue_size, self.scan_data, self.model.global_step.eval(session=self.sess),
                avg_loss, avg_ppx, np.exp(avg_ppx)))

        if self.scan_batch % self.save_per_step == 0:
            self.saver.save(self.sess, save_path="%s/model" % self.model_dir, global_step=self.model.global_step)
            # eval model
            all_eval_loss = []
            all_eval_ppx = []
            for idx in range(eval_loader.n_batch):
                eval_data, _ = eval_loader.get_batch(batch_idx=idx)
                if self.trans_model is not None:
                    eval_repr = self.trans_model.transfer_encode(self.trans_sess, eval_data)
                else:
                    eval_repr = None
                eval_ppx, eval_loss = self.model.eval_batch(self.sess, eval_data, eval_repr)
                all_eval_loss.append(float(np.mean(eval_loss)))
                all_eval_ppx.append(float(np.mean(eval_ppx)))
            avg_eval_loss = np.mean(all_eval_loss)
            avg_eval_ppx = np.mean(all_eval_ppx)
            if avg_eval_loss < self.best_eval_loss:
                self.best_eval_loss = float(avg_eval_loss)
                self.best_saver.save(self.sess, save_path="%s/best.model" % self.best_model_dir)
                print("Eval loss=%.3f ppx=%.3f. Saved to [%s/best.model]" %
                      (self.best_eval_loss, np.exp(avg_eval_ppx), self.best_model_dir))

    def train(self, train_batcher, eval_loader, epoch_idx):
        self._reset_optm_info()
        while train_batcher.get_epoch() == epoch_idx:
            queue_size = train_batcher.loader_queue_size
            batch_data, local_size = train_batcher.get_batch()
            #print("posts:", batch_data['post'][:2])
            #print("response:", batch_data['response'][:2])
            #print("triples:", batch_data['all_triples'][:2])
            #print("entities:", batch_data['all_entities'][:2])
            self.scan_data += local_size
            self.scan_batch += 1
            self._optimize(batch_data=batch_data, eval_loader=eval_loader, queue_size=queue_size)
        else:
            return
