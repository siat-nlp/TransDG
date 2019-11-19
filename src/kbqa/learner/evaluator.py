# -*- coding: utf-8 -*-
"""
Wrapper for evaluating different kernels
"""
import numpy as np

from ..utils.log_util import LogInfo


class Evaluator:

    def __init__(self, model, sess, ob_batch_num=100, show_detail=True):
        self.model = model
        self.sess = sess
        self.ob_batch_num = ob_batch_num
        self.show_detail = show_detail

        self.scan_data = 0
        self.scan_batch = 0
        self.tb_point = 0       # x-axis in tensorboard
        self.eval_detail_dict = {}      # store evaluation detail of each data

    def _reset_eval_info(self):
        self.scan_data = self.scan_batch = 0
        self.eval_detail_dict = {}

    def _evaluate(self, eval_data_loader, batch_idx):
        local_data, local_size = eval_data_loader.get_batch(batch_idx=batch_idx)

        rm_score, rm_final_feats = self.model.eval_batch(self.sess, local_data)
        local_eval_dict = {'rm_score': rm_score,
                           'rm_final_feats': rm_final_feats}

        for tensor_name, batch_val in local_eval_dict.items():
            for val in batch_val:
                self.eval_detail_dict.setdefault(tensor_name, []).append(val)

        # Collect all input / outputs of this batch, saving into eval_detail_dict (split by each data point)
        self.scan_data += local_size
        self.scan_batch += 1
        self.tb_point += 1
        if self.scan_batch % self.ob_batch_num == 0:
            LogInfo.logs('[eval-%s-%d/%d] scanned = %d/%d',
                         eval_data_loader.mode,
                         self.scan_batch,
                         eval_data_loader.n_batch,
                         self.scan_data,
                         len(eval_data_loader))
    
    def _post_process(self, eval_data_loader, detail_fp, result_fp):
        """
        Given all the evaluation detail, calculate the final result.
        """
        assert len(self.eval_detail_dict) > 0

        ret_q_score_dict = {}
        for scan_idx, (q_idx, cand) in enumerate(eval_data_loader.eval_sc_tup_list):
            # put all output results into sc.run_info
            cand.run_info = {k: data_values[scan_idx] for k, data_values in self.eval_detail_dict.items()}
            ret_q_score_dict.setdefault(q_idx, []).append(cand)

        score_key = 'rm_score'
        f1_key = 'rm_f1'
        f1_list = []
        for q_idx, score_list in ret_q_score_dict.items():
            score_list.sort(key=lambda x: x.run_info[score_key], reverse=True)  # sort by score DESC
            if len(score_list) == 0:
                f1_list.append(0.)
            else:
                f1_list.append(getattr(score_list[0], f1_key))

        LogInfo.logs('Predict %d out of %d questions.', len(f1_list), eval_data_loader.total_questions)
        ret_metric = np.sum(f1_list).astype('float32') / eval_data_loader.total_questions

        if detail_fp is not None:
            bw = open(detail_fp, 'w')
            LogInfo.redirect(bw)
            #np.set_printoptions(threshold=np.nan)
            LogInfo.logs('Avg_f1 = %.6f', ret_metric)
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            for q_idx in srt_q_idx_list:
                LogInfo.begin_track('Q-%04d:', q_idx)
                srt_list = ret_q_score_dict[q_idx]  # already sorted
                best_label_f1 = np.max([getattr(sc, f1_key) for sc in srt_list])
                best_label_f1 = max(best_label_f1, 0.000001)
                for rank, sc in enumerate(srt_list):
                    cur_f1 = getattr(sc, f1_key)
                    if rank < 20 or cur_f1 == best_label_f1:
                        LogInfo.begin_track('#-%04d [F1 = %.6f] [row_in_file = %d]', rank+1, cur_f1, sc.ori_idx)
                        LogInfo.logs('%s: %.6f', score_key, sc.run_info[score_key])
                        if self.show_detail:
                            self.show_rm_info(sc)
                        else:
                            LogInfo.logs('Current: not output detail.')
                        LogInfo.end_track()
                LogInfo.end_track()
            LogInfo.logs('Avg_f1 = %.6f', ret_metric)

            #np.set_printoptions()  # reset output format
            LogInfo.stop_redirect()
            bw.close()

        # Save detail information
        if result_fp is not None:
            srt_q_idx_list = sorted(ret_q_score_dict.keys())
            with open(result_fp, 'w') as bw:  # write question --> selected schema
                for q_idx in srt_q_idx_list:
                    srt_list = ret_q_score_dict[q_idx]
                    ori_idx = -1
                    task_f1 = 0.
                    if len(srt_list) > 0:
                        best_sc = srt_list[0]
                        ori_idx = best_sc.ori_idx
                        task_f1 = getattr(best_sc, f1_key)
                    bw.write('%d\t%d\t%.6f\n' % (q_idx, ori_idx, task_f1))

        return ret_metric

    @staticmethod
    def show_rm_info(sc):
        rm_final_feats = sc.run_info['rm_final_feats'].tolist()
        LogInfo.logs('rm_final_feats = [%s]', ' '.join(['%6.3f' % x for x in rm_final_feats]))

        show_path_list = sc.path_list
        show_path_words_list = sc.path_words_list
        show_path_size = len(show_path_list)

        # show the detail of each path one by one
        for path_idx in range(show_path_size):
            LogInfo.begin_track('Showing path-%d / %d:', path_idx + 1, show_path_size)
            LogInfo.logs('Path: [%s]', '-->'.join(show_path_list[path_idx]))
            LogInfo.logs('Path-Word: [%s]', ' | '.join(show_path_words_list[path_idx]))
            LogInfo.end_track()
    
    def evaluate_all(self, eval_data_loader, detail_fp=None, result_fp=None):
        self._reset_eval_info()
        for batch_idx in range(eval_data_loader.n_batch):
            self._evaluate(eval_data_loader=eval_data_loader, batch_idx=batch_idx)

        ret_f1 = self._post_process(eval_data_loader=eval_data_loader, detail_fp=detail_fp, result_fp=result_fp)
        LogInfo.logs('%s_F1 = %.6f', eval_data_loader.mode, ret_f1)
        return ret_f1
