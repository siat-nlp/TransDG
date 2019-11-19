# -*- coding: utf-8 -*-
from .data_loader import SchemaOptmDataLoader, SchemaEvalDataLoader
from ..utils.log_util import LogInfo
import numpy as np


train_pos = 75910
valid_pos = 86755


class SchemaBuilder:

    def __init__(self, schema_dataset, feature_helper, neg_f1_ths, neg_max_sample, neg_strategy):
        LogInfo.logs('SchemaBuilder initializing ...')
        self.schema_dataset = schema_dataset
        self.feat_gen_helper = feature_helper
        self.neg_f1_ths = neg_f1_ths
        self.neg_max_sample = neg_max_sample
        self.neg_strategy = neg_strategy
        self.q_optm_pairs_dict, self.q_evals_dict = self._prepare_optm_eval_data()

    def build_eval_dataloader(self, eval_batch_size, shuffle=False):
        LogInfo.begin_track('Building eval data loader...')
        q_optm_pairs_dict = self.q_optm_pairs_dict
        q_evals_dict = self.q_evals_dict

        full_q_idx_list = sorted(q_optm_pairs_dict.keys())
        train_q_idx_list = list(filter(lambda x: x < train_pos, full_q_idx_list))
        valid_q_idx_list = list(filter(lambda x: train_pos <= x < valid_pos, full_q_idx_list))
        test_q_idx_list = list(filter(lambda x: x >= valid_pos, full_q_idx_list))

        modes = ['train', 'valid', 'test']
        q_idx_lists = [train_q_idx_list, valid_q_idx_list, test_q_idx_list]
        eval_dl_list = []
        for mode, cur_q_idx_list in zip(modes, q_idx_lists):
            cur_q_evals_dict = {q_idx: q_evals_dict[q_idx] for q_idx in cur_q_idx_list}
            eval_dl = SchemaEvalDataLoader(
                q_evals_dict=cur_q_evals_dict,
                mode=mode, batch_size=eval_batch_size, shuffle=shuffle
            )
            eval_dl_list.append(eval_dl)

        LogInfo.end_track('Eval train/valid/test data loader returned.')
        return eval_dl_list

    def build_optm_dataloader(self, optm_batch_size, shuffle=True):
        LogInfo.begin_track('Building optm data loader...')
        q_optm_pairs_dict = self.q_optm_pairs_dict
        full_q_idx_list = sorted(q_optm_pairs_dict.keys())
        train_q_idx_list = list(filter(lambda x: x < train_pos, full_q_idx_list))

        optm_dl = SchemaOptmDataLoader(
            q_optm_pairs_dict={q_idx: q_optm_pairs_dict[q_idx] for q_idx in train_q_idx_list},
            mode='optm', batch_size=optm_batch_size, shuffle=shuffle
        )

        LogInfo.end_track('Optm dataloader returned.')
        return optm_dl

    def _prepare_optm_eval_data(self):
        """
        Given the fixed S-MART based schemas and dynamic generated schemas,
        build the Optm/T/v/t data loader for the specific epoch.
        That's to say, we control the negative sampling strategy here.
        Negative strategy:
            1. fix: (threshold + random sample)
            2. dynamic: (threshold + weighed sample based on delta)
        """
        runtime_score_key = 'rm_score'
        LogInfo.logs('runtime_score_key = [%s]', runtime_score_key)

        q_optm_pairs_dict = {}  # < q_idx, [(pos_sc, neg_sc)] >
        q_evals_dict = {}       # < q_idx, [sc] >
        total_optm_pair_size = 0
        total_eval_sc_size = 0
        schema_dataset = self.schema_dataset

        for scan_idx, q_idx in enumerate(schema_dataset.q_idx_list):
            if scan_idx > 0 and scan_idx % 1000 == 0:
                LogInfo.logs('scanned %d / %d questions. optm_pairs = %d, eval_sc = %d.',
                             scan_idx, len(schema_dataset.q_idx_list),
                             total_optm_pair_size, total_eval_sc_size)

            cand_list = schema_dataset.smart_q_cand_dict[q_idx]
            dedup_sc_tups = self._schema_dedup(sc_list=cand_list)   # [(sc, sc.f1)]
            np.random.shuffle(dedup_sc_tups)    # shuffle, avoid data leaking.
            pos_tups = list(filter(lambda _tup: _tup[-1] >= self.neg_f1_ths, dedup_sc_tups))
            neg_tups = list(filter(lambda _tup: _tup[-1] < self.neg_f1_ths, dedup_sc_tups))

            for sc, _ in dedup_sc_tups:
                self._input_feat_gen(sc)     # Generate input features here

            eval_sc_list = list(filter(lambda _sc: _sc.ans_size > 0, [tup[0] for tup in dedup_sc_tups]))

            optm_sc_pair_list = []
            for sc1, f1_1 in pos_tups:
                if self.neg_strategy == 'Fix':
                    for sc2, f1_2 in pos_tups:       # both sc+ and sc- come from positive list
                        if f1_1 > f1_2:
                            optm_sc_pair_list.append((sc1, sc2))

                    np.random.shuffle(neg_tups)
                    for sc2, _ in neg_tups[:self.neg_max_sample]:    # sc- comes from negative list
                        optm_sc_pair_list.append((sc1, sc2))
                else:
                    sample_tups = []
                    runtime_score_1 = (0. if sc1.run_info is None else
                                       sc1.run_info.get(runtime_score_key, 0.))
                    for sc2, f1_2 in pos_tups + neg_tups:
                        if f1_1 > f1_2:
                            runtime_score_2 = (0. if sc2.run_info is None else
                                               sc2.run_info.get(runtime_score_key, 0.))
                            delta = runtime_score_2 - runtime_score_1   # the larger, the more critical
                            sample_tups.append((sc1, sc2, delta))
                    local_picked_tups = self._weighted_sampling(sample_tups=sample_tups,
                                                                neg_max_sample=self.neg_max_sample)
                    optm_sc_pair_list += local_picked_tups
            q_optm_pairs_dict[q_idx] = optm_sc_pair_list
            q_evals_dict[q_idx] = eval_sc_list
            total_optm_pair_size += len(optm_sc_pair_list)
            total_eval_sc_size += len(eval_sc_list)

        LogInfo.logs('In total: optm_pairs = %d, eval_sc = %d.', total_optm_pair_size, total_eval_sc_size)
        return q_optm_pairs_dict, q_evals_dict

    def _schema_dedup(self, sc_list):
        """
        Given all candidates of a question, remove duplicate schemas
        :return: [(sc, specific_f1)]
        """
        key_cands_dict = {}
        ret_tup_list = []
        for sc in sc_list:
            key = self._get_rm_key(sc)
            # separate schemas by task-specific key
            key_cands_dict.setdefault(key, []).append(sc)

        for key, cand_list in key_cands_dict.items():
            # All schemas under the same key could be shrink into one candidate
            max_f1 = max([sc.f1 for sc in cand_list])
            for sc in cand_list:
                sc.rm_f1 = max_f1
            # just pick the first schema as representative one
            first_sc = cand_list[0]
            ret_tup_list.append((first_sc, max_f1))
        return ret_tup_list

    def _get_rm_key(self, sc):
        # category, start, end, detail_path
        # Just copy from kq_schema.get_rm_key(), and no need to change "Main" into "Entity"
        rep_list = []
        for raw_path, using_pred_seq in zip(sc.raw_paths, sc.path_list):
            category, gl_data, _ = raw_path
            local_rep = '%s:%s:%s:%s' % (category, gl_data.start, gl_data.end, '|'.join(using_pred_seq))
            rep_list.append(local_rep)
        rep_list.sort()
        return '\t'.join(rep_list)

    def _input_feat_gen(self, sc):
        if sc.input_np_dict is not None:
            return

        qw_input, qw_len = self.feat_gen_helper.generate_qw_feat(sc=sc)

        dep_input, dep_len = self.feat_gen_helper.generate_dep_feat(sc=sc)

        path_size, path_ids, pw_input, pw_len, pseq_ids, pseq_len = \
            self.feat_gen_helper.generate_whole_path_feat(sc=sc)

        sc.input_np_dict = {
            'path_size': path_size,
            'path_ids': path_ids,
            'pw_input': pw_input,
            'pw_len': pw_len,
            'pseq_ids': pseq_ids,
            'pseq_len': pseq_len,

            'qw_input': qw_input,
            'qw_len': qw_len,
            'dep_input': dep_input,
            'dep_len': dep_len
        }

    def _weighted_sampling(self, sample_tups, neg_max_sample, cool_down=1.0):
        """ weighted sampling without replacements """
        if len(sample_tups) == 0:
            return []
        delta = np.array([tup[-1] for tup in sample_tups])
        cd_delta = delta * cool_down
        raw_prob = np.exp(cd_delta)
        prob = raw_prob / np.sum(raw_prob)
        sample_size = neg_max_sample
        pick_idx_list = np.random.choice(a=len(sample_tups),
                                         size=sample_size,
                                         replace=True,
                                         p=prob)
        pick_tups = []
        for idx in pick_idx_list:
            tup = sample_tups[idx]
            pick_tups.append((tup[0], tup[1]))
        return pick_tups
