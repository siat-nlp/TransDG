# -*- coding: utf-8 -*-
import os
import pickle
import argparse
import numpy as np
from src.kbqa.dataset.feature_helper import FeatureHelper


CHUNK_SIZE = 5000


class DialogueDataset:

    def __init__(self, data_dir, candgen_dir, mode='train'):

        self.data_dir = data_dir
        self.candgen_dir = candgen_dir
        self.mode = mode

        self.q_idx_list = []
        self.dialog_list = []
        self.q_cand_dict = {}
        self.csk_triples = []
        self.csk_entities = []

    def load_all_data(self):
        self.load_dialog_list(mode=self.mode)
        self.load_schemas()
        self.load_csk_triples()
        self.load_csk_entities()

        print('Meta statistics:')
        print('Total posts = %d' % len(self.q_idx_list))
        cand_size_dist = np.array([len(v) for v in self.q_cand_dict.values()])
        print('Total schemas = %d, avg = %.3f.' % (np.sum(cand_size_dist), np.mean(cand_size_dist)))
        qlen_dist = np.array([len(qa['tokens']) for qa in self.dialog_list])
        print('Avg post length = %.3f.' % np.mean(qlen_dist))

    def load_dialog_list(self, mode='train'):
        pickle_fp = '%s/Reddit.%s.pkl' % (self.data_dir, mode)
        print('Loading Reddit dialogs from [%s] ...' % pickle_fp)
        with open(pickle_fp, 'rb') as br:
            self.dialog_list = pickle.load(br)
        print('%d dialogs loaded.' % len(self.dialog_list))

    def load_schemas(self):
        print('Loading schemas from [%s] ...' % self.candgen_dir)
        with open("%s/q_idx.pkl" % self.candgen_dir, 'rb') as br:
            self.q_idx_list = pickle.load(br)
        with open("%s/q_cand.pkl" % self.candgen_dir, 'rb') as br:
            self.q_cand_dict = pickle.load(br)

    def load_csk_triples(self):
        kb_triple_fp = "%s/csk_triples.txt" % self.data_dir
        print('Loading commonsense triples from [%s]' % kb_triple_fp)
        with open(kb_triple_fp, 'r') as fr:
            for idx, triple in enumerate(fr):
                self.csk_triples.append(triple.strip())
        print("%d triples loaded" % len(self.csk_triples))

    def load_csk_entities(self):
        kb_entity_fp = "%s/csk_entities.txt" % self.data_dir
        print('Loading commonsense entities from [%s]' % kb_entity_fp)
        with open(kb_entity_fp, 'r') as fr:
            for idx, entity in enumerate(fr):
                self.csk_entities.append(entity.strip())
        print("%d entities loaded" % len(self.csk_entities))


class DialogueBuilder:

    def __init__(self, schema_dataset, feature_helper,
                 max_post_len=30, max_resp_len=30, max_triple_num=8, max_triple_len=20):
        print('SchemaBuilder initializing ...')
        self.mode = schema_dataset.mode
        self.q_idx_list = schema_dataset.q_idx_list
        self.dialog_list = schema_dataset.dialog_list
        self.q_cand_dict = schema_dataset.q_cand_dict
        self.csk_triples = schema_dataset.csk_triples
        self.csk_entities = schema_dataset.csk_entities
        self.feat_gen_helper = feature_helper

        self.max_post_len = max_post_len
        self.max_resp_len = max_resp_len
        self.max_triple_num = max_triple_num
        self.max_triple_len = max_triple_len

    def show_statistic(self):
        posts = [len(self.dialog_list[idx]['utterance'].split()) for idx in self.q_idx_list]
        resps = [len(self.dialog_list[idx]['response'].split()) for idx in self.q_idx_list]
        triple_nums = [len(self.dialog_list[idx]['all_triples']) for idx in self.q_idx_list]
        triple_lens = [len(triple) for idx in self.q_idx_list for triple in self.dialog_list[idx]['all_triples']]
        entity_nums = [len(self.dialog_list[idx]['all_entities']) for idx in self.q_idx_list]
        entity_lens = [len(ent) for idx in self.q_idx_list for ent in self.dialog_list[idx]['all_entities']]

        max_post_len = max(posts)
        max_resp_len = max(resps)
        max_triple_num = max(triple_nums)
        max_triple_len = max(triple_lens)
        max_ent_num = max(entity_nums)
        max_ent_len = max(entity_lens)

        avg_post_len = np.mean(posts)
        avg_resp_len = np.mean(resps)
        avg_triple_num = np.mean(triple_nums)
        avg_triple_len = np.mean(triple_lens)
        avg_ent_num = np.mean(entity_nums)
        avg_ent_len = np.mean(entity_lens)

        print("Max post length = %d" % max_post_len)  # 64
        print("Max response length = %d" % max_resp_len)  # 60
        print("Max triple num = %d" % max_triple_num)  # 23
        print("Max triple length = %d" % max_triple_len)  # 25
        print("Max entity num = %d" % max_ent_num)
        print("Max entity length = %d" % max_ent_len)

        print("Avg post length = %.3f" % avg_post_len)  # 20
        print("Avg response length = %.3f" % avg_resp_len)  # 20
        print("Avg triple num = %.3f" % avg_triple_num)  # 5
        print("Avg triple length = %.3f" % avg_triple_len)  # 18
        print("Avg entity num = %.3f" % avg_ent_num)
        print("Avg entity length = %.3f" % avg_ent_len)

        return max_post_len, max_resp_len

    def save_to_pickle(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if self.mode == 'train':
            q_dict = {}
            chunk_idx = 0
            chunk_list = []
            for scan_idx, q_idx in enumerate(self.q_idx_list):
                cand_list = self.q_cand_dict[q_idx]
                schema = cand_list[0]  # we just pick the first schema

                # Generate input features here
                input_np_dict = self._input_feat_gen(schema, max_post_len=self.max_post_len,
                                                     max_resp_len=self.max_resp_len,
                                                     max_triple_num=self.max_triple_num,
                                                     max_triple_len=self.max_triple_len)
                q_dict[q_idx] = input_np_dict
                if scan_idx+1 > 0 and (scan_idx+1) % CHUNK_SIZE == 0:
                    print('scanned %d / %d questions.' % (scan_idx+1, len(self.q_idx_list)))
                    # save to chunk file
                    chunk_idx += 1
                    chunk_file = "train.%03d.pkl" % chunk_idx
                    chunk_list.append(chunk_file)
                    save_path = "%s/%s" % (save_dir, chunk_file)
                    with open(save_path, 'wb') as bw:
                        pickle.dump(q_dict, bw)
                    print("Saved data to pickle [%s]" % save_path)
                    q_dict = {}
            # remain data save to chunk file
            chunk_idx += 1
            chunk_file = "train.%03d.pkl" % chunk_idx
            chunk_list.append(chunk_file)
            save_path = "%s/%s" % (save_dir, chunk_file)
            with open(save_path, 'wb') as bw:
                pickle.dump(q_dict, bw)
            print("Saved data to pickle [%s]" % save_path)

            # save index list
            index_save_path = "%s/all_list" % save_dir
            with open(index_save_path, 'w') as fw:
                for idx in chunk_list:
                    fw.write(idx + '\n')
            print("Index saved to [%s]" % index_save_path)
        else:
            # TODO: use ground-truth length for valid/test data
            # max_post_len, max_resp_len = self.show_statistic()
            q_dict = {}
            total_size = 0
            for scan_idx, q_idx in enumerate(self.q_idx_list):
                cand_list = self.q_cand_dict[q_idx]
                schema = cand_list[0]  # we just pick the first schema

                # Generate input features here
                input_np_dict = self._input_feat_gen(schema, max_post_len=self.max_post_len,
                                                     max_resp_len=self.max_resp_len,
                                                     max_triple_num=self.max_triple_num,
                                                     max_triple_len=self.max_triple_len)
                q_dict[q_idx] = input_np_dict
                total_size += 1
            print('In total: data size = %d.' % total_size)
            save_path = "%s/%s.pkl" % (save_dir, self.mode)
            with open(save_path, 'wb') as bw:
                pickle.dump(q_dict, bw)
            print("Saved data to pickle [%s]" % save_path)

    def _input_feat_gen(self, schema, max_post_len, max_resp_len, max_triple_num, max_triple_len):
        q_idx = schema.q_idx
        dialog = self.dialog_list[q_idx]
        post_tokens = dialog['utterance'].split()
        post_len, post = self._pad_sent(post_tokens, max_post_len)

        response_tokens = dialog['response'].split()
        response_len, response = self._pad_sent(response_tokens, max_resp_len)

        corr_responses = dialog['corr_responses']
        pad_corr_responses = []
        for res in corr_responses:
            tokens = res.split()
            _, token_pad = self._pad_sent(tokens, max_resp_len)
            pad_corr_responses.append(token_pad)

        all_triple_ids = dialog['all_triples']
        pad_all_triples = self._pad_triple(
            [[self.csk_triples[x].split(', ') for x in triple] for triple in all_triple_ids],
            max_triple_num, max_triple_len
        )

        all_entity_ids = dialog['all_entities']
        pad_all_entities = self._pad_entity(
            [[self.csk_entities[x] for x in entity] for entity in all_entity_ids],
            max_triple_num, max_triple_len
        )

        qw_input, qw_len = self.feat_gen_helper.generate_qw_feat(sc=schema)
        dep_input, dep_len = self.feat_gen_helper.generate_dep_feat(sc=schema)

        if self.mode == 'test':
            entities = []
            for ent_id_list in all_entity_ids:
                for ent_id in ent_id_list:
                    entities.append(self.csk_entities[ent_id])

            input_np_dict = {
                'post': post,
                'post_len': post_len,
                'response': response,
                'response_len': response_len,
                'corr_responses': pad_corr_responses,
                'all_triples': pad_all_triples,
                'all_entities': pad_all_entities,
                'entities': entities,
                'qw_input': qw_input,
                'qw_len': qw_len,
                'dep_input': dep_input,
                'dep_len': dep_len
            }
        else:
            input_np_dict = {
                'post': post,
                'post_len': post_len,
                'response': response,
                'response_len': response_len,
                'corr_responses': pad_corr_responses,
                'all_triples': pad_all_triples,
                'all_entities': pad_all_entities,
                'qw_input': qw_input,
                'qw_len': qw_len,
                'dep_input': dep_input,
                'dep_len': dep_len
            }
        return input_np_dict

    @staticmethod
    def _pad_sent(s, max_len):
        if len(s) >= max_len - 1:
            sentence = s[:max_len - 1] + ['_EOS']
            sent_len = max_len

        else:
            sentence = s + ['_EOS'] + ['_PAD'] * (max_len - len(s) - 1)
            sent_len = len(s) + 1
        return sent_len, sentence

    @staticmethod
    def _pad_triple(triple, max_num, max_len):
        new_triple = []
        for tri in triple:
            if len(tri) >= max_len:
                new_triple.append(tri[: max_len])
            else:
                new_triple.append(tri + [['_PAD_H', '_PAD_R', '_PAD_T']] * (max_len - len(tri)))
        pad_triple = [['_PAD_H', '_PAD_R', '_PAD_T']] * max_len
        if len(new_triple) >= max_num:
            final_triple = new_triple[: max_num]
        else:
            final_triple = new_triple + [pad_triple] * (max_num - len(new_triple))
        return final_triple

    @staticmethod
    def _pad_entity(ents, max_num, max_len):
        new_ent = []
        for ent in ents:
            if len(ent) >= max_len:
                new_ent.append(ent[: max_len])
            else:
                new_ent.append(ent + ['_PAD'] * (max_len - len(ent)))

        pad_entity = ['_PAD'] * max_len
        if len(new_ent) >= max_num:
            final_entity = new_ent[: max_num]
        else:
            final_entity = new_ent + [pad_entity] * (max_num - len(new_ent))
        return final_entity


def main(args):
    dataset = DialogueDataset(args.data_dir, args.candgen_dir, mode=args.mode)
    dataset.load_all_data()
    dialog_list = dataset.dialog_list
    with open(args.dict_path, 'rb') as br:
        active_dicts = pickle.load(br)

    feature_helper = FeatureHelper(active_dicts, dialog_list, freebase_helper=None, path_max_size=args.path_max_size,
                                   qw_max_len=args.qw_max_len, pw_max_len=args.pw_max_len,
                                   pseq_max_len=args.pseq_max_len)
    ds_builder = DialogueBuilder(dataset, feature_helper)
    ds_builder.save_to_pickle(save_dir=args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument('--mode', type=str, choices=['train', 'valid', 'test'])
    parser.add_argument('--data_dir', type=str, help="Reddit data directory")
    parser.add_argument('--candgen_dir', type=str, help="Reddit candidates directory")
    parser.add_argument('--dict_path', type=str, help='word/mid/path dict path')
    parser.add_argument('--kb_triple', type=str, help='commonsense kb triple path')
    parser.add_argument('--qw_max_len', type=int, default=20, help='max length of question')
    parser.add_argument('--pw_max_len', type=int, default=8, help='max length of path at word level')
    parser.add_argument('--path_max_size', type=int, default=3, help='max size of path')
    parser.add_argument('--pseq_max_len', type=int, default=3, help='max length of path at sequence level')
    parser.add_argument('--save_dir', type=str, help="Output directory for final data")
    parsed_args = parser.parse_args()

    main(parsed_args)
