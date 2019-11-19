"""
Generate candidate query graph for Reddit
"""
import os
import json
import codecs
import shutil
import pickle
import argparse

from src.kbqa.utils.link_data import LinkData
from src.kbqa.utils.log_util import LogInfo
from src.kbqa.dataset.schema import Schema


class RedditCandidateGenerator:
    def __init__(self, freebase_fp, links_fp, verbose=0):
        self.subj_pred_dict = {}
        self.p_links_dict = {}  # <p_idx, [LinkData]>
        # verbose = 0: show basic flow of the process
        # verbose = 1: show detail linking information
        self.verbose = verbose

        self._load_fb_subset(freebase_fp)
        self._load_linkings(links_fp)

    def _load_fb_subset(self, fb_fp):
        LogInfo.begin_track('Loading freebase subset from [%s] ...', fb_fp)
        prefix = 'www.freebase.com/'
        pref_len = len(prefix)

        with codecs.open(fb_fp, 'r', 'utf-8') as br:
            lines = br.readlines()
        LogInfo.logs('%d lines loaded.', len(lines))
        for line_idx, line in enumerate(lines):
            if line_idx % 500000 == 0:
                LogInfo.logs('Current: %d / %d', line_idx, len(lines))
            s, p, _ = line.strip().split('\t')
            s = s[pref_len:].replace('/', '.')
            p = p[pref_len:].replace('/', '.')
            self.subj_pred_dict.setdefault(s, set([])).add(p)
        LogInfo.logs('%d related entities and %d <S, P> pairs saved.',
                     len(self.subj_pred_dict), sum([len(v) for v in self.subj_pred_dict.values()]))
        LogInfo.end_track()

    def _load_linkings(self, links_fp):
        with codecs.open(links_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                if line.startswith('#'):
                    continue
                spt = line.strip().split('\t')
                q_idx, st, ed, mention, mid, wiki_name, feats = spt
                q_idx = int(q_idx)
                st = int(st)
                ed = int(ed)
                feat_dict = json.loads(feats)
                for k in feat_dict:
                    v = float('%.6f' % feat_dict[k])
                    feat_dict[k] = v
                link_data = LinkData(category='Entity',
                                     start=st, end=ed,
                                     mention=mention, comp='==',
                                     value=mid, name=wiki_name,
                                     link_feat=feat_dict)
                self.p_links_dict.setdefault(q_idx, []).append(link_data)
        LogInfo.logs('%d questions of link data loaded.', len(self.p_links_dict))

    def single_post_candgen(self, p_idx, post, link_fp, schema_fp):
        # =================== Linking first ==================== #
        if os.path.isfile(link_fp):
            gather_linkings = []
            with codecs.open(link_fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    tup_list = json.loads(line.strip())
                    ld_dict = {k: v for k, v in tup_list}
                    gather_linkings.append(LinkData(**ld_dict))
        else:
            gather_linkings = self.p_links_dict.get(p_idx, [])
            for idx in range(len(gather_linkings)):
                gather_linkings[idx].gl_pos = idx
        # ==================== Save linking results ================ #
        if not os.path.isfile(link_fp):
            with codecs.open(link_fp + '.tmp', 'w', 'utf-8') as bw:
                for gl in gather_linkings:
                    bw.write(json.dumps(gl.serialize()) + '\n')
            shutil.move(link_fp + '.tmp', link_fp)
        # ===================== simple predicate finding ===================== #
        sc_list = []
        for gl_data in gather_linkings:
            entity = gl_data.value
            pred_set = self.subj_pred_dict.get(entity, set([]))
            for pred in pred_set:
                sc = Schema()
                sc.hops = 1
                sc.main_pred_seq = [pred]
                sc.raw_paths = [('Main', gl_data, [pred])]
                sc.ans_size = 1
                sc_list.append(sc)
        if len(sc_list) == 0:
            LogInfo.logs("=============q_idx: %d sc_list=0======================" % p_idx)
        # ==================== Save schema results ================ #
        # ans_size, hops, raw_paths
        # raw_paths: (category, gl_pos, gl_mid, pred_seq)
        with codecs.open(schema_fp + '.tmp', 'w', 'utf-8') as bw:
            for sc in sc_list:
                sc_info_dict = {k: getattr(sc, k) for k in ('ans_size', 'hops')}
                opt_raw_paths = []
                for cate, gl, pred_seq in sc.raw_paths:
                    opt_raw_paths.append((cate, gl.gl_pos, gl.value, pred_seq))
                sc_info_dict['raw_paths'] = opt_raw_paths
                bw.write(json.dumps(sc_info_dict) + '\n')
        shutil.move(schema_fp + '.tmp', schema_fp)


def main(args):
    data_path = "%s/Reddit.%s.pkl" % (args.data_dir, args.mode)
    freebase_path = "%s/freebase-FB2M.txt" % args.freebase_dir
    links_path = "%s/Reddit.%s.links" % (args.data_dir, args.mode)

    with open(data_path, 'rb') as br:
        dg_list = pickle.load(br)
    LogInfo.logs('%d Reddit dialogs loaded.' % len(dg_list))

    cand_gen = RedditCandidateGenerator(freebase_fp=freebase_path, links_fp=links_path,
                                        verbose=args.verbose)
    
    output_dir = args.output_prefix + "_%s" % args.mode
    all_list_fp = output_dir + '/all_list'
    all_lists = []
    for p_idx, post in enumerate(dg_list):
        LogInfo.begin_track('Entering P %d / %d:', p_idx, len(dg_list))
        sub_idx = int(p_idx / 10000) * 10000
        index = 'data/%d-%d/%d_schema' % (sub_idx, sub_idx + 9999, p_idx)
        all_lists.append(index)
        sub_dir = '%s/data/%d-%d' % (output_dir, sub_idx, sub_idx + 9999)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        schema_fp = '%s/%d_schema' % (sub_dir, p_idx)
        link_fp = '%s/%d_links' % (sub_dir, p_idx)
        if os.path.isfile(schema_fp):
            LogInfo.end_track('Skip this post, already saved.')
            continue

        cand_gen.single_post_candgen(p_idx=p_idx, post=post,
                                     link_fp=link_fp, schema_fp=schema_fp)
        LogInfo.end_track()
    with open(all_list_fp, 'w') as fw:
        for i, idx_str in enumerate(all_lists):
            if i == len(all_lists)-1:
                fw.write(idx_str)
            else:
                fw.write(idx_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reddit candidates generation")
    parser.add_argument('--data_dir', type=str, help="Reddit data directory")
    parser.add_argument('--mode', type=str, choices=['train', 'valid', 'test'])
    parser.add_argument('--freebase_dir', type=str, help="Freebase subset directory")
    parser.add_argument('--output_prefix', type=str, help="Output candidates directory prefix")
    parser.add_argument('--verbose', type=int, default=0)
    parsed_args = parser.parse_args()

    main(parsed_args)
