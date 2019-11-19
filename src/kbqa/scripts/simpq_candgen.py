"""
Generate candidate query graph for SimpQ
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


class SimpleQCandidateGenerator:
    def __init__(self, freebase_fp, links_fp, verbose=0):
        self.subj_pred_dict = {}
        self.q_links_dict = {}  # <q_idx, [LinkData]>
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
                self.q_links_dict.setdefault(q_idx, []).append(link_data)
        LogInfo.logs('%d questions of link data loaded.', len(self.q_links_dict))

    def single_question_candgen(self, q_idx, qa, link_fp, schema_fp):
        # =================== Linking first ==================== #
        if os.path.isfile(link_fp):
            gather_linkings = []
            with codecs.open(link_fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    tup_list = json.loads(line.strip())
                    ld_dict = {k: v for k, v in tup_list}
                    gather_linkings.append(LinkData(**ld_dict))
            LogInfo.logs('Read %d links from file.', len(gather_linkings))
        else:
            gather_linkings = self.q_links_dict.get(q_idx, [])
            for idx in range(len(gather_linkings)):
                gather_linkings[idx].gl_pos = idx

        LogInfo.begin_track('Show %d E links :', len(gather_linkings))
        if self.verbose >= 1:
            for gl in gather_linkings:
                LogInfo.logs(gl.display())
        LogInfo.end_track()
        # ==================== Save linking results ================ #
        if not os.path.isfile(link_fp):
            with codecs.open(link_fp + '.tmp', 'w', 'utf-8') as bw:
                for gl in gather_linkings:
                    bw.write(json.dumps(gl.serialize()) + '\n')
            shutil.move(link_fp + '.tmp', link_fp)
            LogInfo.logs('%d link data save to file.', len(gather_linkings))
        # ===================== simple predicate finding ===================== #
        gold_entity, gold_pred, _ = qa['targetValue']
        sc_list = []
        for gl_data in gather_linkings:
            entity = gl_data.value
            pred_set = self.subj_pred_dict.get(entity, set([]))
            for pred in pred_set:
                sc = Schema()
                sc.hops = 1
                sc.aggregate = False
                sc.main_pred_seq = [pred]
                sc.raw_paths = [('Main', gl_data, [pred])]
                sc.ans_size = 1
                if entity == gold_entity and pred == gold_pred:
                    sc.f1 = sc.p = sc.r = 1.
                else:
                    sc.f1 = sc.p = sc.r = 0.
                sc_list.append(sc)
        # ==================== Save schema results ================ #
        # p, r, f1, ans_size, hops, raw_paths, (agg)
        # raw_paths: (category, gl_pos, gl_mid, pred_seq)
        with codecs.open(schema_fp + '.tmp', 'w', 'utf-8') as bw:
            for sc in sc_list:
                sc_info_dict = {k: getattr(sc, k) for k in ('p', 'r', 'f1', 'ans_size', 'hops')}
                if sc.aggregate is not None:
                    sc_info_dict['agg'] = sc.aggregate
                opt_raw_paths = []
                for cate, gl, pred_seq in sc.raw_paths:
                    opt_raw_paths.append((cate, gl.gl_pos, gl.value, pred_seq))
                sc_info_dict['raw_paths'] = opt_raw_paths
                bw.write(json.dumps(sc_info_dict) + '\n')
        shutil.move(schema_fp + '.tmp', schema_fp)
        LogInfo.logs('%d schemas successfully saved into [%s].', len(sc_list), schema_fp)


def main(args):
    data_path = "%s/simpQ.data.pkl" % args.data_dir
    freebase_path = "%s/freebase-FB2M.txt" % args.freebase_dir
    links_path = "%s/SimpQ.all.links" % args.data_dir
    
    with open(data_path, 'rb') as br:
        qa_list = pickle.load(br)
    LogInfo.logs('%d SimpleQuestions loaded.' % len(qa_list))
    
    cand_gen = SimpleQCandidateGenerator(freebase_fp=freebase_path, links_fp=links_path,
                                         verbose=args.verbose)

    all_list_fp = args.output_dir + '/all_list'
    all_lists = []
    for q_idx, qa in enumerate(qa_list):
        LogInfo.begin_track('Entering Q %d / %d [%s]:',
                            q_idx, len(qa_list), qa['utterance'])
        sub_idx = int(q_idx / 1000) * 1000
        index = 'data/%d-%d/%d_schema' % (sub_idx, sub_idx + 999, q_idx)
        all_lists.append(index)
        sub_dir = '%s/data/%d-%d' % (args.output_dir, sub_idx, sub_idx + 999)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        schema_fp = '%s/%d_schema' % (sub_dir, q_idx)
        link_fp = '%s/%d_links' % (sub_dir, q_idx)
        if os.path.isfile(schema_fp):
            LogInfo.end_track('Skip this question, already saved.')
            continue

        cand_gen.single_question_candgen(q_idx=q_idx, qa=qa,
                                         link_fp=link_fp, schema_fp=schema_fp)
        LogInfo.end_track()
    with open(all_list_fp, 'w') as fw:
        for i, idx_str in enumerate(all_lists):
            if i == len(all_lists)-1:
                fw.write(idx_str)
            else:
                fw.write(idx_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimpQ candidates generation")
    parser.add_argument('--data_dir', type=str, help="SimpQ data directory")
    parser.add_argument('--freebase_dir', type=str, help="Freebase subset directory")
    parser.add_argument('--output_dir', type=str, help="Output candidates directory")
    parser.add_argument('--verbose', type=int, default=0)
    parsed_args = parser.parse_args()

    main(parsed_args)

