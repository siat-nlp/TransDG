# -*- coding:utf-8 -*-
"""
Implementation of entity linking
Goal: Entity/Type/Time linking based on n-grams.
Note: all linking procedure ignore cases.
"""
import re
import codecs
import json
import pickle
import argparse
from stanfordcorenlp import StanfordCoreNLP
from src.kbqa.utils.log_util import LogInfo

# set StandfordCoreNLP lib path
CORENLP_PATH = "/lib/stanford-corenlp-full-2018-02-27/"
punc_mask_str = "?!:', "
year_re = re.compile(r'^[1-2][0-9][0-9][0-9]$')


class LukovLinkTuple:

    def __init__(self, category, start, end, mention, mid, name, feat_dict):
        self.category = category  # category: this mid is an Entity or Type
        self.start = start  # start: the starting token index of the mention
        self.end = end   # end:  the end token index of the mention
        self.mention = mention  # mention: the mention surface in the question
        self.mid = mid  # mid: the linked entity mid
        self.name = name  # name: corresponding type.object.name
        self.feat_dict = feat_dict  # feat_dict: feature dict of linking


class LukovLinker:
    def __init__(self, freebase_fp, mid_name_fp, type_name_fp, pred_name_fp,
                 entity_pop_fp, type_pop_fp, allow_alias=False):
        self.subj_pred_keys = set()
        self.surface_mid_dict = {}  # <surface, set([mid])>
        self.mid_name_dict = {}  # <mid, type.object.name>
        self.type_set = set([])
        self.pred_set = set([])
        self.pop_dict = {}  # <mid, popularity>
        self.skip_domain_set = {'common', 'type', 'user', 'base', 'freebase', 'g'}

        self._load_fb_subset(freebase_fp)
        self._load_mid(mid_name_fp, allow_alias=allow_alias)
        self._load_type(type_name_fp)
        self._load_pred(pred_name_fp)
        self._load_pop_dict(entity_pop_fp, type_pop_fp)

    def _load_fb_subset(self, freebase_fp):
        LogInfo.begin_track('Loading freebase subset from [%s] ...', freebase_fp)
        prefix = 'www.freebase.com/'
        pref_len = len(prefix)
        with codecs.open(freebase_fp, 'r', 'utf-8') as br:
            lines = br.readlines()
        LogInfo.logs('%d lines loaded.', len(lines))
        for line_idx, line in enumerate(lines):
            if line_idx > 0 and line_idx % 500000 == 0:
                LogInfo.logs('Current: %d / %d', line_idx, len(lines))
            s, p, _ = line.strip().split('\t')
            s = s[pref_len:].replace('/', '.')
            self.subj_pred_keys.add(s)
        LogInfo.logs('%d related entities loaded.', len(self.subj_pred_keys))
        LogInfo.end_track()

    def _load_mid(self, mid_name_fp, allow_alias=False):
        LogInfo.begin_track('Loading surface --> mid dictionary from [%s] ...', mid_name_fp)
        with codecs.open(mid_name_fp, 'r', 'utf-8') as br:
            scan = 0
            while True:
                line = br.readline()
                if line is None or line == '':
                    break
                spt = line.strip().split('\t')
                if len(spt) < 3:
                    continue
                mid = spt[0]
                name = spt[2]
                surface = name.lower()      # save lowercase as searching entrance
                skip = False                # ignore some subjects at certain domain
                mid_prefix_pos = mid.find('.')
                if mid_prefix_pos == -1:
                    skip = True
                else:
                    mid_prefix = mid[: mid_prefix_pos]
                    if mid_prefix in self.skip_domain_set:
                        skip = True
                if not skip:
                    if spt[1] == 'type.object.name':
                        self.mid_name_dict[mid] = name
                    if spt[1] == 'type.object.name' or allow_alias:
                        self.surface_mid_dict.setdefault(surface, set([])).add(mid)
                scan += 1
                if scan % 100000 == 0:
                    LogInfo.logs('%d lines scanned.', scan)
        LogInfo.logs('%d lines scanned.', scan)
        LogInfo.logs('%d <surface, mid_set> loaded.', len(self.surface_mid_dict))
        LogInfo.logs('%d <mid, name> loaded.', len(self.mid_name_dict))
        LogInfo.end_track()

    def _load_type(self, type_name_fp):
        with codecs.open(type_name_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                type_mid, type_name = spt[0], spt[1]
                surface = type_name.lower().replace('(s)', '')
                type_prefix = type_mid[: type_mid.find('.')]
                if type_prefix not in self.skip_domain_set:
                    self.surface_mid_dict.setdefault(surface, set([])).add(type_mid)
                    self.mid_name_dict[type_mid] = type_name
                    self.type_set.add(type_mid)
        LogInfo.logs('After scanning %d types, %d <surface, mid_set> loaded.',
                     len(self.type_set), len(self.surface_mid_dict))

    def _load_pred(self, pred_name_fp):
        with codecs.open(pred_name_fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                self.pred_set.add(spt[0])
        LogInfo.logs('%d predicates scanned.', len(self.pred_set))

    def _load_pop_dict(self, entity_pop_fp, type_pop_fp):
        for pop_fp in [entity_pop_fp, type_pop_fp]:
            LogInfo.logs('Reading popularity from %s ...', pop_fp)
            with codecs.open(pop_fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    spt = line.strip().split('\t')
                    self.pop_dict[spt[0]] = int(spt[1])
        LogInfo.logs('%d <mid, popularity> loaded.', len(self.pop_dict))

    def _lowercased_surface(self, token_list, st, ed):
        """
        Construct the surface form in [st, ed)
        Be careful: Their may have or may not have a blank before the token starting with a punctuation
                    For example, "'s" is a token starting with a punctuation.
                    It's hard to say whether we shall directly connect two the last token, or adding a blank
                    But we can enumerate each possibility.
        Thus, the return value is a list of possible surfaces.
        """
        surface_list = ['']
        for idx in range(st, ed):
            tok = token_list[idx].lower()
            if idx == st:  # must not add blank
                new_surface_list = [x + tok for x in surface_list]
            elif idx > st and tok[0] not in punc_mask_str:  # must add blank
                new_surface_list = [x + ' ' + tok for x in surface_list]
            else:  # both add or not add are possible
                tmp1_list = [x + tok for x in surface_list]
                tmp2_list = [x + ' ' + tok for x in surface_list]
                new_surface_list = tmp1_list + tmp2_list
            surface_list = new_surface_list
        return surface_list

    def link_single_question(self, q_tokens):
        link_tups = []
        token_list = q_tokens
        token_len = len(token_list)
        for i in range(token_len):
            for j in range(i+1, token_len):
                std_surface = ' '.join([tok.lower() for tok in token_list[i: j]])
                # LogInfo.begin_track('std_surface: %s', std_surface)
                lower_surface_list = self._lowercased_surface(token_list, i, j)

                match_set = set([])
                if re.match(year_re, std_surface):
                    match_set.add(std_surface)      # time value
                for surf in lower_surface_list:
                    match_set |= self.surface_mid_dict.get(surf, set([]))
                for match_mid in match_set:
                    if match_mid in self.pred_set:
                        continue        # won't match a mention into predicates
                    if re.match(year_re, match_mid):
                        mid_name = match_mid
                        pop = 1000      # set to a rather high value
                        category = 'Time'
                    else:
                        mid_name = self.mid_name_dict[match_mid]
                        pop = self.pop_dict.get(match_mid, 1)
                        category = 'Type' if match_mid in self.type_set else 'Entity'
                    # LogInfo.logs('match_mid: %s, name: %s', match_mid, mid_name)
                    feat_dict = {'pop': pop}
                    tup = LukovLinkTuple(category=category,
                                         start=i, end=j,
                                         mention=std_surface,
                                         mid=match_mid, name=mid_name,
                                         feat_dict=feat_dict)
                    link_tups.append(tup)
                # LogInfo.end_track()
        if len(link_tups) == 0:
            print("link_tups = 0")
            print("q:", q_tokens)
            feat_dict = {'pop': 1}
            tup = LukovLinkTuple(category='Entity',
                                 start=0, end=1,
                                 mention='a',
                                 mid='m.02p6x0', name='A',
                                 feat_dict=feat_dict)
            link_tups.append(tup)

        link_tups.sort(key=lambda x: x.feat_dict['pop'], reverse=True)
        final_tup = link_tups[0]
        is_in_fb = False
        for tup in link_tups:
            if tup.mid in self.subj_pred_keys:
                final_tup = tup
                is_in_fb = True
                break
        if not is_in_fb:
            feat_dict = {'pop': 1}
            tup = LukovLinkTuple(category='Entity',
                                 start=0, end=1,
                                 mention='a',
                                 mid='m.02p6x0', name='A',
                                 feat_dict=feat_dict)
            final_tup = tup

        return final_tup


def load_simpq(data_dir):
    LogInfo.logs('SimpQ initializing ... ')
    qa_list = []
    corenlp = StanfordCoreNLP(CORENLP_PATH)
    for Tvt in ('train', 'valid', 'test'):
        fp = '%s/annotated_fb_data_%s.txt' % (data_dir, Tvt)
        with codecs.open(fp, 'r', 'utf-8') as br:
            for line in br.readlines():
                qa = {}
                s, p, o, q = line.strip().split('\t')
                s = _remove_simpq_header(s)
                p = _remove_simpq_header(p)
                o = _remove_simpq_header(o)
                qa['utterance'] = q
                qa['targetValue'] = (s, p, o)  # different from other datasets
                qa['tokens'] = corenlp.word_tokenize(qa['utterance'])
                qa['parse'] = corenlp.dependency_parse(qa['utterance'])
                qa_list.append(qa)
                if len(qa_list) % 1000 == 0:
                    LogInfo.logs('%d scanned.', len(qa_list))
    pickle_fp = '%s/simpQ.data.pkl' % data_dir
    with open(pickle_fp, 'wb') as bw:
        pickle.dump(qa_list, bw)
    LogInfo.logs('%d SimpleQuestions loaded.' % len(qa_list))
    return qa_list


def load_reddit(data_dir, mode='train'):
    LogInfo.logs('Reddit initializing ... ')
    dg_list = []
    corenlp = StanfordCoreNLP(CORENLP_PATH)
    fp = '%s/%s_v3.txt' % (data_dir, mode)
    with open(fp, 'r') as br:
        for line in br:
            dg_line = json.loads(line)
            dialog = {'utterance': dg_line['post'].strip(),
                      'tokens': dg_line['post'].split(),
                      'parse': corenlp.dependency_parse(dg_line['post']),
                      'response': dg_line['response'].strip(),
                      'corr_responses': dg_line['corr_responses'],
                      'all_triples': dg_line['all_triples'],
                      'all_entities': dg_line['all_entities']
                      }

            dg_list.append(dialog)
            if len(dg_list) % 10000 == 0:
                LogInfo.logs('%d scanned.', len(dg_list))
    pickle_fp = '%s/Reddit.%s.pkl' % (data_dir, mode)
    with open(pickle_fp, 'wb') as bw:
        pickle.dump(dg_list, bw)
    LogInfo.logs('%d Reddit saved in [%s].' % (len(dg_list), pickle_fp))
    return dg_list


def _remove_simpq_header(mid):
    mid = mid[17:]  # remove 'www.freebase.com/'
    mid = mid.replace('/', '.')
    return mid


def main(args):
    if args.data_name == "SimpQ":
        qa_list = load_simpq(args.data_dir)
        output_file = "%s/SimpQ.all.links" % args.data_dir
    else:
        qa_list = load_reddit(args.data_dir, mode=args.mode)
        output_file = "%s/Reddit.%s.links" % (args.data_dir, args.mode)

    freebase_path = "%s/freebase-FB2M.txt" % args.fb_dir
    mid_name_path = "%s/S-NAP-ENO-triple.txt" % args.fb_meta_dir
    type_name_path = "%s/TS-name.txt" % args.fb_meta_dir
    pred_name_path = "%s/PS-name.txt" % args.fb_meta_dir
    entity_pop_path = "%s/entity_pop_5m.txt" % args.fb_meta_dir
    type_pop_path = "%s/type_pop.txt" % args.fb_meta_dir

    linker = LukovLinker(freebase_fp=freebase_path, mid_name_fp=mid_name_path, type_name_fp=type_name_path,
                         pred_name_fp=pred_name_path, entity_pop_fp=entity_pop_path, type_pop_fp=type_pop_path)

    LogInfo.begin_track('Linking data save to: %s' % output_file)
    with codecs.open(output_file, 'w', 'utf-8') as bw:
        for q_idx, qa in enumerate(qa_list):
            q_tokens = qa['tokens']
            if q_idx > 0 and q_idx % 10000 == 0:
                LogInfo.logs('Entering Q-%d', q_idx)
            tup = linker.link_single_question(q_tokens)
            bw.write('%04d\t%d\t%d\t%s\t%s\t%s\t%s\n' % (
                q_idx, tup.start, tup.end, tup.mention, tup.mid, tup.name, json.dumps(tup.feat_dict)))
    LogInfo.end_track()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entity linking generation")
    parser.add_argument('--data_name', type=str, choices=['SimpQ', 'Reddit'])
    parser.add_argument('--mode', type=str, choices=['train', 'valid', 'test'])
    parser.add_argument('--data_dir', type=str, help="data directory")
    parser.add_argument('--fb_dir', type=str, help="freebase subset directory")
    parser.add_argument('--fb_meta_dir', type=str, help="freebase metadata directory")
    parsed_args = parser.parse_args()

    main(parsed_args)
