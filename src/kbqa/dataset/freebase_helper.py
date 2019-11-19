# -*- coding:utf-8 -*-
import re
import codecs
from ..utils.log_util import LogInfo


class FreebaseHelper:
    def __init__(self, meta_dir):
        self.meta_dir = meta_dir

        self.type_pred_dict = {}  # <type, set([pred])>
        self.type_name_dict = {}
        self.sup_type_dict = {}
        self.sub_type_dict = {}
        self.med_type_set = set([])
        self.pred_domain_dict = {}
        self.pred_range_dict = {}
        self.pred_name_dict = {}
        self.pred_inverse_dict = {}
        self.time_pred_dict = {}  # <pred, (target_direction, target_pred)>
        self.entity_name_dict = {}

        self.ordinal_type_set = {'type.int', 'type.float', 'type.datetime'}
        self.ignore_type_domain_set = {'base', 'common', 'freebase', 'm', 'type', 'user'}
        self.ignore_pred_domain_set = {'common', 'freebase', 'm', 'type'}

        self._load_pred_name()
        self._load_type_name()
        self._load_domain_range()
        self._load_mediator()
        self._load_sup_sub_types()

    def _load_pred_name(self):
        with codecs.open('%s/PS-name.txt' % self.meta_dir, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                pred, raw_name = spt
                pred_name = adjust_name(raw_name)
                tp = pred[:pred.rfind('.')]
                self.type_pred_dict.setdefault(tp, set([])).add(pred)
                self.pred_name_dict[pred] = pred_name
        LogInfo.logs('FBHelper: %d predicate names loaded.', len(self.pred_name_dict))

    def _load_type_name(self):
        with codecs.open('%s/TS-name.txt' % self.meta_dir, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                if len(spt) < 2:
                    continue
                tp, raw_name = spt
                tp_name = adjust_name(raw_name)  # lowercased, remove punc, remove (s)
                self.type_name_dict[tp] = tp_name
        LogInfo.logs('FBHelper: %d type names loaded.', len(self.type_name_dict))

    def _load_domain_range(self):
        with codecs.open('%s/PS-TP-triple.txt' % self.meta_dir, 'r', 'utf-8') as br:
            for line in br.readlines():
                s, p, o = line.strip().split('\t')
                if p == 'type.property.schema':
                    self.pred_domain_dict[s] = o
                else:
                    self.pred_range_dict[s] = o
        LogInfo.logs('FBHelper: %d domain + %d range info loaded.',
                     len(self.pred_domain_dict), len(self.pred_range_dict))

    def _load_mediator(self):
        with codecs.open('%s/mediators.tsv' % self.meta_dir, 'r', 'utf-8') as br:
            for line in br.readlines():
                self.med_type_set.add(line.strip())
        LogInfo.logs('FBHelper: %d mediator types loaded.', len(self.med_type_set))

    def _load_sup_sub_types(self):
        type_uhash = {}
        with codecs.open('%s/type_dict.tsv' % self.meta_dir, 'r', 'utf-8') as br:
            for line in br.readlines():
                spt = line.strip().split('\t')
                idx = int(spt[0])
                mid = spt[1]
                type_uhash[idx] = mid
        with codecs.open('%s/version_0.9.txt' % self.meta_dir, 'r', 'utf-8') as br:
            pairs = 0
            for line in br.readlines():
                idx_list = list(map(lambda x: int(x), line.strip().split('\t')))
                ch_idx = idx_list[0]
                ch_type = type_uhash[ch_idx]
                for fa_idx in idx_list[1:]:  # ignore child type itself
                    pairs += 1
                    fa_type = type_uhash[fa_idx]
                    self.sup_type_dict.setdefault(ch_type, set([])).add(fa_type)
                    self.sub_type_dict.setdefault(fa_type, set([])).add(ch_type)
        LogInfo.logs('FBHelper: %d sub/super type pairs loaded.', pairs)

    def _get_pred_name(self, pred):
        use_pred = pred[1:] if pred[0] == '!' else pred  # consider inverse predicates
        if use_pred.startswith('m.__') and use_pred.endswith('__'):  # virtual predicate
            return use_pred[4: -2]
        else:  # normal predicates
            return self.pred_name_dict.get(use_pred, '')

    def _get_type_name(self, tp):
        return self.type_name_dict.get(tp, '')

    def get_item_name(self, mid):
        item_name = ''
        p_name = self._get_pred_name(mid)
        if p_name != '':
            item_name = p_name
        t_name = self._get_type_name(mid)
        if t_name != '':
            item_name = t_name
        if mid.startswith('m.') and not mid.startswith('m.0'):  # it's a type/predicate, but not an entity
            last_dot = mid.rfind('.')
            last_part = mid[last_dot + 1:]
            item_name = last_part.split('_')  # simply pick the name from id
        return item_name

    def is_mediator_as_expect(self, pred):
        if pred[0] == '!':  # inverse predicate
            t = self.pred_domain_dict.get(pred[1:], '')
        else:
            t = self.pred_range_dict.get(pred, '')
        return t in self.med_type_set

    def is_type_contained_by(self, ta, tb):  # whether tb is a super type of ta
        return tb in self.sup_type_dict.get(ta, set([]))

    def get_domain(self, pred):
        if pred[0] == '!':  # inverse predicate
            return self.pred_range_dict.get(pred[1:], '')
        else:
            return self.pred_domain_dict.get(pred, '')

    def get_range(self, pred):
        if pred[0] == '!':  # inverse predicate
            return self.pred_domain_dict.get(pred[1:], '')
        else:
            return self.pred_range_dict.get(pred, '')


def adjust_name(name):
    name = name.lower()
    name = remove_parenthesis(name)
    name = re.sub(r'[/|\\,.?!@#$%^&*():;"]', '', name)          # remove puncs
    name = re.sub(' +', ' ', name).strip()                      # remove extra blanks
    return name


def remove_parenthesis(name):
    while True:
        lf_pos = name.find('(')
        if lf_pos == -1:
            break
        rt_pos = name.find(')', lf_pos+1)
        if rt_pos == -1:
            rt_pos = len(name) - 1
        name = name[:lf_pos] + name[rt_pos+1:]
    return name
