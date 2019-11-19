# -*- coding: utf-8 -*-
import json
from collections import namedtuple


RawPath = namedtuple('RawPath', ['path_cate', 'focus', 'pred_seq'])

tml_comp_dict = {
    '==': u'm.__in__',
    '<': u'm.__before__',
    '>': u'm.__after__',
    '>=': u'm.__since__'
}

ordinal_dict = {
    'max': u'm.__max__',
    'min': u'm.__min__'
}


class Schema(object):

    def __init__(self):
        self.q_idx = None
        self.gather_linkings = None     # all related linkings of this question (either used or not used)

        self.use_idx = None     # UNIQUE schema index mainly used in dataset & dataloader
        self.ori_idx = None     # original index, equals to the row number of the schema in the file

        self.p = self.r = self.f1 = None
        self.rm_f1 = self.el_f1 = None      # the adjusted F1 for different tasks
        self.ans_size = None
        self.hops = None
        self.aggregate = None           # whether using COUNT(*) as aggregation operator or not.
        self.full_constr = None         # whether to use full length of constraints (or ignore predicates at main path)
        self.main_pred_seq = None       # saving [p1, p2]
        self.qa_list = None
        self.path_list = None           # a list of PURE mid sequence
        self.raw_paths = []
        # [ (path category, linking result, predicate sequence) ]
        # path category: 'Main', 'Entity', 'Type', 'Ordinal', 'Time'
        # linking result: LinkData
        #                   old: eff_candgen/combinator.py: LinkData,
        #                   new: candgen_acl18/global_linker.py: LinkData
        # predicate sequence: sequences of predicates only
        #   main path: focus --> answer
        #   constraint path: answer --> constraint node

        self.replace_linkings = None
        # If one linking result is not used in the path list,
        # we save it here, and replace the corresponding mention by placeholders.

        self.input_np_dict = None    # the dictionary storing various input tensors
        self.run_info = None         # additional attribute to save runtime results

    def read_schema_from_json(self, q_idx, json_line, gather_linkings, ori_idx, full_constr=False):
        """
        Read the schema from json files. (provided with detail linking results)
        We will read the following information from the json file:
            1. p / r / f1
            2. raw_paths: [ (category, focus_idx, focus_mid, pred_seq) ]
        :param q_idx: question index
        :param json_line: a line of schema (json format, usually a dict)
        :param gather_linkings: [LinkData]
        :param ori_idx: original index
        :param full_constr: whether to use full length constraint, or the shorter length one.
        :return: A schema instance
        """
        info_dict = json.loads(json_line.strip())

        if 'p' in info_dict.keys():
            self.p = info_dict['p']
            self.r = info_dict['r']
            self.f1 = info_dict['f1']
            self.ans_size = info_dict['ans_size']

        else:
            self.ans_size = info_dict['ans_size']

        self.q_idx = q_idx
        self.gather_linkings = gather_linkings
        self.ori_idx = ori_idx + 1
        self.full_constr = full_constr

        self.aggregate = info_dict.get('agg', False)
        self.hops = info_dict.get('hops')

        for raw_path in info_dict['raw_paths']:
            category, focus_idx, focus_mid, pred_seq = raw_path
            if category == 'Main':
                self.hops = len(pred_seq)
                self.main_pred_seq = pred_seq
            elif not self.full_constr:
                # shorten constraints, be careful with the constraint direction, still ANS-->CONSTR
                assert 1 <= len(pred_seq) <= 2
                if len(pred_seq) == 2:
                    pred_seq = pred_seq[1:]  # ignore the overlapping predicate at the main path
            focus = gather_linkings[focus_idx]
            self.raw_paths.append(RawPath(path_cate=category, focus=focus, pred_seq=pred_seq))

    def construct_path_list(self):
        # Convert raw path into formal mid sequence
        if self.path_list is not None and self.replace_linkings is not None:
            return

        self.path_list = []
        self.replace_linkings = []

        assert self.raw_paths is not None
        for raw_path in self.raw_paths:
            path_cate, link_data, pred_seq = raw_path
            if path_cate == 'Main':
                self.path_list.append(pred_seq)
                self.replace_linkings.append(link_data)
            elif path_cate == 'Type':
                type_mid = link_data.value
                use_mid_seq = list(pred_seq)
                use_mid_seq.append(type_mid)
                self.path_list.append(use_mid_seq)
            else:
                # Prepare to fix the direction, if possible
                use_mid_seq = []
                if path_cate == 'Entity':
                    use_mid_seq = list(pred_seq)
                    self.replace_linkings.append(link_data)
                elif path_cate == 'Time':
                    comp = link_data.comp
                    assert comp in tml_comp_dict
                    use_mid_seq = list(pred_seq)
                    # virtual mid for time comparison
                    use_mid_seq.append(tml_comp_dict[comp])
                    self.replace_linkings.append(link_data)
                elif path_cate == 'Ordinal':
                    comp = link_data.comp
                    assert comp in ordinal_dict
                    use_mid_seq = list(pred_seq)
                    use_mid_seq.append(ordinal_dict[comp])

                self.path_list.append(use_mid_seq)
