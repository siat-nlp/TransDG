# -*- coding: utf-8 -*-
import re
import os
import time
import shutil
from .log_util import LogInfo


def save_model(saver, sess, model_dir, epoch, valid_metric):
    t0 = time.time()
    LogInfo.logs('Saving model into [%s] ...', model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_fp = model_dir + '/best.model'
    saver.save(sess, model_fp)
    with open(model_dir + '/epoch', 'w') as bw:
        bw.write('epoch=%d\n' % epoch)
        bw.write('valid_F1=%.6f\n' % valid_metric)
    LogInfo.logs('Saved [%.3fs]', time.time() - t0)


def load_model(saver, sess, model_dir):
    t0 = time.time()
    LogInfo.logs('Loading model from [%s] ...', model_dir)
    model_fp = model_dir + '/best.model'
    saver.restore(sess, model_fp)
    LogInfo.logs('Loaded [%.3fs]', time.time() - t0)

    start_epoch = 0
    valid_metric = 0.
    epoch_nb_fp = model_dir + '/epoch'
    if os.path.isfile(epoch_nb_fp):
        with open(epoch_nb_fp, 'r') as br:
            start_epoch = int(br.readline().strip())
            valid_metric = float(br.readline().strip())
    return start_epoch, valid_metric


def analyze_status(status_fp):
    """ Read the status file, and keep the last running information """
    lines = []
    with open(status_fp, 'r') as br:
        for line in br.readlines():
            lines.append(re.sub(' +', '\t', line).strip())
    start_idx = 0
    for line_idx, line in enumerate(lines):
        if line.startswith('Epoch'):
            start_idx = line_idx

    ret_dict = {}       # <Key, [Value]>
    header_spt = lines[start_idx].strip().replace(' ', '\t').split('\t')
    for key in header_spt:
        ret_dict.setdefault(key, [])
    for line in lines[start_idx+1:]:
        value_spt = line.strip().replace(' ', '\t').split('\t')
        for key, value in zip(header_spt, value_spt):
            try:
                v_float = float(value)
                ret_dict[key].append(v_float)
            except ValueError:
                ret_dict[key].append(value)

    return ret_dict


def construct_display_header():
    raw_header_list = ['Epoch']
    local_header_list = ['rm_loss']
    for mark in ['train', 'valid', 'test']:
        local_header_list.append('%s_F1' % mark)
    raw_header_list.append(' |  ')
    raw_header_list += local_header_list
    raw_header_list += [' |  ', 'Status', 'Time']
    disp_header_list = []
    no_tab = True
    for idx, header in enumerate(raw_header_list):
        if not (no_tab or header.endswith(' ')):
            disp_header_list.append('\t')
        disp_header_list.append(header)
        no_tab = header.endswith(' ')
    return disp_header_list


def delete_dir(target_dir):
    if os.path.islink(target_dir):
        os.remove(target_dir)
    elif os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
