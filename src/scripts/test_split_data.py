import os
import pickle
import json

data_dir = "./data/Reddit"
mode = 'train'

pickle_fp = '%s/Reddit.%s.pkl' % (data_dir, mode)
print('Loading Reddit dialogs from [%s] ...' % pickle_fp)

with open(pickle_fp, 'rb') as br:
    dg_list = pickle.load(br)

dg_save_list = []

fp = '%s/%sset.txt' % (data_dir, mode)
with open(fp, 'r') as br:
    for idx, line in enumerate(br):
        dg_line = json.loads(line)
        dialog = {'utterance': dg_list[idx]['utterance'],
                  'tokens': dg_list[idx]['tokens'],
                  'parse': dg_list[idx]['parse'],
                  'response': dg_list[idx]['response'],
                  'corr_responses': dg_list[idx]['corr_responses'],
                  'all_triples': dg_list[idx]['all_triples'],
                  'all_entities': dg_line['all_entities']
                  }
        dg_save_list.append(dialog)
        if len(dg_save_list) % 10000 == 0:
            print('%d scanned.' % len(dg_save_list))

pickle_save_fp = '%s/Reddit.%s_v2.pkl' % (data_dir, mode)
with open(pickle_save_fp, 'wb') as bw:
    pickle.dump(dg_save_list, bw)
    print('%d Reddit saved in [%s].' % (len(dg_save_list), pickle_save_fp))
