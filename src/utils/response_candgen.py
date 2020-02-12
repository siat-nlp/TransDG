# -*- coding: utf-8 -*-
import json
import argparse
from .retriever import Queryer


DATA_DIR = "./data/Reddit"
INDEX_DIR = "./index"


def data_gen(top_k=3):
    train_post, train_res = [], []
    valid_post, valid_res = [], []
    test_post, test_res, test_entities = [], [], []

    with open('%s/trainset.txt' % DATA_DIR) as f:
        for idx, line in enumerate(f):
            text_line = json.loads(line)
            post = " ".join(text_line['post'])
            response = " ".join(text_line['response'])
            train_post.append(post)
            train_res.append(response)
            if idx % 100000 == 0 and idx > 0:
                print('read train file line %d' % idx)

    with open('%s/validset.txt' % DATA_DIR) as f:
        for line in f:
            text_line = json.loads(line)
            post = " ".join(text_line['post'])
            response = " ".join(text_line['response'])
            valid_post.append(post)
            valid_res.append(response)

    with open('%s/testset.txt' % DATA_DIR) as f:
        for line in f:
            text_line = json.loads(line)
            post = " ".join(text_line['post'])
            response = " ".join(text_line['response'])
            test_post.append(post)
            test_res.append(response)
            test_entities.append(text_line['all_entities'])

    with open("%s/id2response.json" % INDEX_DIR, 'r') as file:
        id2response = json.load(file)

    cnt = 0
    queryer = Queryer(INDEX_DIR, top_k=top_k)

    with open('%s/train.txt' % DATA_DIR, 'w') as fw:
        for post, res in zip(train_post, train_res):
            # search corresponding responses of top-k posts
            query = _validate(post)
            result = queryer.run_query(query)
            result_ids = result['ids']
            response_k = []
            for idx in result_ids:
                sent = id2response[idx]
                response_k.append(sent)
            train = {'post': post,
                     'response': res,
                     'corr_responses': response_k}
            json_str = json.dumps(train)
            fw.write(json_str + '\n')
            cnt += 1
            if cnt % 10000 == 0:
                print("%d train done" % cnt)
    cnt = 0
    with open('%s/valid.txt' % DATA_DIR, 'w') as fw:
        for post, res in zip(valid_post, valid_res):
            # search corresponding responses of top-k posts
            query = _validate(post)
            result = queryer.run_query(query)
            result_ids = result['ids']
            response_k = []
            for idx in result_ids:
                sent = id2response[idx]
                response_k.append(sent)
            valid = {'post': post,
                     'response': res,
                     'corr_responses': response_k}
            json_str = json.dumps(valid)
            fw.write(json_str + '\n')
            cnt += 1
            if cnt % 1000 == 0:
                print("%d valid done" % cnt)

    entity_lists = []
    with open('%s/csk_entity.txt' % DATA_DIR) as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_lists.append(e)

    cnt = 0
    with open('%s/test.txt' % DATA_DIR, 'w') as fw:
        for post, res, all_entity in zip(test_post, test_res, test_entities):
            # search corresponding responses of top-k posts
            query = _validate(post)
            result = queryer.run_query(query)
            result_ids = result['ids']
            response_k = []
            for idx in result_ids:
                sent = id2response[idx]
                response_k.append(sent)

            ent_indexs = []
            for ent_list in all_entity:
                for idx in ent_list:
                    ent_indexs.append(idx)
            entities = [entity_lists[idx] for idx in ent_indexs]

            test = {'post': post,
                    'response': res,
                    'corr_responses': response_k,
                    'entities': entities}
            json_str = json.dumps(test)
            fw.write(json_str + '\n')
            cnt += 1
            if cnt % 1000 == 0:
                print("%d test done" % cnt)


def _validate(query):
    valid_query = str(query).strip()
    remove_str = ['*', '?', '!', ':', '-', '(', ')', '[', ']', '{', '}']
    for s in remove_str:
        if s in valid_query:
            valid_query = valid_query.replace(s, '')

    return valid_query


def main(args):
    data_gen(top_k=args.top_k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate top-k similar responses for dataset")
    parser.add_argument('--top_k', type=int, default=3, help='top-k')
    parsed_args = parser.parse_args()

    main(parsed_args)
    