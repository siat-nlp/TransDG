#!/usr/bin/env python
import os
import json
import argparse
from src.utils.retriever import Indexer, Queryer


def load_raw_trainset(data_dir):
    raw_train = []
    with open('%s/trainset.txt' % data_dir) as f:
        for idx, line in enumerate(f):
            raw_train.append(json.loads(line))
            if idx % 100000 == 0 and idx > 0:
                print('read raw train line %d' % idx)
    return raw_train


def build_mapping(data, index_dir):
    posts = [item["post"] for item in data]
    responses = [item['response'] for item in data]
    keys = []
    values = []
    for idx in range(len(posts)):
        keys.append(str(" ".join(posts[idx])))
        values.append(str(" ".join(responses[idx])))

    id2post = {}
    id2response = {}
    global_id = 0
    for i in range(len(keys)):
        id2post[str(global_id)] = keys[i]
        id2response[str(global_id)] = values[i]
        global_id += 1
    print("total id:", global_id)
    with open("%s/id2post.json" % index_dir, "w") as fw:
        json.dump(id2post, fw)
    with open("%s/id2response.json" % index_dir, "w") as fw:
        json.dump(id2response, fw)

    return id2post


def main(args):
    if not os.path.exists(args.index_dir):
        os.makedirs(args.index_dir)
    # load train data
    data_train = load_raw_trainset(args.data_dir)

    # build mappings: <id, post>, <id, response>
    id2post = build_mapping(data_train, args.index_dir)

    # build data index of posts
    indexer = Indexer(args.index_dir)
    indexer.build_index(id2post)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build index for retrieving")
    parser.add_argument('--data_dir', type=str, help="Reddit data directory")
    parser.add_argument('--index_dir', type=str, help="Reddit index directory")
    parsed_args = parser.parse_args()

    main(parsed_args)
