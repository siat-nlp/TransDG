#!/usr/bin/env python
import os
import json
from .retriever import Indexer, Queryer

DATA_DIR = "./data/Reddit"
INDEX_DIR = "./index"


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


def main():
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)
    # load train data
    data_train = load_raw_trainset(DATA_DIR)

    # build mappings: <id, post>, <id, response>
    id2post = build_mapping(data_train, INDEX_DIR)

    # build data indexes of posts
    indexer = Indexer(INDEX_DIR)
    indexer.build_index(id2post)


if __name__ == "__main__":
    main()

    query = "i reddit at work because it is better than working . what makes this a reason"
    queryer = Queryer(INDEX_DIR, top_k=5)
    results = queryer.run_query(query)
    print("query:", query)

    ids = results["ids"]
    contents = results["contents"]
    for i in range(len(ids)):
        print("id:", ids[i])
        print("content:", contents[i])
