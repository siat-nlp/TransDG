# -*- coding: utf-8 -*-
import json
import argparse
from src.utils.retriever import Queryer


def data_gen(data_dir, index_dir, top_k=3, mode="valid"):
    posts, resps, all_triples, all_entities = [], [], [] ,[]

    with open("%s/%sset.txt" % (data_dir, mode)) as f:
        for idx, line in enumerate(f):
            text_line = json.loads(line)
            post = " ".join(text_line['post'])
            resp = " ".join(text_line['response'])
            triples = text_line['all_triples']
            entities = text_line['all_entities']
            posts.append(post)
            resps.append(resp)
            all_triples.append(triples)
            all_entities.append(entities)
            if idx % 100000 == 0 and idx > 0:
                print("loading %d samples..." % idx)
    print("load %s set done." % mode)
    
    with open("%s/id2response.json" % index_dir, 'r') as fr:
        id2response = json.load(fr)
    print("load index id2response done.")
    
    entity_lists = []
    if mode == "test":
        with open("%s/csk_entity.txt" % data_dir) as f:
            for i, line in enumerate(f):
                e = line.strip()
                entity_lists.append(e)
        print("load csk_entity done.")

    cnt = 0
    queryer = Queryer(index_dir, top_k=top_k)
    with open("%s/%s.txt" % (data_dir, mode), 'w') as fw:
        for post, resp, all_triple, all_entity in zip(posts, resps, all_triples, all_entities):
            # search corresponding responses of top-k posts
            query = _validate(post)
            result = queryer.run_query(query)
            result_ids = result['ids']
            response_k = []
            for idx in result_ids:
                sent = id2response[idx]
                response_k.append(sent)
            if mode == "test":
                ent_indexs = []
                for ent_list in all_entity:
                    for idx in ent_list:
                        ent_indexs.append(idx)
                entities = [entity_lists[idx] for idx in ent_indexs]
                data = {'post': post,
                        'response': resp,
                        'corr_responses': response_k,
                        'all_triples': all_triple,
                        'all_entities': all_entity,
                        'entities': entities
                    }
            else:
                data = {'post': post,
                        'response': resp,
                        'corr_responses': response_k,
                        'all_triples': all_triple,
                        'all_entities': all_entity
                    }
            json_str = json.dumps(data)
            fw.write(json_str + '\n')
            cnt += 1
            if cnt % 1000 == 0:
                print("writing %d samples..." % cnt)
    print("process %s done." % mode)

def _validate(query):
    valid_query = str(query).strip()
    remove_str = ['*', '?', '!', ':', '-', '(', ')', '[', ']', '{', '}']
    for s in remove_str:
        if s in valid_query:
            valid_query = valid_query.replace(s, '')

    return valid_query


def main(args):
    data_gen(data_dir=args.data_dir, index_dir=args.index_dir, top_k=args.top_k, mode=args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate top-k similar responses for Reddit dataset")
    parser.add_argument('--data_dir', type=str, help="Reddit data directory")
    parser.add_argument('--index_dir', type=str, help="Reddit index directory")
    parser.add_argument('--mode', type=str, choices=['train', 'valid', 'test'])
    parser.add_argument('--top_k', type=int, default=3, help='top-k')
    parsed_args = parser.parse_args()

    main(parsed_args)
    