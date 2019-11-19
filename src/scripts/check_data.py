import pickle
import json
from src.dataset.data_loader import DataLoader
from src.dataset.knowledge_loader import KnowledgeLoader


kd_dir = "./data/Reddit"
vocab_size=30000
dim_emb=300


with open('%s/vocab' % kd_dir) as fr:
    raw_vocab = json.load(fr)
vocab_list = sorted(raw_vocab, key=raw_vocab.get, reverse=True)
with open("%s/vocab.txt" % kd_dir, 'w') as fw:
    for w in vocab_list:
        fw.write(w)
        fw.write('\n')
print("total vocabs:", len(vocab_list))


entity_list = []
with open('%s/entity.txt' % kd_dir) as f:
    for i, line in enumerate(f):
        e = line.strip()
        entity_list.append(e)

# load knowledge
kd_loader = KnowledgeLoader(kd_dir)
word_vocab, word_embed = kd_loader.load_vocab(vocab_size=vocab_size, embed_dim=dim_emb)
#kd_vocab, kd_embed = kd_loader.load_entity_relation()

common_words = set(vocab_list) & set(entity_list)
common_words = list(common_words)
print("entity_vocab:", len(entity_list))
print("common_words:", len(common_words))

'''
data_dir = "./data/final_data"
fill_list_path = "./data/final_data/all_list"
batch_size=100

# wait for train_batcher queue caching
train_chunk_list = []
with open("%s/all_list" % data_dir, 'r') as fr:
    for line in fr:
        train_chunk_list.append(line.strip())

f = train_chunk_list[0]
with open("%s/%s" % (data_dir, f), 'br') as reader:
    q_dict = pickle.load(reader)
    data_loader = DataLoader(batch_size=batch_size)
    data_loader.feed_by_data(q_dict)
    print('file=[%s] n_rows = %d, batch_size = %d, n_batch = %d.' %
          (f, data_loader.n_rows, data_loader.batch_size, data_loader.n_batch))
    for batch_idx in range(data_loader.n_batch):
        local_data, local_size = data_loader.get_batch(batch_idx=batch_idx)
        print("local_data=%s local_size=%d" % (local_data['post'].shape, local_size))
        if local_size == 1:
            print("local_post:", local_data['post'])
            print("local_response:", local_data['response'])

for idx, f in enumerate(train_chunk_list):
    with open("%s/%s" % (data_dir, f), 'br') as reader:
        q_dict = pickle.load(reader)
        data_loader = DataLoader(batch_size=batch_size)
        data_loader.feed_by_data(q_dict)
        print('file=[%s] n_rows = %d, batch_size = %d, n_batch = %d.' %
              (f, data_loader.n_rows, data_loader.batch_size, data_loader.n_batch))
        for batch_idx in range(data_loader.n_batch):
            local_data, local_size = data_loader.get_batch(batch_idx=batch_idx)
            if local_data['post'].shape[0] != local_size:
                print("Error! [%s]" % f)
                print("post:", local_data['post'].shape)
            if local_data['post_len'].shape[0] != local_size:
                print("Error! [%s]" % f)
                print("post_len:", local_data['post_len'].shape)
            if local_data['response'].shape[0] != local_size:
                print("Error! [%s]" % f)
                print("response:", local_data['response'].shape)
            if local_data['response_len'].shape[0] != local_size:
                print("Error! [%s]" % f)
                print("response_len:", local_data['response_len'].shape)
            if local_data['corr_responses'].shape[0] != local_size:
                print("Error! [%s]" % f)
                print("corr_responses:", local_data['corr_responses'].shape)
            if local_data['all_triples'].shape[0] != local_size:
                print("Error! [%s]" % f)
                print("all_triples:", local_data['all_triples'].shape)
'''