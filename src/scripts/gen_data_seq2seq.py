import json


data_dir = "./data/Reddit"

train_file = "%s/test_v3.txt" % data_dir
train_post = []
train_resp = []
with open(train_file, 'r') as fr:
    for idx, line in enumerate(fr):
        dialog_line = json.loads(line)
        post = dialog_line['post']
        resp = dialog_line['response']
        train_post.append(post)
        train_resp.append(resp)
        if idx > 0 and idx % 100000 == 0:
            print("%d readed" % idx)
print("train post:", len(train_post))
print("train resp:", len(train_resp))
out_post_file = "%s/test_post.txt" % data_dir
out_resp_file = "%s/test_response.txt" % data_dir
with open(out_post_file, 'w') as fw:
    for post in train_post:
        fw.write(post)
        fw.write('\n')
with open(out_resp_file, 'w') as fw:
    for resp in train_resp:
        fw.write(resp)
        fw.write('\n')
