# -*- coding: utf-8 -*-
"""
Wrapper for generation
"""
import os
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class Generator:

    def __init__(self, model, sess, trans_model, trans_sess, log_dir, ckpt, stop_words, csk_entities, set_num=5000):
        self.model = model
        self.sess = sess
        self.trans_model = trans_model
        self.trans_sess = trans_sess
        self.output_dir = "%s/output" % log_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.ckpt = ckpt
        self.stop_words = stop_words
        self.csk_entities = csk_entities
        self.set_num = set_num
        self.scan_data = 0

    def _decode(self, local_data, local_size):
        if self.trans_model is not None:
            trans_repr = self.trans_model.transfer_encode(self.trans_sess, local_data)
        else:
            trans_repr = None
        generation_batch, ppx_loss_batch, loss_batch = self.model.decode_batch(self.sess, local_data, trans_repr)
        self.scan_data += local_size
        print("decoding %d finished. loss_batch=%.3f ppx_loss=%.3f ppx=%.3f" %
              (self.scan_data, np.mean(loss_batch), np.mean(ppx_loss_batch), np.exp(np.mean(ppx_loss_batch))))
        return generation_batch, ppx_loss_batch, loss_batch
    
    def _post_process(self, posts, responses, all_entities, results, ppx_loss):
        """
        Write results to files
        """
        res_path = "%s/eval-%s.log" % (self.output_dir, self.ckpt)
        out_path = "%s/out-%s.txt" % (self.output_dir, self.ckpt)

        with open(res_path, 'w') as resfile, open(out_path, 'w') as outfile:
            print("writing evaluation results to [%s]..." % res_path)
            print("writing generation results to [%s]..." % out_path)
            outfile.write('model: %s\n' % self.ckpt)
            match_entity_sum = [.0] * 4
            cnt = 0
            hypotheses = []
            references = []
            for post, response, result, entities in zip(posts, responses, results, all_entities):
                references.append([response.split()])  # only 1 reference, shape=[[token1, token2, ...]]
                hypotheses.append(result.split())      # hypothese = [token1, token2, ...]

                setidx = int(cnt / self.set_num)
                result_matched_entities = []
                result_tokens = result.split()
                for word in result_tokens:
                    if word not in self.stop_words and word in entities:
                        if word not in result_matched_entities:
                            result_matched_entities.append(word)

                match_entity_sum[setidx] += len(set(result_matched_entities))
                cnt += 1
                match_entity_str = " ".join(result_matched_entities)
                outfile.write('post: %s\nresponse: %s\nresult: %s\nmatch_entity: %s\n\n' %
                              (post, response, result, match_entity_str))

            match_entity_sum = [m / self.set_num for m in match_entity_sum] + [sum(match_entity_sum) / len(responses)]
            losses = [np.sum(ppx_loss[x:x + self.set_num]) / float(self.set_num) for x in range(0, self.set_num * 4, self.set_num)] + \
                     [np.sum(ppx_loss) / float(self.set_num * 4)]
            ppxs = [np.exp(x) for x in losses]

            smooth_func = SmoothingFunction(epsilon=0.1).method1
            cor_bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
            cor_bleu_2 = corpus_bleu(references, hypotheses, smoothing_function=smooth_func, weights=(0, 1, 0, 0))
            cor_bleu_3 = corpus_bleu(references, hypotheses, smoothing_function=smooth_func, weights=(0, 0, 1, 0))
            cor_bleu_4 = corpus_bleu(references, hypotheses, smoothing_function=smooth_func, weights=(0, 0, 0, 1))
            cor_bleu = corpus_bleu(references, hypotheses)
            bleus = [cor_bleu_1, cor_bleu_2, cor_bleu_3, cor_bleu_4, cor_bleu]

            def show(x):
                return ', '.join([str(v) for v in x])

            eval_res_str = "model: %s\n\tbleu: %s\n\tperplexity: %s\n\tmatch_entity_rate: %s" % \
                           (self.ckpt, show(bleus), show(ppxs), show(match_entity_sum))
            resfile.write(eval_res_str)
            print(eval_res_str)
    
    def generate(self, data_loader):
        posts = []
        responses = []
        all_entities = []
        ppx_loss = []
        results = []
        for batch_idx in range(data_loader.n_batch):
            batch_data, local_size = data_loader.get_batch(batch_idx=batch_idx)
            batch_post = batch_data['post']
            batch_resp = batch_data['response']
            batch_ents = batch_data['entities']
            for idx, data in enumerate(batch_post):
                post = " ".join(data)
                post = post.split('_EOS')[0]
                posts.append(post)
                resp = " ".join(batch_resp[idx])
                resp = resp.split('_EOS')[0]
                responses.append(resp)

                entities = set(batch_ents[idx])
                all_entities.append(entities)

            generation_batch, ppx_loss_batch, loss_batch = self._decode(local_data=batch_data, local_size=local_size)
            ppx_loss += [x for x in ppx_loss_batch]
            for sent in generation_batch:
                tokens = [str(r, encoding='utf-8') for r in sent]
                result = []
                for token in tokens:
                    if token != '_EOS':
                        result.append(token)
                    else:
                        break
                result = " ".join(result)
                results.append(result)
        self._post_process(posts, responses, all_entities, results, ppx_loss)
