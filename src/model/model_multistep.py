# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.lookup import MutableHashTable
from .seq_helper import define_rnn_cell, sequence_loss, sentence_ppx
from .attention_decoder import create_output_fn, prepare_attention, prepare_multistep_attention, \
    attention_decoder_train, attention_decoder_inference
from .dynamic_decoder import dynamic_rnn_decoder
from ..dataset.knowledge_loader import UNK_ID, GO_ID, EOS_ID, NONE_ID


class TransDGModelMultistep(object):
    def __init__(self, word_embed, kd_embed, param_dict, use_trans_repr=True, use_trans_select=True, use_guiding=True,
                 vocab_size=30000, dim_emb=300, dim_trans=100, cell_class='GRU', num_units=512, num_layers=2,
                 max_length=60, lr_rate=0.0001, max_grad_norm=5.0, drop_rate=0.2, beam_size=1):
        # initialize params
        self.use_trans_repr = use_trans_repr
        self.use_trans_select = use_trans_select
        self.use_guiding = use_guiding
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.dim_trans = dim_trans
        self.cell_class = cell_class
        self.num_units = num_units
        self.num_layers = num_layers
        self.lr_rate = lr_rate
        self.max_grad_norm = max_grad_norm
        self.drop_rate = drop_rate
        self.max_length = max_length
        self.beam_size = beam_size

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self._init_embed(word_embed, kd_embed)
        self._init_placeholders()
        self._init_vocabs()

        self.select_mode = None
        if self.use_trans_select:
            self.select_layer = self._init_select_layer(param_dict=param_dict)
        else:
            self.select_layer = None

        # build model
        self.ppx_loss, self.loss = self.build_model(train_mode=True)
        self.generation = self.build_model(train_mode=False)

        # construct graphs for minimizing loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_rate)
        self.params = tf.global_variables()
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        self.update = optimizer.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

    def _init_vocabs(self):
        self.symbol2index = MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64,
                                             default_value=UNK_ID, shared_name="w2id_table",
                                             name="w2id_table", checkpoint=True)
        self.index2symbol = MutableHashTable(key_dtype=tf.int64, value_dtype=tf.string,
                                             default_value='_UNK', shared_name="id2w_table",
                                             name="id2w_table", checkpoint=True)
        self.kd2index = MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64,
                                         default_value=NONE_ID, shared_name="kd2id_table",
                                         name="kd2id_table", checkpoint=True)
        self.index2kd = MutableHashTable(key_dtype=tf.int64, value_dtype=tf.string,
                                         default_value='_NONE', shared_name="id2kd_table",
                                         name="id2kd_table", checkpoint=True)

    def _init_placeholders(self):
        self.posts = tf.placeholder(tf.string, (None, None), 'post')  # [batch, len]
        self.posts_length = tf.placeholder(tf.int32, (None), 'post_lens')  # batch
        self.responses = tf.placeholder(tf.string, (None, None), 'resp')  # [batch, len]
        self.responses_length = tf.placeholder(tf.int32, (None), 'resp_lens')  # batch
        self.corr_responses = tf.placeholder(tf.string, (None, None, None), 'corr_resps')  # [batch, topk, len]
        self.triples = tf.placeholder(tf.string, (None, None, None, 3), 'triples')
        self.entities = tf.placeholder(tf.string, (None, None, None), 'entities')
        self.trans_reprs = tf.placeholder(tf.float32, (None, None, self.num_units), 'trans_reprs')

    def _init_embed(self, word_embed, kd_embed=None):
        self.word_embed = tf.get_variable('word_embed', dtype=tf.float32,
                                          initializer=word_embed, trainable=False)  # [vocab_size, dim_emb]

    def _init_select_layer(self, param_dict):
        """
        :param param_dict: type dict
        :return: Defined bilinear layer or mlp layer
        """
        if "bilinear_mat" in param_dict.keys():
            self.select_mode = 'bilinear'

            def bilinear_layer(inputs1, inputs2, trainable=False):
                bilinear_mat = tf.get_variable('bilinear_mat', dtype=tf.float32,
                                               initializer=param_dict['bilinear_mat'], trainable=trainable)
                proj_repr = tf.matmul(inputs2, bilinear_mat)
                scores = tf.reduce_sum(inputs1 * proj_repr, axis=-1)
                return scores
            return bilinear_layer
        else:
            self.select_mode = 'mlp'

            def fully_connected_layer(inputs, trainable=False):
                fc1_weights = tf.get_variable('fc1_weights', dtype=tf.float32,
                                              initializer=param_dict['fc1_weights'], trainable=trainable)
                fc1_biases = tf.get_variable('fc1_biases', dtype=tf.float32,
                                             initializer=param_dict['fc1_biases'], trainable=trainable)
                hidden_outs = tf.nn.relu(tf.matmul(inputs, fc1_weights) + fc1_biases)

                fc2_weights = tf.get_variable('fc2_weights', dtype=tf.float32,
                                              initializer=param_dict['fc2_weights'], trainable=trainable)
                fc2_biases = tf.get_variable('fc2_biases', dtype=tf.float32,
                                             initializer=param_dict['fc2_biases'], trainable=trainable)
                scores = tf.matmul(hidden_outs, fc2_weights) + fc2_biases
                return scores
            return fully_connected_layer

    def build_model(self, train_mode=True):
        # build the vocab table (string to index)
        batch_size = tf.shape(self.posts)[0]
        post_word_id = self.symbol2index.lookup(self.posts)
        post_word_input = tf.nn.embedding_lookup(self.word_embed, post_word_id)  # [batch, len, unit]

        corr_responses_id = self.symbol2index.lookup(self.corr_responses)  # [batch, topk, len]
        corr_responses_input = tf.nn.embedding_lookup(self.word_embed, corr_responses_id)  # [batch, topk, len, unit]

        triple_id = self.symbol2index.lookup(self.triples)
        triple_input = tf.nn.embedding_lookup(self.word_embed, triple_id)
        triple_num = tf.shape(self.triples)[1]
        triple_input = tf.reshape(triple_input, [batch_size, triple_num, -1, 3 * self.dim_emb])
        triple_input = tf.reduce_mean(triple_input, axis=2)  # [batch, triple_num, 3*dim_emb]

        encoder_output, encoder_state = self.build_encoder(post_word_input,
                                                           corr_responses_input,
                                                           train_mode=train_mode)

        if train_mode:
            resp_target = self.symbol2index.lookup(self.responses)
            decoder_len = tf.shape(self.responses)[1]
            resp_word_id = tf.concat([tf.ones([batch_size, 1], dtype=tf.int64) * GO_ID,
                                      tf.split(resp_target, [decoder_len - 1, 1], 1)[0]], 1)  # [batch,len]
            resp_word_input = tf.nn.embedding_lookup(self.word_embed, resp_word_id)  # [batch, len, unit]
            decoder_mask = tf.reshape(tf.cumsum(
                tf.one_hot(self.responses_length - 1, decoder_len), reverse=True, axis=1),
                [-1, decoder_len])


            uniq_ids, indices = tf.unique(tf.reshape(resp_word_id, [-1]))

            # independent sample for each instance
            rand_mask = tf.random_uniform([batch_size, tf.size(uniq_ids)], dtype=tf.float32)

            # prepare indices for tf.gather_nd
            batch_wise = tf.tile(tf.expand_dims(tf.range(batch_size), axis=-1), [1, decoder_len])
            uniq_ids_wise = tf.reshape(indices, [batch_size, decoder_len])

            # gather mask and convert it to binary mask
            mask_indices = tf.stack([batch_wise, uniq_ids_wise], axis=-1)
            binary_mask = tf.floor(tf.gather_nd(rand_mask, mask_indices) + 0.5)

            # apply mask
            resp_word_dropped = resp_word_input * tf.expand_dims(binary_mask, axis=-1)


            output_logits1, output_logits2 = self.build_decoder(encoder_output, encoder_state,
                                                                triple_input, resp_word_input, resp_word_dropped,
                                                                train_mode=train_mode)

            seq_loss1 = sequence_loss(self.vocab_size, output_logits1, resp_target, decoder_mask)
            seq_loss2 = sequence_loss(self.vocab_size, output_logits2, resp_target, decoder_mask)
            sent_ppx1 = sentence_ppx(self.vocab_size, output_logits1, resp_target, decoder_mask)
            sent_ppx2 = sentence_ppx(self.vocab_size, output_logits2, resp_target, decoder_mask)
            sent_ppx = (sent_ppx1 + sent_ppx2) * 0.5
            ppx_loss = tf.identity(sent_ppx, name="ppx_loss")
            seq_loss = seq_loss1 + seq_loss2
            loss = tf.identity(seq_loss, name="loss")
            return ppx_loss, loss
        else:

            decoder_dist = self.build_decoder(encoder_output, encoder_state,
                                              triple_input, decoder_input=None, decoder_dropped=None,
                                              train_mode=False)
            generation_index = tf.argmax(decoder_dist, 2)
            generation = self.index2symbol.lookup(generation_index)
            generation = tf.identity(generation, name='generation')
            return generation

    def build_encoder(self, post_word_input, corr_responses_input, train_mode=True):
        if train_mode:
            keep_prob = 1.0 - self.drop_rate
        else:
            keep_prob = 1.0
        encoder_cell = define_rnn_cell(cell_class=self.cell_class, num_units=self.num_units,
                                       num_layers=self.num_layers, keep_prob=keep_prob)

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
            encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                              post_word_input,
                                                              self.posts_length,
                                                              dtype=tf.float32, scope=scope)
        if self.use_guiding:
            batch_size, encoder_len = tf.shape(self.posts)[0], tf.shape(self.posts)[1]
            corr_response_input = tf.reshape(corr_responses_input, [batch_size, -1, self.dim_emb])
            corr_cum_len = tf.shape(corr_response_input)[1]
            with tf.variable_scope('mutual_attention', reuse=tf.AUTO_REUSE):
                encoder_out_trans = tf.layers.dense(encoder_output, self.num_units,
                                                    name='encoder_out_transform')
                corr_response_trans = tf.layers.dense(corr_response_input, self.num_units,
                                                      name='corr_response_transform')
                encoder_out_trans = tf.expand_dims(encoder_out_trans, axis=1)
                encoder_out_trans = tf.tile(encoder_out_trans, [1, corr_cum_len, 1, 1])
                encoder_out_trans = tf.reshape(encoder_out_trans, [-1, encoder_len, self.num_units])

                corr_response_trans = tf.reshape(corr_response_trans, [-1, self.num_units])
                corr_response_trans = tf.expand_dims(corr_response_trans, axis=1)

                # TODO: try bilinear attention
                v = tf.get_variable("attention_v", [self.num_units], dtype=tf.float32)
                score = tf.reduce_sum(v * tf.tanh(encoder_out_trans + corr_response_trans), axis=2)
                alignments = tf.nn.softmax(score)

                encoder_out_tiled = tf.expand_dims(encoder_output, axis=1)
                encoder_out_tiled = tf.tile(encoder_out_tiled, [1, corr_cum_len, 1, 1])
                encoder_out_tiled = tf.reshape(encoder_out_tiled, [-1, encoder_len, self.num_units])

                context_mutual = tf.reduce_sum(tf.expand_dims(alignments, 2) * encoder_out_tiled, axis=1)
                context_mutual = tf.reshape(context_mutual, [batch_size, -1, self.num_units])
                context_mutual = tf.reduce_mean(context_mutual, axis=1)  # [batch, num_units]
            encoder_output = tf.concat([encoder_output, tf.expand_dims(context_mutual, 1)], axis=1)

        if self.use_trans_repr:
            trans_output = tf.layers.dense(self.trans_reprs, self.num_units,
                                           name='trans_reprs_transform', reuse=tf.AUTO_REUSE)
            encoder_output = tf.concat([encoder_output, trans_output], axis=1)
        print("encoder_output:", encoder_output.shape)
        return encoder_output, encoder_state

    def build_decoder(self, encoder_output, encoder_state, triple_input, decoder_input, decoder_dropped, train_mode=True):
        if train_mode:
            keep_prob = 1.0 - self.drop_rate
        else:
            keep_prob = 1.0
        decoder_cell_step1 = define_rnn_cell(cell_class=self.cell_class, num_units=self.num_units,
                                             num_layers=self.num_layers, keep_prob=keep_prob)
        decoder_cell_step2 = define_rnn_cell(cell_class=self.cell_class, num_units=self.num_units,
                                             num_layers=self.num_layers, keep_prob=keep_prob)

        if train_mode:
            with tf.variable_scope('decoder_step1', reuse=tf.AUTO_REUSE) as scope1:
                if self.use_trans_select:
                    kd_context1 = self.transfer_matching(encoder_output, triple_input)
                else:
                    kd_context1 = None

                # prepare attention
                attention_keys1, attention_values1, attention_construct_fn1 \
                    = prepare_attention(encoder_output, kd_context1, 'luong', self.num_units)

                decoder_fn_train1 = attention_decoder_train(
                    encoder_state=encoder_state,
                    attention_keys=attention_keys1,
                    attention_values=attention_values1,
                    attention_construct_fn=attention_construct_fn1)
                
                # train decoder step1
                decoder_output1, _, _ = dynamic_rnn_decoder(cell=decoder_cell_step1,
                                                            decoder_fn=decoder_fn_train1,
                                                            inputs=decoder_dropped,
                                                            sequence_length=self.responses_length,
                                                            scope=scope1)
                output_fn1 = create_output_fn(vocab_size=self.vocab_size)
                output_logits1 = output_fn1(decoder_output1)

            with tf.variable_scope('decoder_step2', reuse=tf.AUTO_REUSE) as scope2:
                if self.use_trans_select:
                    kd_context2 = self.transfer_matching(decoder_output1, triple_input)
                else:
                    kd_context2 = None
                attention_keys2, attention_values2, attention_construct_fn2 \
                    = prepare_multistep_attention(encoder_output, decoder_output1,
                                                  kd_context1, kd_context2,
                                                  'luong', self.num_units)
                decoder_fn_train2 = attention_decoder_train(
                    encoder_state=encoder_state,
                    attention_keys=attention_keys2,
                    attention_values=attention_values2,
                    attention_construct_fn=attention_construct_fn2)
                # train decoder step2
                decoder_output2, _, _ = dynamic_rnn_decoder(cell=decoder_cell_step2,
                                                            decoder_fn=decoder_fn_train2,
                                                            inputs=decoder_input,
                                                            sequence_length=self.responses_length,
                                                            scope=scope2)
                output_fn2 = create_output_fn(vocab_size=self.vocab_size)
                output_logits2 = output_fn2(decoder_output2)
                return output_logits1, output_logits2

        else:
            with tf.variable_scope('decoder_step1', reuse=tf.AUTO_REUSE) as scope1:
                if self.use_trans_select:
                    kd_context1 = self.transfer_matching(encoder_output, triple_input)
                else:
                    kd_context1 = None
                attention_keys1, attention_values1, attention_construct_fn1 \
                    = prepare_attention(encoder_output, kd_context1, 'luong', self.num_units, reuse=tf.AUTO_REUSE)
                output_fn1 = create_output_fn(vocab_size=self.vocab_size)
                # inference decoder step 1
                decoder_hidden_inference1 = attention_decoder_inference(
                    num_units=self.num_units, num_decoder_symbols=self.vocab_size,
                    output_fn=output_fn1, encoder_state=encoder_state,
                    attention_keys=attention_keys1, attention_values=attention_values1,
                    attention_construct_fn=attention_construct_fn1, embeddings=self.word_embed,
                    start_of_sequence_id=GO_ID, end_of_sequence_id=EOS_ID, maximum_length=self.max_length,
                    is_pass1=True
                )
                # get decoder hidden output
                decoder_hidden1, _, _ = dynamic_rnn_decoder(cell=decoder_cell_step1,
                                                            decoder_fn=decoder_hidden_inference1,
                                                            scope=scope1)

            with tf.variable_scope('decoder_step2', reuse=tf.AUTO_REUSE) as scope2:
                if self.use_trans_select:
                    kd_context2 = self.transfer_matching(decoder_hidden1, triple_input)
                else:
                    kd_context2 = None
                attention_keys2, attention_values2, attention_construct_fn2 \
                    = prepare_multistep_attention(encoder_output, decoder_hidden1,
                                                  kd_context1, kd_context2,
                                                  'luong', self.num_units, reuse=tf.AUTO_REUSE)
                output_fn2 = create_output_fn(vocab_size=self.vocab_size)
                # inference decoder
                decoder_fn_inference2 = attention_decoder_inference(
                    num_units=self.num_units, num_decoder_symbols=self.vocab_size,
                    output_fn=output_fn2, encoder_state=encoder_state,
                    attention_keys=attention_keys2, attention_values=attention_values2,
                    attention_construct_fn=attention_construct_fn2, embeddings=self.word_embed,
                    start_of_sequence_id=GO_ID, end_of_sequence_id=EOS_ID, maximum_length=self.max_length
                )
                # get decoder output
                decoder_distribution, _, _ = dynamic_rnn_decoder(
                    cell=decoder_cell_step2, decoder_fn=decoder_fn_inference2, scope=scope2
                )
                return decoder_distribution

    def transfer_matching(self, context_repr, knowledge_repr):
        context = tf.reduce_mean(context_repr, axis=1)  # [batch, num_units]
        kd_num = tf.shape(knowledge_repr)[1]
        context_tile = tf.tile(tf.expand_dims(context, axis=1), [1, kd_num, 1])  # [batch, kd_num, num_units]

        knowledge = tf.layers.dense(knowledge_repr, self.dim_emb,
                                    name='knowledge_transform')  # [batch, kd_num, dim_emb]

        if self.select_mode == 'bilinear':
            context_reshaped = tf.reshape(context_tile, [-1, self.num_units])
            knowledge_reshaped = tf.reshape(knowledge, [-1, self.dim_emb])
            em_scores = self.select_layer(context_reshaped, knowledge_reshaped)
        else:
            concat_repr = tf.concat([context_tile, knowledge], axis=-1)  # [batch, kd_num, num_units+dim_emb]
            concat_repr_reshaped = tf.reshape(concat_repr, [-1, self.num_units + self.dim_emb])
            em_scores = self.select_layer(concat_repr_reshaped)

        batch_size = tf.shape(self.posts)[0]
        em_scores = tf.reshape(em_scores, [batch_size, kd_num])
        kd_context = tf.matmul(tf.expand_dims(em_scores, axis=1), knowledge)
        kd_context = tf.reshape(kd_context, [batch_size, self.dim_emb])  # [batch, dim_emb]
        return kd_context

    def set_vocabs(self, session, vocab, kd_vocab):
        op_in = self.symbol2index.insert(tf.constant(vocab),
                                         tf.constant(list(range(self.vocab_size)), dtype=tf.int64))
        session.run(op_in)
        op_out = self.index2symbol.insert(tf.constant(list(range(self.vocab_size)), dtype=tf.int64),
                                          tf.constant(vocab))
        session.run(op_out)
        op_in = self.kd2index.insert(tf.constant(kd_vocab),
                                     tf.constant(list(range(len(kd_vocab))), dtype=tf.int64))
        session.run(op_in)
        op_out = self.index2kd.insert(tf.constant(list(range(len(kd_vocab))), dtype=tf.int64),
                                      tf.constant(kd_vocab))
        session.run(op_out)

    def show_parameters(self):
        for var in self.params:
            print("%s: %s" % (var.name, var.get_shape().as_list()))

    def train_batch(self, session, data, trans_reprs):
        input_feed = {self.posts: data['post'],
                      self.posts_length: data['post_len'],
                      self.responses: data['response'],
                      self.responses_length: data['response_len'],
                      self.corr_responses: data['corr_responses'],
                      self.triples: data['all_triples'],
                      self.entities: data['all_entities']
                      }
        if self.use_trans_repr:
            input_feed[self.trans_reprs] = trans_reprs

        output_feed = [self.ppx_loss, self.loss, self.update]
        outputs = session.run(output_feed, feed_dict=input_feed)
        return outputs[0], outputs[1]

    def eval_batch(self, session, data, trans_reprs):
        input_feed = {self.posts: data['post'],
                      self.posts_length: data['post_len'],
                      self.responses: data['response'],
                      self.responses_length: data['response_len'],
                      self.corr_responses: data['corr_responses'],
                      self.triples: data['all_triples'],
                      self.entities: data['all_entities']
                      }
        if self.use_trans_repr:
            input_feed[self.trans_reprs] = trans_reprs

        output_feed = [self.ppx_loss, self.loss]
        outputs = session.run(output_feed, feed_dict=input_feed)
        return outputs[0], outputs[1]

    def decode_batch(self, session, data, trans_reprs):
        input_feed = {self.posts: data['post'],
                      self.posts_length: data['post_len'],
                      self.responses: data['response'],
                      self.responses_length: data['response_len'],
                      self.corr_responses: data['corr_responses'],
                      self.triples: data['all_triples'],
                      self.entities: data['all_entities']
                      }
        if self.use_trans_repr:
            input_feed[self.trans_reprs] = trans_reprs

        output_feed = [self.generation, self.ppx_loss, self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[-1]
