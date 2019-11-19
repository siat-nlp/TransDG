# -*- coding: utf-8 -*-
import tensorflow as tf

from .attention import Attention
from .seq_helper import seq_hidden_averaging, seq_encoding, cosine_sim, \
    seq_encoding_with_aggregation, seq_hidden_max_pooling
from .rnn_encoder import BidirectionalRNNEncoder
from ..utils.log_util import LogInfo


class KbqaModel:

    def __init__(self, qw_max_len, pw_max_len, path_max_size, pseq_max_len,
                 dim_emb, w_emb_fix, n_words, n_mids, n_paths, drop_rate,
                 rnn_config, att_config, path_usage, sent_usage, seq_merge_mode, scoring_mode,
                 final_func, loss_margin, optm_name, learning_rate):
        LogInfo.begin_track('Build KBQA Model:')

        self.qw_max_len = qw_max_len
        self.pw_max_len = pw_max_len
        self.path_max_size = path_max_size
        self.pseq_max_len = pseq_max_len

        self.dim_emb = dim_emb
        self.path_usage = path_usage
        self.sent_usage = sent_usage
        self.seq_merge_mode = seq_merge_mode
        self.scoring_mode = scoring_mode
        self.final_func = final_func
        self.margin = loss_margin

        assert optm_name in ('Adam', 'Adadelta', 'Adagrad', 'GradientDescent')
        self.optimizer = getattr(tf.train, optm_name + 'Optimizer')
        self.learning_rate = learning_rate

        if att_config['att_func'] != 'noAtt':
            self.att_config = att_config
        else:
            self.att_config = None

        self.rnn_config = rnn_config
        self.rnn_config['reuse'] = tf.AUTO_REUSE
        self.dim_hidden = 2 * rnn_config['num_units']

        self.input_tensor_dict = self.input_tensor_definition(
            qw_max_len=self.qw_max_len,
            pw_max_len=self.pw_max_len,
            path_max_size=self.path_max_size,
            pseq_max_len=self.pseq_max_len
        )
        LogInfo.logs('Global input tensors defined.')

        with tf.variable_scope('embedding_lookup', reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                w_trainable = True if w_emb_fix == 'Upd' else False
                LogInfo.logs('Word embedding trainable: %s', w_trainable)
                self.w_embedding_init = tf.placeholder(tf.float32, [n_words, dim_emb], 'w_embedding_init')
                self.w_embedding = tf.get_variable(name='w_embedding', initializer=self.w_embedding_init,
                                                   trainable=w_trainable)
                self.m_embedding_init = tf.placeholder(tf.float32, [n_mids, dim_emb], 'm_embedding_init')
                self.m_embedding = tf.get_variable(name='m_embedding', initializer=self.m_embedding_init)
                self.p_embedding_init = tf.placeholder(tf.float32, [n_paths, dim_emb], 'p_embedding_init')
                self.p_embedding = tf.get_variable(name='p_embedding', initializer=self.p_embedding_init)
        self.dropout_layer = tf.layers.Dropout(drop_rate)
        LogInfo.logs('Dropout: %.2f', drop_rate)

        # Build the main graph for both optm and eval
        self.optm_tensor_dict = self.build_graph(mode_str='optm')
        self.eval_tensor_dict = self.build_graph(mode_str='eval')
        self.sent_tensor_dict = self.build_sent()

        rm_weighted_loss = self.get_pair_loss(optm_score=self.optm_tensor_dict['rm_score'])
        self.rm_loss = tf.reduce_mean(rm_weighted_loss, name='rm_loss')
        self.rm_update, self.optm_rm_summary = self.build_update_summary()
        LogInfo.logs('Loss function: Hinge-%.1f', self.margin)
        LogInfo.logs('Loss & update defined.')

        LogInfo.end_track('End of Model')

    @staticmethod
    def input_tensor_definition(qw_max_len, pw_max_len, path_max_size, pseq_max_len):
        input_tensor_dict = {
            # Path input
            'path_size': tf.placeholder(tf.int32, [None], 'path_size'),
            'path_ids': tf.placeholder(tf.int32, [None, path_max_size], 'path_ids'),
            'pw_input': tf.placeholder(tf.int32, [None, path_max_size, pw_max_len], 'pw_input'),
            'pw_len': tf.placeholder(tf.int32, [None, path_max_size], 'pw_len'),
            'pseq_ids': tf.placeholder(tf.int32, [None, path_max_size, pseq_max_len], 'pseq_ids'),
            'pseq_len': tf.placeholder(tf.int32, [None, path_max_size], 'pseq_len'),
            
            # Sentential information
            'qw_input': tf.placeholder(tf.int32, [None, path_max_size, qw_max_len], 'qw_input'),
            'qw_len': tf.placeholder(tf.int32, [None, path_max_size], 'qw_len'),
            
            # Dependency information
            'dep_input': tf.placeholder(tf.int32, [None, path_max_size, qw_max_len], 'dep_input'),
            'dep_len': tf.placeholder(tf.int32, [None, path_max_size], 'dep_len')
        }
        return input_tensor_dict

    def build_sent(self):
        with tf.device('/cpu:0'):
            qw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                            ids=self.input_tensor_dict['qw_input'],
                                            name='qw_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
            dep_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                             ids=self.input_tensor_dict['dep_input'],
                                             name='dep_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
        qw_len = self.input_tensor_dict['qw_len']
        dep_len = self.input_tensor_dict['dep_len']
        qw_emb = self.dropout_layer(qw_emb, training=False)
        dep_emb = self.dropout_layer(dep_emb, training=False)

        encoder_args = {'config': self.rnn_config, 'mode': tf.contrib.learn.ModeKeys.INFER}
        rnn_encoder = BidirectionalRNNEncoder(**encoder_args)

        with tf.variable_scope('rm_task', reuse=tf.AUTO_REUSE):
            # Build question representation
            qw_repr = self.build_question_repr(seq_emb=qw_emb, seq_len=qw_len, path_repr=None,
                                               rnn_encoder=rnn_encoder, scope_name='qw_repr')
            dep_repr = self.build_question_repr(seq_emb=dep_emb, seq_len=dep_len, path_repr=None,
                                                rnn_encoder=rnn_encoder, scope_name='dep_repr')
            sent_repr = self.build_sent_repr(qw_repr=qw_repr, dep_repr=dep_repr)
        # Ready to return
        tensor_dict = {'sent_repr': sent_repr}
        return tensor_dict

    def build_graph(self, mode_str):
        LogInfo.begin_track('Build graph: [MT-%s]', mode_str)
        mode = tf.contrib.learn.ModeKeys.INFER if mode_str == 'eval' else tf.contrib.learn.ModeKeys.TRAIN
        training = False if mode_str == 'eval' else True

        with tf.device('/cpu:0'):
            qw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                            ids=self.input_tensor_dict['qw_input'],
                                            name='qw_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
            dep_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                             ids=self.input_tensor_dict['dep_input'],
                                             name='dep_emb')  # (ds, path_max_size, qw_max_len, dim_emb)
            pw_emb = tf.nn.embedding_lookup(params=self.w_embedding,
                                            ids=self.input_tensor_dict['pw_input'],
                                            name='pw_emb')  # (ds, path_max_size, pw_max_len, dim_emb)
            pseq_emb = tf.nn.embedding_lookup(params=self.m_embedding,
                                              ids=self.input_tensor_dict['pseq_ids'],
                                              name='pseq_emb')  # (ds, path_max_size, pseq_max_size, dim_emb)
            path_emb = tf.nn.embedding_lookup(params=self.p_embedding,
                                              ids=self.input_tensor_dict['path_ids'],
                                              name='path_emb')  # (ds, path_max_size, dim_emb)
        pw_len = self.input_tensor_dict['pw_len']
        pseq_len = self.input_tensor_dict['pseq_len']
        qw_len = self.input_tensor_dict['qw_len']
        dep_len = self.input_tensor_dict['dep_len']

        qw_emb = self.dropout_layer(qw_emb, training=training)
        dep_emb = self.dropout_layer(dep_emb, training=training)
        pw_emb = self.dropout_layer(pw_emb, training=training)
        pseq_emb = self.dropout_layer(pseq_emb, training=training)
        path_emb = self.dropout_layer(path_emb, training=training)

        encoder_args = {'config': self.rnn_config,
                        'mode': mode}
        rnn_encoder = BidirectionalRNNEncoder(**encoder_args)

        # For RM kernel
        with tf.variable_scope('rm_task', reuse=tf.AUTO_REUSE):
            # Build semantic component representation
            path_repr = self.build_path_repr(pw_emb=pw_emb, pw_len=pw_len,
                                             pseq_emb=pseq_emb, pseq_len=pseq_len,
                                             path_emb=path_emb, rnn_encoder=rnn_encoder)
            # Build question representation
            qw_repr = self.build_question_repr(seq_emb=qw_emb, seq_len=qw_len, path_repr=path_repr,
                                               rnn_encoder=rnn_encoder, scope_name='qw_repr')
            dep_repr = self.build_question_repr(seq_emb=dep_emb, seq_len=dep_len, path_repr=path_repr,
                                                rnn_encoder=rnn_encoder, scope_name='dep_repr')
            sent_repr = self.build_sent_repr(qw_repr=qw_repr, dep_repr=dep_repr)

            rm_final_feats, rm_score = self.rm_forward(path_repr=path_repr, sent_repr=sent_repr,
                                                       path_size=self.input_tensor_dict['path_size'])

        # Ready to return
        tensor_dict = {'sent_repr': sent_repr,
                       'rm_score': rm_score,
                       'rm_final_feats': rm_final_feats}
        LogInfo.logs('%d tensors saved and return: %s', len(tensor_dict), tensor_dict.keys())
        LogInfo.end_track()
        return tensor_dict

    def build_path_repr(self, pw_emb, pw_len, path_emb, pseq_emb, pseq_len, rnn_encoder):
        """
        :param pw_emb: (ds, path_max_size, pw_max_len, dim_emb)
        :param pw_len: (ds, path_max_size)
        :param path_emb: (ds, path_max_size, dim_emb)
        :param pseq_emb: (ds, path_max_size, pseq_max_len, dim_emb)
        :param pseq_len: (ds, path_max_size)
        :param rnn_encoder:
        """
        LogInfo.logs('build_path_repr: path_usage = [%s].', self.path_usage)
        assert len(self.path_usage) == 2
        # Word level representation
        pw_repr = self._build_path_repr_pw_side(
            pw_emb=pw_emb, pw_len=pw_len,
            rnn_encoder=rnn_encoder,
            pw_usage=self.path_usage[0]
        )
        # Path level representation
        pseq_repr = self._build_path_repr_pseq_side(
            path_emb=path_emb, pseq_emb=pseq_emb, pseq_len=pseq_len,
            rnn_encoder=rnn_encoder, pseq_usage=self.path_usage[1]
        )
        if pw_repr is None:
            assert pseq_repr is not None
            final_repr = pseq_repr
        elif pseq_repr is None:
            final_repr = pw_repr
        else:   # summation
            final_repr = pw_repr + pseq_repr
        return final_repr       # (ds, path_max_size, dim_emb or dim_hidden)

    def _build_path_repr_pw_side(self, pw_emb, pw_len, rnn_encoder, pw_usage):
        """
        :param pw_emb: (ds, path_max_size, pw_max_len, dim_wd_emb)
        :param pw_len: (ds, path_max_size)
        :param rnn_encoder:
        :param pw_usage: X,B,R (None / BOW / RNN)
        """
        with tf.variable_scope('pw_repr', reuse=tf.AUTO_REUSE):
            pw_emb = tf.reshape(pw_emb, [-1, self.pw_max_len, self.dim_emb])
            pw_len = tf.reshape(pw_len, [-1])
            if pw_usage == 'B':
                pw_repr = seq_hidden_averaging(seq_hidden_input=pw_emb, len_input=pw_len)
                # (ds*path_max_size, dim_wd_emb), simply BOW
                pw_repr = tf.reshape(pw_repr, [-1, self.path_max_size, self.dim_emb], 'pw_repr')
            elif pw_usage == 'R':
                pw_repr = seq_encoding_with_aggregation(
                    emb_input=pw_emb, len_input=pw_len,
                    rnn_encoder=rnn_encoder,
                    seq_merge_mode=self.seq_merge_mode
                )   # (ds*path_max_size, dim_hidden)
                pw_repr = tf.reshape(pw_repr, [-1, self.path_max_size, self.dim_hidden], 'pw_repr')
                # (ds, path_max_size, dim_qw_hidden)
            else:
                assert pw_usage == 'X'
                pw_repr = None
        if pw_repr is not None:
            self.show_tensor(pw_repr)
        return pw_repr

    def _build_path_repr_pseq_side(self, path_emb, pseq_emb, pseq_len, rnn_encoder, pseq_usage):
        """
        :param path_emb: (ds, path_max_size, dim_emb)
        :param pseq_emb: (ds, path_max_size, pseq_max_len, dim_emb)
        :param pseq_len: (ds, path_max_size)
        :param rnn_encoder:
        :param pseq_usage: X,B,R,H (None / BOW / RNN / wHole)
        """
        with tf.variable_scope('pseq_repr', reuse=tf.AUTO_REUSE):
            pseq_emb = tf.reshape(pseq_emb, [-1, self.pseq_max_len, self.dim_emb])
            pseq_len = tf.reshape(pseq_len, [-1])
            if pseq_usage == 'H':
                pseq_repr = path_emb        # (ds, path_max_size, dim_emb)
            elif pseq_usage == 'B':
                pseq_repr = seq_hidden_averaging(seq_hidden_input=pseq_emb, len_input=pseq_len)
                pseq_repr = tf.reshape(pseq_repr, [-1, self.path_max_size, self.dim_emb], 'pseq_repr')
                # (ds, path_max_size, dim_wd_emb)
            elif pseq_usage == 'R':
                pseq_repr = seq_encoding_with_aggregation(
                    emb_input=pseq_emb, len_input=pseq_len,
                    rnn_encoder=rnn_encoder,
                    seq_merge_mode=self.seq_merge_mode
                )   # (ds*path_max_size, dim_hidden)
                pseq_repr = tf.reshape(pseq_repr, [-1, self.path_max_size, self.dim_hidden], 'pseq_repr')
                # (ds, path_max_size, dim_hidden)
            else:
                assert pseq_usage == 'X'
                pseq_repr = None
        if pseq_repr is not None:
            self.show_tensor(pseq_repr)
        return pseq_repr

    def build_question_repr(self, seq_emb, seq_len, path_repr, rnn_encoder, scope_name):
        """
        :param seq_emb: (ds, path_max_size, qw_max_len, dim_wd_emb)
        :param seq_len: (ds, path_max_size)
        :param path_repr: (ds, path_max_size, dim_path_hidden)
        :param rnn_encoder: RNN encoder (could be None)
        :param scope_name: variable_scope name
        """
        seq_emb = tf.reshape(seq_emb, [-1, self.qw_max_len, self.dim_emb])
        seq_len = tf.reshape(seq_len, [-1])

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            if self.att_config is not None:
                dim_att_hidden = self.att_config['dim_att_hidden']
                att_func = self.att_config['att_func']
                dim_path_hidden = path_repr.get_shape().as_list()[-1]
                path_repr = tf.reshape(path_repr, [-1, dim_path_hidden])
                LogInfo.logs('build_seq_repr: att_func = [%s].', att_func)

                seq_hidden = seq_encoding(emb_input=seq_emb, len_input=seq_len, encoder=rnn_encoder)
                # (ds*path_max_size, qw_max_len, dim_hidden)
                seq_mask = tf.sequence_mask(lengths=seq_len,
                                            maxlen=self.qw_max_len,
                                            dtype=tf.float32,
                                            name='seq_mask')  # (ds*path_max_size, qw_max_len)
                simple_att = Attention(lf_max_len=self.qw_max_len,
                                       dim_att_hidden=dim_att_hidden,
                                       att_func=att_func)
                seq_att_rep, att_mat, seq_weight = simple_att.forward(lf_input=seq_hidden,
                                                                      lf_mask=seq_mask,
                                                                      fix_rt_input=path_repr)
                seq_repr = seq_att_rep
            else:  # no attention, similar with above
                LogInfo.logs('build_seq_repr: att_func = [noAtt], seq_merge_mode = [%s].', self.seq_merge_mode)
                seq_repr = seq_encoding_with_aggregation(emb_input=seq_emb, len_input=seq_len,
                                                         rnn_encoder=rnn_encoder,
                                                         seq_merge_mode=self.seq_merge_mode)
            seq_repr = tf.reshape(seq_repr, [-1, self.path_max_size, self.dim_hidden], 'seq_repr')
        return seq_repr     # (ds, path_max_size, dim_hidden)

    def build_sent_repr(self, qw_repr, dep_repr):
        """
        :param qw_repr:   (ds, path_max_len, dim_hidden)
        :param dep_repr:  (ds, path_max_len, dim_hidden)
        """
        LogInfo.logs('build_sent_repr: sent_usage = [%s].', self.sent_usage)
        if self.sent_usage == 'qwOnly':
            sent_repr = qw_repr
        elif self.sent_usage == 'depOnly':
            sent_repr = dep_repr
        elif self.sent_usage == 'mSum':  # merge by summation
            sent_repr = tf.add(qw_repr, dep_repr)
        elif self.sent_usage == 'mMax':  # merge by max pooling
            sent_repr = tf.reduce_max(tf.stack([qw_repr, dep_repr], axis=0), axis=0)
        else:
            concat_repr = tf.concat([qw_repr, dep_repr], axis=-1, name='concat_repr')
            # (ds, path_max_len, 2*dim_hidden)
            sent_repr = tf.contrib.layers.fully_connected(
                inputs=concat_repr,
                num_outputs=self.dim_hidden,
                activation_fn=tf.nn.relu,
                scope='mFC',
                reuse=tf.AUTO_REUSE
            )  # (ds, path_max_len, dim_hidden)
        return sent_repr

    def rm_forward(self, path_repr, sent_repr, path_size):
        """
        Kernel part of rm_forward.
        :param path_repr: (ds, path_max_len, dim_path_hidden)
        :param sent_repr: (ds, path_max_len, dim_hidden)
        :param path_size: (ds, )
        """
        LogInfo.logs('rm_forward: scoring_mode = [%s], final_func = [%s].', self.scoring_mode, self.final_func)

        with tf.variable_scope('rm_forward', reuse=tf.AUTO_REUSE):
            dim_path_hidden = path_repr.get_shape().as_list()[-1]
            assert self.scoring_mode in ('separated', 'compact')
            if self.scoring_mode == 'compact':
                sent_repr = seq_hidden_max_pooling(seq_hidden_input=sent_repr, len_input=path_size)
                path_repr = seq_hidden_max_pooling(seq_hidden_input=path_repr, len_input=path_size)
                # (ds, dim_xx_hidden)
            else:
                sent_repr = tf.reshape(sent_repr, [-1, self.dim_hidden])
                path_repr = tf.reshape(path_repr, [-1, dim_path_hidden])
                # (ds*path_max_size, dim_xx_hidden)

            # Apply final scoring functions
            if self.final_func == 'dot':
                assert dim_path_hidden == self.dim_hidden
                merge_score = tf.reduce_sum(sent_repr * path_repr, axis=-1, name='merge_score')
            elif self.final_func == 'cos':
                assert dim_path_hidden == self.dim_hidden
                merge_score = cosine_sim(lf_input=sent_repr, rt_input=path_repr)
            elif self.final_func == 'bilinear':
                bilinear_mat = tf.get_variable(name='bilinear_mat',
                                               shape=[dim_path_hidden, self.dim_hidden],
                                               dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
                proj_repr = tf.matmul(path_repr, bilinear_mat, name='proj_repr')
                merge_score = tf.reduce_sum(sent_repr * proj_repr, axis=-1, name='merge_score')
            else:
                assert self.final_func.startswith('fc')
                hidden_size = int(self.final_func[2:])
                concat_repr = tf.concat([sent_repr, path_repr], axis=-1, name='concat_repr')
                concat_hidden = tf.contrib.layers.fully_connected(
                    inputs=concat_repr,
                    num_outputs=hidden_size,
                    activation_fn=tf.nn.relu,
                    scope='fc1',
                    reuse=tf.AUTO_REUSE
                )  # (ds / ds*path_max_len, 32)
                merge_score = tf.contrib.layers.fully_connected(
                    inputs=concat_hidden,
                    num_outputs=1,
                    activation_fn=None,
                    scope='fc2',
                    reuse=tf.AUTO_REUSE
                )  # (ds / ds*path_max_len, 1)
                merge_score = tf.squeeze(merge_score, axis=-1, name='merge_score')

            if self.scoring_mode == 'compact':
                rm_score = merge_score
                rm_final_feats = tf.expand_dims(rm_score, -1, 'rm_final_feats')  # (ds, 1)
            else:
                merge_score = tf.reshape(merge_score, [-1, self.path_max_size])  # (ds, path_max_size)
                path_mask = tf.sequence_mask(
                    lengths=path_size, maxlen=self.path_max_size,
                    dtype=tf.float32, name='path_mask'
                )  # (ds, path_max_size) as mask
                rm_score = tf.reduce_sum(merge_score * path_mask, axis=-1, name='rm_score')  # (ds, )
                rm_final_feats = tf.expand_dims(rm_score, -1, 'rm_final_feats')  # (ds, 1)

        return rm_final_feats, rm_score

    def get_pair_loss(self, optm_score):
        """
        :param optm_score:  (ds, )
        in TRAIN mode, we put positive and negative cases together into one tensor
        """
        pos_score, neg_score = tf.unstack(tf.reshape(optm_score, shape=[-1, 2]), axis=1)
        margin_loss = tf.nn.relu(neg_score + self.margin - pos_score, name='margin_loss')
        return margin_loss

    def build_update_summary(self, spec_var_list=None):
        collection_name = 'optm_rm'
        loss_name = 'rm_loss'
        task_loss = getattr(self, loss_name)
        tf.summary.scalar(loss_name, getattr(self, loss_name), collections=[collection_name])
        if spec_var_list is None:
            update_step = self.optimizer(self.learning_rate).minimize(task_loss)
        else:
            update_step = self.optimizer(self.learning_rate).minimize(task_loss, var_list=spec_var_list)
        optm_summary = tf.summary.merge_all(collection_name)
        return update_step, optm_summary

    def show_tensor(self, tensor, name=None):
        show_name = name or tensor.name
        LogInfo.logs('* %s --> %s | %s', show_name, tensor.get_shape().as_list(), str(tensor))

    def train_batch(self, sess, data):
        input_feed = {
            self.input_tensor_dict['path_size']: data['path_size'],
            self.input_tensor_dict['path_ids']: data['path_ids'],
            self.input_tensor_dict['pw_input']: data['pw_input'],
            self.input_tensor_dict['pw_len']: data['pw_len'],
            self.input_tensor_dict['pseq_ids']: data['pseq_ids'],
            self.input_tensor_dict['pseq_len']: data['pseq_len'],
            self.input_tensor_dict['qw_input']: data['qw_input'],
            self.input_tensor_dict['qw_len']: data['qw_len'],
            self.input_tensor_dict['dep_input']: data['dep_input'],
            self.input_tensor_dict['dep_len']: data['dep_len']
        }
        output_feed = [self.rm_update, self.rm_loss, self.optm_rm_summary]
        outputs = sess.run(output_feed, feed_dict=input_feed)
        return outputs[0], outputs[1], outputs[2]

    def eval_batch(self, sess, data):
        input_feed = {
            self.input_tensor_dict['path_size']: data['path_size'],
            self.input_tensor_dict['path_ids']: data['path_ids'],
            self.input_tensor_dict['pw_input']: data['pw_input'],
            self.input_tensor_dict['pw_len']: data['pw_len'],
            self.input_tensor_dict['pseq_ids']: data['pseq_ids'],
            self.input_tensor_dict['pseq_len']: data['pseq_len'],
            self.input_tensor_dict['qw_input']: data['qw_input'],
            self.input_tensor_dict['qw_len']: data['qw_len'],
            self.input_tensor_dict['dep_input']: data['dep_input'],
            self.input_tensor_dict['dep_len']: data['dep_len']
        }
        output_feed = [self.eval_tensor_dict['rm_score'],
                       self.eval_tensor_dict['rm_final_feats']]
        outputs = sess.run(output_feed, feed_dict=input_feed)
        return outputs[0], outputs[1]

    def transfer_encode(self, sess, data):
        input_feed = {
            self.input_tensor_dict['qw_input']: data['qw_input'],
            self.input_tensor_dict['qw_len']: data['qw_len'],
            self.input_tensor_dict['dep_input']: data['dep_input'],
            self.input_tensor_dict['dep_len']: data['dep_len']
        }
        output_feed = [self.sent_tensor_dict['sent_repr']]
        outputs = sess.run(output_feed, feed_dict=input_feed)
        return outputs[0]
