# -*- coding: utf-8 -*-
import tensorflow as tf

from ..utils.log_util import LogInfo


class Attention:
    """
    Implementation of different attentions
    """
    def __init__(self, lf_max_len, dim_att_hidden, att_func):
        self.lf_max_len = lf_max_len
        self.dim_att_hidden = dim_att_hidden
        LogInfo.logs('Attention: lf_max_len = %d, dim_att_hidden = %d, att_func = %s.',
                     lf_max_len, dim_att_hidden, att_func)

        assert att_func in ('dot', 'bilinear', 'bahdanau', 'bdot')
        self.att_func = 'attn_%s' % att_func

    def forward(self, lf_input, lf_mask, fix_rt_input):
        """
        :param lf_input:        (ds, lf_max_len, dim_hidden)
        :param lf_mask:         (ds, lf_max_len) as float32
        :param fix_rt_input:    (ds, dim_hidden), no timestamps
        """
        rt_input = tf.expand_dims(fix_rt_input, axis=1, name='rt_input')    # (ds, 1, dim_hidden)
        with tf.variable_scope('simple_att', reuse=tf.AUTO_REUSE):
            raw_att_mat = self.att_func(lf_input=lf_input, rt_input=rt_input,
                                        lf_max_len=self.lf_max_len, rt_max_len=1,
                                        dim_att_hidden=self.dim_att_hidden)     # (ds, lf_max_len, 1)
            raw_att_mat = tf.reshape(raw_att_mat, shape=[-1, self.lf_max_len], name='raw_att_mat')
            # (ds, lf_max_len)

            masked_att_mat = raw_att_mat * lf_mask + tf.float32.min * (1. - lf_mask)
            lf_norm = tf.nn.softmax(masked_att_mat, name='lf_norm')     # (ds, lf_max_len)
            lf_weighted = tf.expand_dims(lf_norm, axis=2) * lf_input    # (ds, lf_max_len, dim_hidden)
            lf_weighted = tf.reduce_sum(lf_weighted, axis=1,
                                        name='lf_weighted')             # (ds, dim_hidden)

        return lf_weighted, raw_att_mat, lf_norm


def expand_both_dims(lf_input, rt_input, lf_max_len, rt_max_len):
    """
    lf_input:   (ds, lf_max_len, dim_emb)
    rt_input:   (ds, rt_max_len, dim_emb)
    """
    expand_lf_input = tf.stack([lf_input] * rt_max_len, axis=2,
                               name='expand_lf_input')  # (ds, lf_max_len, rt_max_len, dim_emb)
    expand_rt_input = tf.stack([rt_input] * lf_max_len, axis=1,
                               name='expand_rt_input')  # (ds, lf_max_len, rt_max_len, dim_emb)
    return expand_lf_input, expand_rt_input


def attn_dot(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    bilinear: a = x_1 . x_2
    dim_att_hidden is never used in the function.
    """
    assert dim_att_hidden is not None
    expand_lf_input, expand_rt_input = expand_both_dims(
        lf_input=lf_input, rt_input=rt_input,
        lf_max_len=lf_max_len, rt_max_len=rt_max_len
    )  # both (ds, lf_max_len, rt_max_len, dim_emb)
    att_mat = tf.reduce_sum(expand_lf_input * expand_rt_input, axis=-1,
                            name='att_mat')  # (ds, lf_max_len, rt_max_len)
    return att_mat


def attn_bilinear(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    bilinear: a = x_1 . W . x_2
    dim_att_hidden should be equal with dim_hidden,
    otherwise, matmul couldn't work properly
    """
    expand_lf_input, expand_rt_input = expand_both_dims(
        lf_input=lf_input, rt_input=rt_input,
        lf_max_len=lf_max_len, rt_max_len=rt_max_len
    )  # both (ds, lf_max_len, rt_max_len, dim_emb)
    with tf.variable_scope('cross_att_bilinear', reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='w', dtype=tf.float32,
                            shape=[dim_att_hidden, dim_att_hidden])
        att_mat = tf.reduce_sum(
            input_tensor=tf.multiply(
                x=tf.matmul(expand_lf_input, w),
                y=expand_rt_input   # both (ds, lf_max_len, rt_max_len, dim_att_hidden==dim_emb)
            ), axis=-1, name='att_mat'
        )   # (ds, lf_max_len, rt_max_len)
    return att_mat


def attn_bahdanau(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    A = u . relu(W[x_1 : x_2] + b), or understand as:
    A1 = W_1 . x_1
    A2 = W_2 . x_2
    A = u . relu(A1 + A2 + b)
    :param lf_input: (ds, lf_max_len, dim_hidden)
    :param rt_input: (ds, rt_max_len, dim_hidden)
    :param lf_max_len: int value
    :param rt_max_len: int value
    :param dim_att_hidden: the hidden dimension in the attention operation
    :return: (ds, lf_max_len, rt_max_len)
    """
    with tf.variable_scope('cross_att_bahdanau', reuse=tf.AUTO_REUSE):
        lf_att = tf.contrib.layers.fully_connected(inputs=lf_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=None,
                                                   scope='fc_lf')  # (ds, lf_max_len, dim_att_hidden)
        rt_att = tf.contrib.layers.fully_connected(inputs=rt_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=None,
                                                   scope='fc_rt')  # (ds, rt_max_len, dim_att_hidden)
        expand_lf_att, expand_rt_att = expand_both_dims(
            lf_input=lf_att, rt_input=rt_att,
            lf_max_len=lf_max_len, rt_max_len=rt_max_len
        )  # both (ds, lf_max_len, rt_max_len, dim_att_hidden)
        u = tf.get_variable(name='u', shape=[dim_att_hidden], dtype=tf.float32)
        b = tf.get_variable(name='b', shape=[dim_att_hidden], dtype=tf.float32)
        activate = tf.nn.relu(expand_lf_att + expand_rt_att + b)
        att_mat = tf.reduce_sum(activate * u, axis=-1, name='att_mat')      # (ds, lf_max_len, rt_max_len)
    return att_mat


def attn_bdot(lf_input, rt_input, lf_max_len, rt_max_len, dim_att_hidden):
    """
    bahdanau-dot: t_i = relu(Wx_i + b), a = t_1 . t_2
    Check AF-attention paper, formula (1)
    """
    with tf.variable_scope('cross_att_bahdanau_dot', reuse=tf.AUTO_REUSE):
        lf_att = tf.contrib.layers.fully_connected(inputs=lf_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=tf.nn.relu,
                                                   scope='fc_lf')  # (ds, lf_max_len, dim_att_hidden)
        rt_att = tf.contrib.layers.fully_connected(inputs=rt_input,
                                                   num_outputs=dim_att_hidden,
                                                   activation_fn=tf.nn.relu,
                                                   scope='fc_rt')  # (ds, rt_max_len, dim_att_hidden)
    expand_lf_att, expand_rt_att = expand_both_dims(
        lf_input=lf_att, rt_input=rt_att,
        lf_max_len=lf_max_len, rt_max_len=rt_max_len
    )  # both (ds, lf_max_len, rt_max_len, dim_att_hidden)
    att_mat = tf.reduce_sum(expand_lf_att * expand_rt_att, axis=-1,
                            name='att_mat')     # (ds, lf_max_len, rt_max_len)
    return att_mat
