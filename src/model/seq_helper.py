# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, LSTMCell, RNNCell
from tensorflow.contrib.rnn import DropoutWrapper


def define_rnn_cell(cell_class, num_units, num_layers=1, keep_prob=1.0,
                    input_keep_prob=None, output_keep_prob=None):
    if input_keep_prob is None:
        input_keep_prob = keep_prob
    if output_keep_prob is None:
        output_keep_prob = keep_prob

    cells = []
    for _ in range(num_layers):
        if cell_class == 'GRU':
            cell = GRUCell(num_units=num_units)
        elif cell_class == 'LSTM':
            cell = LSTMCell(num_units=num_units)
        else:
            cell = RNNCell(num_units=num_units)

        if keep_prob < 1.0:
            cell = DropoutWrapper(cell=cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
        cells.append(cell)

    if len(cells) > 1:
        final_cell = MultiRNNCell(cells)
    else:
        final_cell = cells[0]

    return final_cell


def sequence_loss(num_symbols, output_logits, targets, masks):
    """Sequence loss"""
    logits = tf.reshape(output_logits, [-1, num_symbols])
    local_labels = tf.reshape(targets, [-1])
    local_masks = tf.reshape(masks, [-1])

    local_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=local_labels, logits=logits)
    local_loss = local_loss * local_masks

    loss = tf.reduce_sum(local_loss)
    total_size = tf.reduce_sum(local_masks)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss = loss / total_size

    return loss


def ppx_loss(num_symbols, output_logits, targets, masks):
    local_masks = tf.reshape(masks, [-1])
    one_hot_targets = tf.one_hot(targets, num_symbols)
    ppx_prob = tf.reduce_sum(tf.nn.softmax(output_logits) * one_hot_targets, axis=2)
    ppx_loss = tf.reduce_sum(tf.reshape(-tf.log(1e-12 + ppx_prob), [-1]) * local_masks)

    total_size = tf.reduce_sum(local_masks)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    ppx_loss = ppx_loss / total_size

    return ppx_loss


def sentence_ppx(num_symbols, output_logits, targets, masks):
    batch_size = tf.shape(output_logits)[0]
    local_masks = tf.reshape(masks, [-1])
    one_hot_targets = tf.one_hot(targets, num_symbols)
    ppx_prob = tf.reduce_sum(tf.nn.log_softmax(output_logits) * one_hot_targets, axis=2)
    sent_ppx = tf.reduce_sum(
        tf.reshape(tf.reshape(-ppx_prob, [-1]) * local_masks, [batch_size, -1]), axis=1)

    sent_ppx = sent_ppx / tf.reduce_sum(masks, axis=1)

    return sent_ppx
