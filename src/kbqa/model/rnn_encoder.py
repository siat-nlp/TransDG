# -*- coding: utf-8 -*-
"""
Implementation of RNN-based encoder
"""

import tensorflow as tf
import copy
from collections import namedtuple

from ..utils.log_util import LogInfo


EncoderOutput = namedtuple(
    "EncoderOutput",
    ["outputs", "final_state", "attention_values", "attention_values_length"])


class UnidirectionalRNNEncoder(object):
    def __init__(self, config, mode):
        self.params = copy.deepcopy(config)
        # only keep dropout during training
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            self.params["dropout_input_keep_prob"] = 1.0
            self.params["dropout_output_keep_prob"] = 1.0
            self.params["reuse"] = True

    def encode(self, inputs, sequence_length, initial_state=None, reuse=None):
        # cell = get_rnn_cell(**self.params["rnn_cell"])
        self.params['reuse'] = reuse            # KQ: temporary solution
        cell = get_rnn_cell(**self.params)
        outputs, state = tf.contrib.rnn.static_rnn(cell=cell,
                                                   inputs=inputs,
                                                   initial_state=initial_state,
                                                   sequence_length=sequence_length,
                                                   dtype=tf.float32)

        attention_values = tf.stack(outputs, axis=1)

        return EncoderOutput(
            outputs=outputs,
            final_state=state,
            attention_values=attention_values,
            attention_values_length=sequence_length)


class BidirectionalRNNEncoder(object):
    def __init__(self, config, mode):
        self.params = copy.deepcopy(config)
        # only keep dropout during training
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            self.params['keep_prob'] = 1.0
        LogInfo.logs('Show Bi-RNN param: %s', self.params)

    def encode(self, inputs, sequence_length, reuse=None):
        self.params['reuse'] = reuse            # KQ: temporary solution
        cell_fw = get_rnn_cell(**self.params)
        cell_bw = get_rnn_cell(**self.params)

        outputs, output_state_fw, output_state_bw = \
            tf.contrib.rnn.static_bidirectional_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
                                                    sequence_length=sequence_length,
                                                    dtype=tf.float32)

        attention_values = tf.stack(outputs, axis=1)

        return EncoderOutput(
            outputs=outputs,
            final_state=(output_state_fw, output_state_bw),
            attention_values=attention_values,
            attention_values_length=sequence_length)


def get_rnn_cell(cell_class,
                 num_units,
                 num_layers=1,
                 keep_prob=1.0,
                 dropout_input_keep_prob=None,
                 dropout_output_keep_prob=None,
                 reuse=None):
    if dropout_input_keep_prob is None:
        dropout_input_keep_prob = keep_prob
    if dropout_output_keep_prob is None:
        dropout_output_keep_prob = keep_prob

    cells = []
    for _ in range(num_layers):
        cell = None
        if cell_class == 'RNN':
            cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units, reuse=reuse)
        elif cell_class == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=num_units, reuse=reuse)
        elif cell_class == 'LSTM':
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True, reuse=reuse)

        if keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                 input_keep_prob=dropout_input_keep_prob,
                                                 output_keep_prob=dropout_output_keep_prob)
        cells.append(cell)

    if len(cells) > 1:
        final_cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    else:
        final_cell = cells[0]

    return final_cell
