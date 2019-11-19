# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modified by Jian Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Attention-based decoder functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def create_output_fn(vocab_size):
    with variable_scope.variable_scope("output_fn") as scope:
        def output_fn(x):
            return layers.linear(x, vocab_size, scope=scope)

        return output_fn


def create_hidden_fn(num_units):
    with variable_scope.variable_scope("hidden_fn") as scope:
        def hidden_fn(x):
            return layers.linear(x, num_units, scope=scope)

        return hidden_fn


def prepare_attention(attention_states,
                      kd_states,
                      attention_option,
                      num_units,
                      reuse=False):
    # Prepare attention keys / values from attention_states
    with variable_scope.variable_scope("attn_keys", reuse=reuse) as scope:
        attention_keys = layers.linear(attention_states, num_units,
                                       biases_initializer=None, scope=scope)
        if kd_states is not None:
            attention_values = (attention_states, kd_states)
        else:
            attention_values = attention_states
        # Attention scoring function
        attention_score_fn = _create_attention_score_fn("attn_score", num_units, attention_option, reuse)

    # Attention construction function
    attention_construct_fn = _create_attention_construct_fn("attn_construct",
                                                            num_units, attention_score_fn, reuse)

    return attention_keys, attention_values, attention_construct_fn


def prepare_multistep_attention(encoder_states,
                                decoder_reprs,
                                kd_states1,
                                kd_states2,
                                attention_option,
                                num_units,
                                reuse=False):
    # Prepare attention keys / values from attention_states
    with variable_scope.variable_scope("attn_keys", reuse=reuse) as scope:
        attention_keys1 = layers.linear(encoder_states, num_units, biases_initializer=None, scope=scope)
        attention_values1 = encoder_states
        # Attention scoring function
        attention_score_fn1 = _create_attention_score_fn("attn_score", num_units,
                                                         attention_option, reuse)

    with variable_scope.variable_scope("attn_reprs", reuse=reuse) as scope:
        if decoder_reprs is not None:
            attention_keys2 = layers.linear(decoder_reprs, num_units, biases_initializer=None, scope=scope)
        else:
            attention_keys2 = None
        attention_values2 = decoder_reprs
        # Attention scoring function
        attention_score_fn2 = _create_attention_score_fn("attn_score", num_units,
                                                         attention_option, reuse)

    attention_keys = (attention_keys1, attention_keys2)
    if kd_states1 is not None and kd_states2 is not None:
        attention_values = (attention_values1, attention_values2, kd_states1, kd_states2)
    else:
        attention_values = (attention_values1, attention_values2, None, None)
    attention_score_fn = (attention_score_fn1, attention_score_fn2)

    # Attention construction function
    attention_construct_fn = _create_attention_construct_fn("attn_construct_multi",
                                                            num_units, attention_score_fn, reuse)

    return attention_keys, attention_values, attention_construct_fn


def attention_decoder_train(encoder_state,
                            attention_keys,
                            attention_values,
                            attention_construct_fn):
    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        if cell_state is None:  # first call, return encoder_state
            cell_state = encoder_state
            # init attention
            attention = _init_attention(encoder_state)
        else:  # construct attention
            attention = attention_construct_fn(cell_output, attention_keys, attention_values)
            cell_output = attention

        # combine cell_input and attention
        next_input = array_ops.concat([cell_input, attention], 1)

        return (None, cell_state, next_input, cell_output, context_state)

    return decoder_fn


def attention_decoder_inference(num_units,
                                num_decoder_symbols,
                                output_fn,
                                encoder_state,
                                attention_keys,
                                attention_values,
                                attention_construct_fn,
                                embeddings,
                                start_of_sequence_id,
                                end_of_sequence_id,
                                maximum_length,
                                dtype=dtypes.int32,
                                is_pass1=False):
    num_units = ops.convert_to_tensor(num_units, dtype)
    num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
    start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
    end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
    maximum_length = ops.convert_to_tensor(maximum_length, dtype)
    encoder_info = nest.flatten(encoder_state)[0]
    batch_size = encoder_info.get_shape()[0].value
    if batch_size is None:
        batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        if cell_input is not None:
            raise ValueError("Expected cell_input to be None, but saw: %s" % cell_input)

        if cell_output is None:
            # invariant that this is time == 0
            next_input_id = array_ops.ones([batch_size, ], dtype=dtype) * start_of_sequence_id
            done = array_ops.zeros([batch_size, ], dtype=dtypes.bool)
            cell_state = encoder_state

            if is_pass1:
                cell_output = array_ops.zeros([num_units], dtype=dtypes.float32)
            else:
                cell_output = array_ops.zeros([num_decoder_symbols], dtype=dtypes.float32)
            cell_input = array_ops.gather(embeddings, next_input_id)
            # init attention
            attention = _init_attention(encoder_state)
        else:
            # construct attention
            attention = attention_construct_fn(cell_output, attention_keys, attention_values)
            cell_output = attention

            if is_pass1:
                # argmax decoder
                cell_output_logits = output_fn(cell_output)  # logits
                next_input_id = math_ops.cast(math_ops.argmax(cell_output_logits, 1), dtype=dtype)
            else:
                # argmax decoder
                cell_output = output_fn(cell_output)  # output logits
                next_input_id = math_ops.cast(math_ops.argmax(cell_output, 1), dtype=dtype)

            done = math_ops.equal(next_input_id, end_of_sequence_id)
            cell_input = array_ops.gather(embeddings, next_input_id)

        # combine cell_input and attention
        next_input = array_ops.concat([cell_input, attention], 1)

        # if time > maxlen, return all true vector
        done = control_flow_ops.cond(math_ops.greater(time, maximum_length),
                                     lambda: array_ops.ones([batch_size, ], dtype=dtypes.bool),
                                     lambda: done)
        return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn


# Helper functions ##
def _init_attention(encoder_state):
    """Initialize attention. Handling both LSTM and GRU.
    Args:
        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
    Returns:
        attn: initial zero attention vector.
    """

    # Multi- vs single-layer
    if isinstance(encoder_state, tuple):
        top_state = encoder_state[-1]
    else:
        top_state = encoder_state

    # LSTM vs GRU
    if isinstance(top_state, rnn_cell_impl.LSTMStateTuple):
        attn = array_ops.zeros_like(top_state.h)
    else:
        attn = array_ops.zeros_like(top_state)

    return attn


def _create_attention_construct_fn(name, num_units, attention_score_fn, reuse):
    """Function to compute attention vectors.
    Args:
        name: to label variables.
        num_units: hidden state dimension.
        attention_score_fn: to compute similarity between key and target states.
        reuse: whether to reuse variable scope.
    Returns:
        attention_construct_fn: to build attention states.
    """
    with variable_scope.variable_scope(name, reuse=reuse) as scope:

        def construct_fn(attention_query, attention_keys, attention_values):
            if isinstance(attention_score_fn, tuple):  # multi-step decoding
                attention_score_fn1, attention_score_fn2 = attention_score_fn
                attention_keys1, attention_keys2 = attention_keys
                attention_values1, decoder_reprs, kd_states1, kd_states2 = attention_values
                context1 = attention_score_fn1(attention_query, attention_keys1, attention_values1)
                if kd_states1 is None or kd_states2 is None:
                    context2 = attention_score_fn2(attention_query, attention_keys2, decoder_reprs)
                    concat_input = array_ops.concat([attention_query, context1, context2], 1)
                else:
                    if decoder_reprs is None:
                        print("concat=3")
                        concat_input = array_ops.concat([attention_query, context1, kd_states1, kd_states2], 1)
                    else:
                        print("concat=4")
                        context2 = attention_score_fn2(attention_query, attention_keys2, decoder_reprs)
                        concat_input = array_ops.concat([attention_query, context1, context2, kd_states1, kd_states2], 1)
            else:  # only one step decoding
                if isinstance(attention_values, tuple):
                    attention_values1, kd_state = attention_values
                    context1 = attention_score_fn(attention_query, attention_keys, attention_values1)
                    concat_input = array_ops.concat([attention_query, context1, kd_state], 1)
                else:
                    context = attention_score_fn(attention_query, attention_keys, attention_values)
                    concat_input = array_ops.concat([attention_query, context], 1)

            attention = layers.linear(concat_input, num_units, biases_initializer=None, scope=scope)
            return attention

        return construct_fn


def _create_attention_score_fn(name, num_units, attention_option, reuse, dtype=dtypes.float32):
    """Different ways to compute attention scores.
    Args:
        name: to label variables.
        num_units: hidden state dimension.
        attention_option: how to compute attention, either "luong" or "bahdanau".
            "bahdanau": additive (Bahdanau et al., ICLR'2015)
            "luong": multiplicative (Luong et al., EMNLP'2015)
        reuse: whether to reuse variable scope.
        dtype: (default: `dtypes.float32`) data type to use.
    Returns:
        attention_score_fn: to compute similarity between key and target states.
    """
    with variable_scope.variable_scope(name, reuse=reuse):
        if attention_option == "bahdanau":
            query_w = variable_scope.get_variable("attnW", [num_units, num_units], dtype=dtype)
            score_v = variable_scope.get_variable("attnV", [num_units], dtype=dtype)

        def attention_score_fn(query, keys, values):
            """Put attention masks on attention_values using attention_keys and query.
            Args:
                query: A Tensor of shape [batch_size, num_units].
                keys: A Tensor of shape [batch_size, attention_length, num_units].
                values: A Tensor of shape [batch_size, attention_length, num_units].
            Returns:
                context_vector: A Tensor of shape [batch_size, num_units].
            Raises:
                ValueError: if attention_option is neither "luong" or "bahdanau".
            """

            if attention_option == "bahdanau":
                # transform query
                query = math_ops.matmul(query, query_w)

                # reshape query: [batch_size, 1, num_units]
                query = array_ops.reshape(query, [-1, 1, num_units])

                # attn_fun
                scores = _attn_add_fun(score_v, keys, query)
            elif attention_option == "luong":
                # reshape query: [batch_size, 1, num_units]
                query = array_ops.reshape(query, [-1, 1, num_units])

                # attn_fun
                scores = _attn_mul_fun(keys, query)
            else:
                raise ValueError("Unknown attention option %s!" % attention_option)

            # Compute alignment weights
            #     scores: [batch_size, length]
            #     alignments: [batch_size, length]
            # TODO(thangluong): not normalize over padding positions.
            alignments = nn_ops.softmax(scores)

            # Now calculate the attention-weighted vector.
            alignments = array_ops.expand_dims(alignments, 2)
            context_vector = math_ops.reduce_sum(alignments * values, [1])
            context_vector.set_shape([None, num_units])

            return context_vector

        return attention_score_fn


# keys: [batch_size, attention_length, attn_size]
# query: [batch_size, 1, attn_size]
# return weights [batch_size, attention_length]
@function.Defun(func_name="attn_add_fun", noinline=True)
def _attn_add_fun(v, keys, query):
    return math_ops.reduce_sum(v * math_ops.tanh(keys + query), [2])


@function.Defun(func_name="attn_mul_fun", noinline=True)
def _attn_mul_fun(keys, query):
    return math_ops.reduce_sum(keys * query, [2])
