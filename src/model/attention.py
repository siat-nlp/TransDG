# -*- coding: utf-8 -*-
import tensorflow as tf


def bahdanau_attention(num_units, memory, query, normalize=False):
    with tf.variable_scope("bahdanau_attention"):
        memory_layer = tf.layers.Dense(num_units, name="memory_layer", use_bias=False)
        query_layer = tf.layers.Dense(num_units, name="query_layer", use_bias=False)
        values = memory
        keys = memory_layer(values)
        processed_query = query_layer(query)
        score = _bahdanau_score(processed_query, keys, normalize=normalize)
        alignments = tf.nn.softmax(score)
        context = tf.reduce_sum(tf.expand_dims(alignments, 2) * values, axis=1)
        return context, alignments


def luong_attention(num_units, memory, query, scale=False):
    with tf.variable_scope("luong_attention"):
        memory_layer = tf.layers.Dense(num_units, name="memory_layer", use_bias=False)
        values = memory
        keys = memory_layer(values)
        score = _luong_score(query, keys, scale=scale)
        alignments = tf.nn.softmax(score)
        context = tf.reduce_sum(tf.expand_dims(alignments, 2) * values, axis=1)
        return context, alignments


def _bahdanau_score(query, keys, normalize=False):
    """Implements Bahdanau-style (additive) scoring function.
      Args:
        query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        normalize: Whether to normalize the score function.

      Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
      """
    dtype = query.dtype
    num_units = keys.shape[2].value

    # Reshape from [batch_size, num_units] to [batch_size, 1, num_units] for broadcasting.
    query = tf.expand_dims(query, 1)
    v = tf.get_variable("attention_v", [num_units], dtype=dtype)
    if normalize:
        # Scalar used in weight normalization
        g = tf.get_variable("attention_g", dtype=dtype, initializer=tf.sqrt((1. / num_units)))
        # Bias added prior to the nonlinearity
        b = tf.get_variable("attention_b", [num_units], dtype=dtype, initializer=tf.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))
        return tf.reduce_sum(normed_v * tf.tanh(keys + query + b), [2])
    else:
        return tf.reduce_sum(v * tf.tanh(keys + query), [2])


def _luong_score(query, keys, scale=False):
    """Implements Luong-style (multiplicative) scoring function.
      Args:
        query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        scale: Whether to apply a scale to the score function.

      Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
      """
    dtype = query.dtype

    # Reshape from [batch_size, num_units] to [batch_size, 1, num_units] for matmul.
    query = tf.expand_dims(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #   [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #   [batch_time, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = tf.matmul(query, keys, transpose_b=True)
    score = tf.squeeze(score, [1])

    if scale:
        # Scalar used in weight scaling
        g = tf.get_variable("attention_g", dtype=dtype, initializer=1.)
        score = g * score
    return score
