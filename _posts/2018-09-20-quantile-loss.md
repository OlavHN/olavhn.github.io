---
layout: post
title: "Quantile losses for modeling distributions in neural networks"
date: 2018-09-20
---

Modeling entire distributions instead of just moments like e.g. the mean or median has been a very successful technique in DeepMinds reinforcement learning benchmarks.

Specifically, they've modeled the reward distribution, instead of the usual mean, and then hypothesized that the extra modeling done by the agent leads to better data efficiency and faster learning.

The success has also lead to increasingly more sophisticated techniques of modeling distributions. Starting with the C51 algorithm, improving it with quantile regressions and finally improving on that again with Implicit Quantile Networks (IQN).

This post will (very informally) explain the three algorithms outside the context of reinforcement learning and provide example implementations in tensorflow.

For some more intuition before jumping into the papers themselves I recommend [this](https://mtomassoli.github.io/2017/12/08/distributional_rl/) blog post. It includes some great visualizations.

## C51

Given a regression problem we can estimate the mean by minimizing a squared difference between the model estimate and target variables, or we can estimate the median by minimizing the absolute difference. If instead we'd like to estimate the entire distribution of the target variable one approach could be the C51 algorithm presented in [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887).

Essentially, the the distribution is discretized over 51 points (called atoms) uniformly spaced over the range of values the target variable can take. During training the target variable is projected onto these points by putting weight on the two closest neighbouring points weighted by how close they are to the target variable. The model is then trained to estimate the same weighting by minimizing the cross entropy between the models estimate and the target variable projected onto the atoms.

Say we've got 10 atoms uniformly spaced from 10, 20, .. , 100 (It's called C51 as 51 atoms worked best on DeepMinds benchmark task) and during training we get a target variable with the value 12.5. First we project 12.5 onto the two closest atoms (75% weight on the atom representing the value 10, and 25% weight on the atom for 20), and then we use that as a training target with a cross entropy loss.

### Code

```
NUM_ATOMS = 51
MAX_VAL = 1000000

y = tf.placeholder([None, 1], tf.float32)  # Target variable
# Assume `y_hat` with shape [None, NUM_ATOMS] from model

def loss_fn(y, y_hat):
    atoms = tf.range(NUM_ATOMS, dtype=tf.float32) * MAX_VAL / NUM_ATOMS

    lower_idx = tf.minimum(tf.floor(y / MAX_VAL * NUM_ATOMS), NUM_ATOMS - 1)
    upper_idx = tf.minimum(tf.ceil(y / MAX_VAL * NUM_ATOMS), NUM_ATOMS - 1)

    lower_val = tf.gather(atoms, tf.to_int64(lower_idx))
    upper_val = tf.gather(atoms, tf.to_int64(upper_idx))

    lower_diff = tf.maximum(y - lower_val, 1.)
    upper_diff = tf.maximum(upper_val - y, 1.)

    lower_weight = upper_diff / (lower_diff + upper_diff)
    upper_weight = lower_diff / (lower_diff + upper_diff)

    batch_index = tf.expand_dims(tf.range(tf.shape(y, out_type=tf.int64)[0]), -1)
    lower_idx_idx = tf.concat([batch_index, tf.to_int64(lower_idx)], 1)
    upper_idx_idx = tf.concat([batch_index, tf.to_int64(upper_idx)], 1)
    indices = tf.concat([lower_idx_idx, upper_idx_idx], 0)
    updates = tf.squeeze(tf.concat([lower_weight, upper_weight], 0), -1)

    y_categorical = tf.scatter_nd(
            indices,
            updates,
            tf.shape(y_hat, out_type=tf.int64),
    )

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_categorical, logits=y_hat)

    return loss
```

The projection step is a bit messy, and it could probably be formulated more elegantly, but it does seem to work.

## Quantile loss
C51 works, but there are a couple of drawbacks. First the support of the atoms must be set in advance, so if the target variable has values outside the support of the weighted atoms it won't work. In the implementation above, the target value is assumed to be positive and clamped to the value of the upper atom. Second, the cross entropy loss between the discretized atoms isn't a very good metric for the actual distance between distributions. As an example we can see that the cross entropy loss between two neighbouring atoms is just a large as the two atoms furthest apart. I.e. in our previous example if the target variable is 12.5, the loss would be just a large if the model put all weight on the atom representing 30 as if it put all weight on the atom for 100. Finally, getting the projection step right in tensorflow is quite error prone.

All of these drawbacks are solved by the approach presented in the paper [Quantile regression, Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044). Instead of predicting discretized target values the algorithm does several quantile regressions. One for each quantile we're interested in. Given enough quantiles these can be seen as a discretization of the inverse cumulative distribution function. The inverse CDF is just another form of the full distribution, so we're done, and the technique it much easier to both conceptualize and implement.

Regressing on the quantiles is a much better metric for estimating the difference between two distributions. This technique is closer to the true earth mover distance, except for only evaluating at fixed, discrete points.

There is also a small trick done in the formulation of the quantile regression where the absolute loss is swapped with a huber loss to make the gradient smooth around the target value.

## Code

```
NUM_TARGETS = 19

# Assume `y_hat` of shape [None, NUM_TARGETS]

y = tf.placeholder([None, 1], tf.float32)

def loss_fn(y, y_hat):
    y = tf.tile(tf.reshape(y, [-1, 1]), [1, NUM_TARGETS])

    quantiles = tf.reshape(tf.linspace(.05, .95, NUM_TARGETS), [1, -1])
    error = tf.losses.huber_loss(
            labels=y,
            predictions=y_hat,
            delta=1.0,
            reduction=tf.losses.Reduction.NONE,
    )
    error = error * tf.sign(y - y_hat)
    loss = tf.reduce_mean(tf.maximum(quantiles * error, (quantiles - 1.) * error))

    return loss
```

The implementation is also simpler and easier to follow.

## Implicit Quantile Networks
This summer Dabney et al published a refinement of the quantile regression technique in [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923).

Instead of defining a fixed set of quantile targets in advance, the quantiles to be estimated are parameterised. Instead of setting them as hyperparameters they are inputs to the model. During training one can randomly sample them, with the result that the model learns to estimate all quantiles.

This reformulation leads to surprisingly more data efficient reinforcement learning agents. It also allows for clever exploration schemes specific to the reinforcement learning context.

Their specific implementation has a base model output an embedding, that embedding is mixed with a separate quantile embedding and finally the resulting mixed embedding is projected into a target estimate. While most ways of structuring the quantile network converge, the method they report involves multiplying the two embeddings together.

The quantile embeddings are also generated with a technique I'm not familiar with, where each dimension in the embedding is multiplied with a cosine function of increasingly shorter periods.

## Code
```
NUM_SAMPLES = 10
EMBEDDING_SIZE = 64

# At test time these variables can be overridden to get preferred quantiles
tau = tf.placeholder_with_default([NUM_SAMPLES], [None])
tau_sample = tf.random_uniform(tau, maxval=1.)

def implicit_quantile(tau, n_embedding=EMBEDDING_SIZE):
    i = tf.expand_dims(tf.range(n_embedding, dtype=tf.float32), 0)
    tau = tf.expand_dims(tau, -1)

    projection = tf.get_variable('quantile_proj', [n_embedding, n_embedding])
    bias = tf.get_variable('quantile_proj_bias', [1, n_embedding])

    tau_embedding = tf.nn.relu(
            tf.matmul(tf.cos(math.pi * i * tau), projection) + bias)

    return tau_embedding

def loss_fn(tau, tau_sample, y, y_hat):
    tau_embedding = implicit_quantile(tau_sample)  # n_samples, n_embedding

    y_hat = tf.expand_dims(y_hat, 1) * tf.expand_dims(
            tau_embedding, 0)  # batch, n_samples, n_embedding

    y_hat = tf.layers.dense(y_hat, 1)

    y = tf.tile(tf.reshape(y, [-1, 1, 1]), [1, tf.size(tau_sample), 1])

    quantiles = tf.expand_dims(tau_sample, 0)

    error = tf.losses.huber_loss(
            labels=y,
            predictions=y_hat,
            delta=1000.0,
            reduction=tf.losses.Reduction.NONE,
    )
    error = error * tf.sign(y - y_hat)
    loss = tf.reduce_mean(tf.maximum(quantiles * error, (quantiles - 1.) * error))

    return loss
```
