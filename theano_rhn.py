from __future__ import absolute_import, division, print_function

import numbers
import cPickle

import numpy as np

import theano
import theano.tensor as tt
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


floatX = theano.config.floatX

def cast_floatX(n):
  return np.asarray(n, dtype=floatX)


class Model(object):
  
  def __init__(self, config):

    self._params = []                                             # shared variables for learned parameters
    self._sticky_hidden_states = []                               # shared variables which are reset before each epoch
    self._np_rng = np.random.RandomState(config.seed // 2 + 123)
    self._theano_rng = RandomStreams(config.seed // 2 + 321)      # generates random numbers directly on GPU
    self._init_scale = config.init_scale
    self._is_training = tt.iscalar('is_training')
    self._lr = theano.shared(cast_floatX(config.learning_rate), 'lr')

    input_data = tt.imatrix('input_data')     # (batch_size, num_steps)
    targets = tt.imatrix('targets')           # (batch_size, num_steps)
    noise_x = tt.matrix('noise_x')            # (batch_size, num_steps)

    # Embed input words and apply variational dropout (for each sample, the embedding of
    # a dropped word-type consists of all zeros at all occurrences of word-type in sample).
    embedding = self.make_param((config.vocab_size, config.hidden_size), 'uniform')
    inputs = embedding[input_data.T]          # (num_steps, batch_size, hidden_size)
    inputs = self.apply_dropout(inputs, tt.shape_padright(noise_x.T))

    rhn_updates = []
    for _ in range(config.num_layers):
      # y shape: (num_steps, batch_size, hidden_size)
      y, sticky_state_updates = self.RHNLayer(
        inputs,
        config.depth, config.batch_size, config.hidden_size,
        config.drop_i, config.drop_s,
        config.init_T_bias, config.init_other_bias,
        config.tied_noise)
      rhn_updates += sticky_state_updates
      inputs = y

    noise_o = self.get_dropout_noise((config.batch_size, config.hidden_size), config.drop_o)
    outputs = self.apply_dropout(y, tt.shape_padleft(noise_o))               # (num_steps, batch_size, hidden_size)
      
    # logits
    softmax_w = embedding.T if config.tied_embeddings else self.make_param((config.hidden_size, config.vocab_size), 'uniform')
    softmax_b = self.make_param((config.vocab_size,), config.init_other_bias)
    logits = tt.dot(outputs, softmax_w) + softmax_b                          # (num_steps, batch_size, vocab_size)

    # probabilities and prediction loss
    flat_logits = logits.reshape((config.batch_size * config.num_steps, config.vocab_size))
    flat_probs = tt.nnet.softmax(flat_logits)
    flat_targets = targets.T.flatten()                                       # (batch_size * num_steps,)
    xentropies = tt.nnet.categorical_crossentropy(flat_probs, flat_targets)  # (batch_size * num_steps,)
    pred_loss = xentropies.sum() / config.batch_size

    # weight decay
    l2_loss = 0.5 * tt.sum(tt.stack([tt.sum(p**2) for p in self._params]))

    loss = pred_loss + config.weight_decay * l2_loss
    grads = theano.grad(loss, self._params)

    # gradient clipping
    global_grad_norm = tt.sqrt(tt.sum(tt.stack([tt.sum(g**2) for g in grads])))
    clip_factor = ifelse(global_grad_norm < config.max_grad_norm,
      cast_floatX(1),
      tt.cast(config.max_grad_norm / global_grad_norm, floatX))

    param_updates = [(p, p - self._lr * clip_factor * g) for p, g in zip(self._params, grads)]

    self.train = theano.function(
      [input_data, targets, noise_x],
      loss,
      givens = {self._is_training: np.int32(1)},
      updates = rhn_updates + param_updates)

    self.evaluate = theano.function(
      [input_data, targets],
      loss,
      # Note that noise_x is unused in computation graph of this function since _is_training is false.
      givens = {self._is_training: np.int32(0), noise_x: tt.zeros((config.batch_size, config.num_steps))},
      updates = rhn_updates)

    self._num_params = np.sum([param.get_value().size for param in self._params])

    if config.load_model:
      self.load(config.load_model)


  @property
  def lr(self):
    return self._lr.get_value()

  @property
  def num_params(self):
    return self._num_params


  def make_param(self, shape, init_scheme):
    """Create Theano shared variables, which are used as trainable model parameters."""
    if isinstance(init_scheme, numbers.Number):
      init_value = np.full(shape, init_scheme, floatX)
    elif init_scheme == 'uniform':
      init_value = self._np_rng.uniform(low=-self._init_scale, high=self._init_scale, size=shape).astype(floatX)
    else:
      raise AssertionError('unsupported init_scheme')
    p = theano.shared(init_value)
    self._params.append(p)
    return p

  def apply_dropout(self, x, noise):
    return ifelse(self._is_training, noise * x, x)

  def get_dropout_noise(self, shape, dropout_p):
    keep_p = 1 - dropout_p
    noise = cast_floatX(1. / keep_p) * self._theano_rng.binomial(size=shape, p=keep_p, n=1, dtype=floatX)
    return noise
    
  def assign_lr(self, lr):
    self._lr.set_value(cast_floatX(lr))

  def reset_hidden_state(self):
    for sticky_hidden_state in self._sticky_hidden_states:
      sticky_hidden_state.set_value(np.zeros_like(sticky_hidden_state.get_value()))

  def save(self, save_path):
    with open(save_path, 'wb') as f:
      for p in self._params:
        cPickle.dump(p.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
    
  def load(self, load_path):
    with open(load_path, 'rb') as f:
      for p in self._params:
        p.set_value(cPickle.load(f))


  def linear(self, x, in_size, out_size, bias, bias_init=None):
    assert bias == (bias_init is not None)
    w = self.make_param((in_size, out_size), 'uniform')
    y = tt.dot(x, w)
    if bias:
      b = self.make_param((out_size,), bias_init)
      y += b
    return y


  def RHNLayer(self, inputs, depth, batch_size, hidden_size, drop_i, drop_s, init_T_bias, init_H_bias, tied_noise):
    """Variational Recurrent Highway Layer (Theano implementation).

    References:
      Zilly, J, Srivastava, R, Koutnik, J, Schmidhuber, J., "Recurrent Highway Networks", 2016
    Args:
      inputs: Theano variable, shape (num_steps, batch_size, hidden_size).
      depth: int, the number of RHN inner layers i.e. the number of micro-timesteps per timestep.
      drop_i: float, probability of dropout over inputs.
      drop_s: float, probability of dropout over recurrent hidden state.
      init_T_bias: a valid bias_init argument for linear(), initialization of bias of transform gate T.
      init_H_bias: a valid bias_init argument for linear(), initialization of bias of non-linearity H.
      tied_noise: boolean, whether to use the same dropout masks when calculating H and when calculating T.
    Returns:
      y: Theano variable, recurrent hidden states at each timestep. Shape (num_steps, batch_size, hidden_size).
      sticky_state_updates: a list of (shared variable, new shared variable value).
    """
    # We first compute the linear transformation of the inputs over all timesteps.
    # This is done outside of scan() in order to speed up computation.
    # The result is then fed into scan()'s step function, one timestep at a time.
    noise_i_for_H = self.get_dropout_noise((batch_size, hidden_size), drop_i)
    noise_i_for_T = self.get_dropout_noise((batch_size, hidden_size), drop_i) if not tied_noise else noise_i_for_H

    i_for_H = self.apply_dropout(inputs, noise_i_for_H)
    i_for_T = self.apply_dropout(inputs, noise_i_for_T)

    i_for_H = self.linear(i_for_H, in_size=hidden_size, out_size=hidden_size, bias=True, bias_init=init_H_bias)
    i_for_T = self.linear(i_for_T, in_size=hidden_size, out_size=hidden_size, bias=True, bias_init=init_T_bias)

    # Dropout noise for recurrent hidden state.
    noise_s = self.get_dropout_noise((batch_size, hidden_size), drop_s)
    if not tied_noise:
      noise_s = tt.stack(noise_s, self.get_dropout_noise((batch_size, hidden_size), drop_s))


    def step_fn(i_for_H_t, i_for_T_t, y_tm1, noise_s):
      """
      Args:
        Elements of sequences given to scan():
          i_for_H_t: linear trans. of inputs for calculating non-linearity H at timestep t. Shape (batch_size, hidden_size).
          i_for_T_t: linear trans. of inputs for calculating transform gate T at timestep t. Shape (batch_size, hidden_size).
        Result of previous step function invocation (equals the outputs_info given to scan() on first timestep):
          y_tm1: Shape (batch_size, hidden_size).
        Non-sequences given to scan() (these are the same at all timesteps):
          noise_s: (batch_size, hidden_size) or (2, batch_size, hidden_size), depending on value of tied_noise.
      """
      tanh, sigm = tt.tanh, tt.nnet.sigmoid
      noise_s_for_H = noise_s if tied_noise else noise_s[0]
      noise_s_for_T = noise_s if tied_noise else noise_s[1]

      s_lm1 = y_tm1
      for l in range(depth):
        s_lm1_for_H = self.apply_dropout(s_lm1, noise_s_for_H)
        s_lm1_for_T = self.apply_dropout(s_lm1, noise_s_for_T)
        if l == 0:
          # On the first micro-timestep of each timestep we already have bias
          # terms summed into i_for_H_t and into i_for_T_t.
          H = tanh(i_for_H_t + self.linear(s_lm1_for_H, in_size=hidden_size, out_size=hidden_size, bias=False))
          T = sigm(i_for_T_t + self.linear(s_lm1_for_T, in_size=hidden_size, out_size=hidden_size, bias=False))
        else:
          H = tanh(self.linear(s_lm1_for_H, in_size=hidden_size, out_size=hidden_size, bias=True, bias_init=init_H_bias))
          T = sigm(self.linear(s_lm1_for_T, in_size=hidden_size, out_size=hidden_size, bias=True, bias_init=init_T_bias))
        s_l = (H - s_lm1) * T + s_lm1
        s_lm1 = s_l

      y_t = s_l
      return y_t

    # The recurrent hidden state of the RHN is sticky (the last hidden state of one batch is carried over to the next batch,
    # to be used as an initial hidden state).  These states are kept in shared variables and are reset before every epoch.
    y_0 = theano.shared(np.zeros((batch_size, hidden_size), floatX))
    self._sticky_hidden_states.append(y_0)

    y, _ = theano.scan(step_fn,
      sequences = [i_for_H, i_for_T],
      outputs_info = [y_0],
      non_sequences = [noise_s])

    y_last = y[-1]
    sticky_state_updates = [(y_0, y_last)]

    return y, sticky_state_updates

