"""Word/Symbol level next step prediction using Recurrent Highway Networks.

To run:
$ python rhn_train.py

"""
from __future__ import absolute_import, division, print_function
from copy import deepcopy
import time
import os

import numpy as np
import tensorflow as tf

from sacred import Experiment
from rhn import Model
from data.reader import data_iterator

ex = Experiment('rhn_prediction')
logging = tf.logging

class Config:
  pass
C = Config()


@ex.config
def hyperparameters():
  data_path = 'data'
  dataset = 'ptb'
  init_scale = 0.04
  init_bias = -2.0
  num_layers = 1
  depth = 4  #  the recurrence depth
  learning_rate = 0.2
  lr_decay = 1.02
  weight_decay = 1e-7
  max_grad_norm = 10
  num_steps = 35
  hidden_size = 1000
  max_epoch = 20
  max_max_epoch = 500
  batch_size = 20
  drop_x = 0.25
  drop_i = 0.75
  drop_h = 0.25
  drop_o = 0.75
  tied = True
  load_model = ''
  mc_steps = 0
  if dataset == 'ptb':
    vocab_size = 10000
  elif dataset == 'enwik8':
    vocab_size = 205
  elif dataset == 'text8':
    vocab_size = 27
  else:
    raise AssertionError("Unsupported dataset! Only 'ptb',",
                         "'enwik8' and 'text8' are currently supported.")


@ex.named_config
def ptb_sota():
  data_path = 'data'
  dataset = 'ptb'
  init_scale = 0.04
  init_bias = -2.0
  num_layers = 1
  depth = 10
  learning_rate = 0.2
  lr_decay = 1.02
  weight_decay = 1e-7
  max_grad_norm = 10
  num_steps = 35
  hidden_size = 830
  max_epoch = 20
  max_max_epoch = 500
  batch_size = 20
  drop_x = 0.25
  drop_i = 0.75
  drop_h = 0.25
  drop_o = 0.75
  tied = True
  vocab_size = 10000


@ex.named_config
def enwik8_sota():
  # test BPC 1.27
  data_path = 'data'
  dataset = 'enwik8'
  init_scale = 0.04
  init_bias = -4.0
  num_layers = 1
  depth = 10
  learning_rate = 0.2
  lr_decay = 1.03
  weight_decay = 1e-7
  max_grad_norm = 10
  num_steps = 50
  hidden_size = 1500
  max_epoch = 5
  max_max_epoch = 500
  batch_size = 128
  drop_x = 0.10
  drop_i = 0.40
  drop_h = 0.10
  drop_o = 0.40
  tied = False
  vocab_size = 205

@ex.named_config
def text8_sota():
  # test BPC 1.27
  data_path = 'data'
  dataset = 'text8'
  init_scale = 0.04
  init_bias = -4.0
  num_layers = 1
  depth = 10
  learning_rate = 0.2
  lr_decay = 1.03
  weight_decay = 1e-7
  max_grad_norm = 10
  num_steps = 50
  hidden_size = 1500
  max_epoch = 5
  max_max_epoch = 500
  batch_size = 128
  drop_x = 0.10
  drop_i = 0.40
  drop_h = 0.10
  drop_o = 0.40
  tied = False
  vocab_size = 27


@ex.capture
def get_config(_config):

  C.__dict__ = dict(_config)
  return C


def get_data(data_path, dataset):
  if dataset == 'ptb':
    from tensorflow.models.rnn.ptb import reader
    raw_data = reader.ptb_raw_data(data_path)
  elif dataset == 'enwik8':
    from data import reader
    raw_data = reader.enwik8_raw_data(data_path)
  elif dataset == 'text8':
    from data import reader
    raw_data = reader.text8_raw_data(data_path)
  return reader, raw_data


def get_noise(x, m, drop_x, drop_i, drop_h, drop_o):
  keep_x, keep_i, keep_h, keep_o = 1.0 - drop_x, 1.0 - drop_i, 1.0 - drop_h, 1.0 - drop_o
  if keep_x < 1.0:
    noise_x = (np.random.random_sample((m.batch_size, m.num_steps, 1)) < keep_x).astype(np.float32) / keep_x
    for b in range(m.batch_size):
      for n1 in range(m.num_steps):
        for n2 in range(n1 + 1, m.num_steps):
          if x[b][n2] == x[b][n1]:
            noise_x[b][n2][0] = noise_x[b][n1][0]
            break
  else:
    noise_x = np.ones((m.batch_size, m.num_steps, 1), dtype=np.float32)

  if keep_i < 1.0:
    noise_i = (np.random.random_sample((m.batch_size, m.in_size, m.num_layers)) < keep_i).astype(np.float32) / keep_i
  else:
    noise_i = np.ones((m.batch_size, m.in_size, m.num_layers), dtype=np.float32)
  if keep_h < 1.0:
    noise_h = (np.random.random_sample((m.batch_size, m.size, m.num_layers)) < keep_h).astype(np.float32) / keep_h
  else:
    noise_h = np.ones((m.batch_size, m.size, m.num_layers), dtype=np.float32)
  if keep_o < 1.0:
    noise_o = (np.random.random_sample((m.batch_size, 1, m.size)) < keep_o).astype(np.float32) / keep_o
  else:
    noise_o = np.ones((m.batch_size, 1, m.size), dtype=np.float32)
  return noise_x, noise_i, noise_h, noise_o


def run_epoch(session, m, data, eval_op, config, verbose=False):
  """Run the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = [x.eval() for x in m.initial_state]
  for step, (x, y) in enumerate(data_iterator(data, m.batch_size, m.num_steps)):
    noise_x, noise_i, noise_h, noise_o = get_noise(x, m, config.drop_x, config.drop_i, config.drop_h, config.drop_o)
    feed_dict = {m.input_data: x, m.targets: y,
                 m.noise_x: noise_x, m.noise_i: noise_i, m.noise_h: noise_h, m.noise_o: noise_o}
    feed_dict.update({m.initial_state[i]: state[i] for i in range(m.num_layers)})
    cost, state, _ = session.run([m.cost, m.final_state, eval_op], feed_dict)
    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
                                                       iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


@ex.command
def evaluate(data_path, dataset, load_model):
  """Evaluate the model on the given data."""
  ex.commands["print_config"]()
  print("Evaluating model:", load_model)
  reader, (train_data, valid_data, test_data, _) = get_data(data_path, dataset)

  config = get_config()
  val_config = deepcopy(config)
  test_config = deepcopy(config)
  val_config.drop_x = test_config.drop_x = 0.0
  val_config.drop_i = test_config.drop_i = 0.0
  val_config.drop_h = test_config.drop_h = 0.0
  val_config.drop_o = test_config.drop_o = 0.0
  test_config.batch_size = test_config.num_steps = 1

  with tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      _ = Model(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = Model(is_training=False, config=val_config)
      mtest = Model(is_training=False, config=test_config)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(session, load_model)

    print("Testing on batched Valid ...")
    valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(), config=val_config)
    print("Valid Perplexity (batched): %.3f, Bits: %.3f" % (valid_perplexity, np.log2(valid_perplexity)))

    print("Testing on non-batched Valid ...")
    valid_perplexity = run_epoch(session, mtest, valid_data, tf.no_op(), config=test_config, verbose=True)
    print("Full Valid Perplexity: %.3f, Bits: %.3f" % (valid_perplexity, np.log2(valid_perplexity)))

    print("Testing on non-batched Test ...")
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(), config=test_config, verbose=True)
    print("Full Test Perplexity: %.3f, Bits: %.3f" % (test_perplexity, np.log2(test_perplexity)))


def run_mc_epoch(seed, session, m, data, eval_op, config, mc_steps, verbose=False):
  """Run the model with noise on the given data multiple times for MC evaluation."""
  n_steps = len(data)
  all_probs = np.array([0.0]*n_steps)
  sum_probs = np.array([0.0]*n_steps)
  mc_i = 1
  print("Total MC steps to do:", mc_steps)
  if not os.path.isdir('./probs'):
    print('Creating probs directory')
    os.mkdir('./probs')
  while mc_i <= mc_steps:
    print("MC sample number:", mc_i)
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = [x.eval() for x in m.initial_state]

    for step, (x, y) in enumerate(data_iterator(data, m.batch_size, m.num_steps)):
      if step == 0:
        noise_x, noise_i, noise_h, noise_o = get_noise(x, m, config.drop_x, config.drop_i, config.drop_h, config.drop_o)
      feed_dict = {m.input_data: x, m.targets: y,
                   m.noise_x: noise_x, m.noise_i: noise_i, m.noise_h: noise_h, m.noise_o: noise_o}
      feed_dict.update({m.initial_state[i]: state[i] for i in range(m.num_layers)})
      cost, state, _ = session.run([m.cost, m.final_state, eval_op], feed_dict)
      costs += cost
      iters += m.num_steps
      all_probs[step] = np.exp(-cost)
      if verbose and step % (epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
                                                         iters * m.batch_size / (time.time() - start_time)))
    perplexity = np.exp(costs / iters)
    print("Perplexity:", perplexity)
    if perplexity < 500:
      savefile = 'probs/' + str(seed) + '_' + str(mc_i)
      print("Accepted. Saving to:", savefile)
      np.save(savefile, all_probs)
      sum_probs += all_probs
      mc_i += 1

  return np.exp(np.mean(-np.log(np.clip(sum_probs/mc_steps, 1e-10, 1-1e-10))))


@ex.command
def evaluate_mc(data_path, dataset, load_model, mc_steps, seed):
  """Evaluate the model on the given data using MC averaging."""
  ex.commands['print_config']()
  print("MC Evaluation of model:", load_model)
  assert mc_steps > 0
  reader, (train_data, valid_data, test_data, _) = get_data(data_path, dataset)

  config = get_config()
  val_config = deepcopy(config)
  test_config = deepcopy(config)
  test_config.batch_size = test_config.num_steps = 1
  with tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      _ = Model(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      _ = Model(is_training=False, config=val_config)
      mtest = Model(is_training=False, config=test_config)
    tf.initialize_all_variables()
    saver = tf.train.Saver()
    saver.restore(session, load_model)

    print("Testing on non-batched Test ...")
    test_perplexity = run_mc_epoch(seed, session, mtest, test_data, tf.no_op(), test_config, mc_steps, verbose=True)
    print("Full Test Perplexity: %.3f, Bits: %.3f" % (test_perplexity, np.log2(test_perplexity)))


@ex.automain
def main(data_path, dataset, seed, _run):
  ex.commands['print_config']()
  np.random.seed(seed)
  reader, (train_data, valid_data, test_data, _) = get_data(data_path, dataset)

  config = get_config()
  val_config = deepcopy(config)
  test_config = deepcopy(config)
  val_config.drop_x = test_config.drop_x = 0.0
  val_config.drop_i = test_config.drop_i = 0.0
  val_config.drop_h = test_config.drop_h = 0.0
  val_config.drop_o = test_config.drop_o = 0.0
  test_config.batch_size = test_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    tf.set_random_seed(seed)
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      mtrain = Model(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = Model(is_training=False, config=val_config)
      mtest = Model(is_training=False, config=test_config)

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    trains, vals, tests, best_val = [np.inf], [np.inf], [np.inf], np.inf

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch + 1, 0.0)
      mtrain.assign_lr(session, config.learning_rate / lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
      train_perplexity = run_epoch(session, mtrain, train_data, mtrain.train_op, config=config,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f, Bits: %.3f" % (i + 1, train_perplexity, np.log2(train_perplexity)))

      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(), config=val_config)
      print("Epoch: %d Valid Perplexity (batched): %.3f, Bits: %.3f" % (i + 1, valid_perplexity, np.log2(valid_perplexity)))

      test_perplexity = run_epoch(session, mvalid, test_data, tf.no_op(), config=val_config)
      print("Epoch: %d Test Perplexity (batched): %.3f, Bits: %.3f" % (i + 1, test_perplexity, np.log2(test_perplexity)))

      trains.append(train_perplexity)
      vals.append(valid_perplexity)
      tests.append(test_perplexity)

      if valid_perplexity < best_val:
        best_val = valid_perplexity
        print("Best Batched Valid Perplexity improved to %.03f" % best_val)
        save_path = saver.save(session, './' + dataset + "_" + str(seed) + "_best_model.ckpt")
        print("Saved to:", save_path)

      _run.info['epoch_nr'] = i + 1
      _run.info['nr_parameters'] = mtrain.nvars.item()
      _run.info['logs'] = {'train_perplexity': trains, 'valid_perplexity': vals, 'test_perplexity': tests}


    print("Training is over.")
    best_val_epoch = np.argmin(vals)
    print("Best Batched Validation Perplexity %.03f (Bits: %.3f) was at Epoch %d" %
          (vals[best_val_epoch], np.log2(vals[best_val_epoch]), best_val_epoch))
    print("Training Perplexity at this Epoch was %.03f, Bits: %.3f" %
          (trains[best_val_epoch], np.log2(trains[best_val_epoch])))
    print("Batched Test Perplexity at this Epoch was %.03f, Bits: %.3f" %
          (tests[best_val_epoch], np.log2(tests[best_val_epoch])))

    _run.info['best_val_epoch'] = best_val_epoch
    _run.info['best_valid_perplexity'] = vals[best_val_epoch]

    with tf.Session() as sess:
      saver.restore(sess, './'  + dataset + "_" + str(seed) + "_best_model.ckpt")

      print("Testing on non-batched Valid ...")
      valid_perplexity = run_epoch(sess, mtest, valid_data, tf.no_op(), config=test_config, verbose=True)
      print("Full Valid Perplexity: %.3f, Bits: %.3f" % (valid_perplexity, np.log2(valid_perplexity)))

      print("Testing on non-batched Test ...")
      test_perplexity = run_epoch(sess, mtest, test_data, tf.no_op(), config=test_config, verbose=True)
      print("Full Test Perplexity: %.3f, Bits: %.3f" % (test_perplexity, np.log2(test_perplexity)))

      _run.info['full_best_valid_perplexity'] = valid_perplexity
      _run.info['full_test_perplexity'] = test_perplexity

  return vals[best_val_epoch]
