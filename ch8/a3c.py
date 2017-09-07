"""Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning."""

import numpy as np
import tensorflow as tf
import copy
import multiprocessing
import os
import re
import threading
from collections import Sequence

import pickle
import threading
import time

import numpy as np
import os
import six
import tensorflow as tf
import tempfile
from deepchem.models import Model

class TensorGraph(Model):

  def __init__(self,
               batch_size=100,
               random_seed=None,
               graph=None,
               learning_rate=0.001,
               model_dir=None,
               **kwargs):
    """
    Parameters
    ----------
    batch_size: int
      default batch size for training and evaluating
    graph: tensorflow.Graph
      the Graph in which to create Tensorflow objects.  If None, a new Graph
      is created.
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    kwargs
    """

    # Layer Management
    self.layers = dict()
    self.features = list()
    self.labels = list()
    self.outputs = list()
    self.task_weights = list()
    self.loss = None
    self.built = False
    self.optimizer = None
    self.learning_rate = learning_rate

    # Singular place to hold Tensor objects which don't serialize
    # These have to be reconstructed on restoring from pickle
    # See TensorGraph._get_tf() for more details on lazy construction
    self.tensor_objects = {
        "Graph": graph,
        "train_op": None,
    }
    self.global_step = 0

    self.batch_size = batch_size
    self.random_seed = random_seed
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
      self.model_dir_is_temp = True
    self.model_dir = model_dir
    self.save_file = "%s/%s" % (self.model_dir, "model")
    self.model_class = None

  def _add_layer(self, layer):
    if layer.name is None:
      layer.name = "%s_%s" % (layer.__class__.__name__, len(self.layers) + 1)
    if layer.name in self.layers:
      return
    #if isinstance(layer, Feature):
    if isinstance(layer, Input):
      self.features.append(layer)
    self.layers[layer.name] = layer
    for in_layer in layer.in_layers:
      self._add_layer(in_layer)

  def topsort(self):

    def add_layers_to_list(layer, sorted_layers):
      if layer in sorted_layers:
        return
      for in_layer in layer.in_layers:
        add_layers_to_list(in_layer, sorted_layers)
      sorted_layers.append(layer)

    sorted_layers = []
    for l in self.features + self.labels + self.task_weights + self.outputs:
      add_layers_to_list(l, sorted_layers)
    add_layers_to_list(self.loss, sorted_layers)
    return sorted_layers

  def build(self):
    if self.built:
      return
    with self._get_tf("Graph").as_default():
      self._training_placeholder = tf.placeholder(dtype=tf.float32, shape=())
      if self.random_seed is not None:
        tf.set_random_seed(self.random_seed)
      for layer in self.topsort():
        with tf.name_scope(layer.name):
          layer.create_tensor(training=self._training_placeholder)
      self.session = tf.Session()

      self.built = True

  def set_loss(self, layer):
    self._add_layer(layer)
    self.loss = layer

  def add_output(self, layer):
    self._add_layer(layer)
    self.outputs.append(layer)

  def set_optimizer(self, optimizer):
    """Set the optimizer to use for fitting."""
    self.optimizer = optimizer

  def get_layer_variables(self, layer):
    """Get the list of trainable variables in a layer of the graph."""
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      if layer.variable_scope == "":
        return []
      return tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=layer.variable_scope)

  def get_global_step(self):
    return self._get_tf("GlobalStep")

  def _get_tf(self, obj):
    """Fetches underlying TensorFlow primitives.

    Parameters
    ----------
    obj: str
      If "Graph", returns tf.Graph instance. If "Optimizer", returns the
      optimizer. If "train_op", returns the train operation. If "GlobalStep" returns
      the global step.
    Returns
    -------
    TensorFlow Object

    """

    if obj in self.tensor_objects and self.tensor_objects[obj] is not None:
      return self.tensor_objects[obj]
    if obj == "Graph":
      self.tensor_objects["Graph"] = tf.Graph()
    elif obj == "Optimizer":
      self.tensor_objects["Optimizer"] = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-7)
    elif obj == "GlobalStep":
      with self._get_tf("Graph").as_default():
        self.tensor_objects["GlobalStep"] = tf.Variable(0, trainable=False)
    return self._get_tf(obj)

  def restore(self):
    """Reload the values of all variables from the most recent checkpoint file."""
    if not self.built:
      self.build()
    last_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if last_checkpoint is None:
      raise ValueError("No checkpoint found")
    with self._get_tf("Graph").as_default():
      saver = tf.train.Saver()
      saver.restore(self.session, last_checkpoint)

  def __del__(self):
    pass

class Layer(object):
  layer_number_dict = {}

  def __init__(self, in_layers=None, **kwargs):
    if "name" in kwargs:
      self.name = kwargs["name"]
    else:
      self.name = None
    if in_layers is None:
      in_layers = list()
    if not isinstance(in_layers, Sequence):
      in_layers = [in_layers]
    self.in_layers = in_layers
    self.op_type = "gpu"
    self.variable_scope = ""
    self.tb_input = None

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    raise NotImplementedError("Subclasses must implement for themselves")

  def _get_input_tensors(self, in_layers, reshape=False):
    """Get the input tensors to his layer.

    Parameters
    ----------
    in_layers: list of Layers or tensors
      the inputs passed to create_tensor().  If None, this layer's inputs will
      be used instead.
    reshape: bool
      if True, try to reshape the inputs to all have the same shape
    """
    if in_layers is None:
      in_layers = self.in_layers
    if not isinstance(in_layers, Sequence):
      in_layers = [in_layers]
    tensors = []
    for input in in_layers:
      tensors.append(tf.convert_to_tensor(input))
    if reshape and len(tensors) > 1:
      shapes = [t.get_shape() for t in tensors]
      if any(s != shapes[0] for s in shapes[1:]):
        # Reshape everything to match the input with the most dimensions.

        shape = shapes[0]
        for s in shapes:
          if len(s) > len(shape):
            shape = s
        shape = [-1 if x is None else x for x in shape.as_list()]
        for i in range(len(tensors)):
          tensors[i] = tf.reshape(tensors[i], shape)
    return tensors

  def _record_variable_scope(self, local_scope):
    """Record the scope name used for creating variables.

    This should be called from create_tensor().  It allows the list of variables
    belonging to this layer to be retrieved later."""
    parent_scope = tf.get_variable_scope().name
    if len(parent_scope) > 0:
      self.variable_scope = "%s/%s" % (parent_scope, local_scope)
    else:
      self.variable_scope = local_scope

def _convert_layer_to_tensor(value, dtype=None, name=None, as_ref=False):
  return tf.convert_to_tensor(value.out_tensor, dtype=dtype, name=name)


tf.register_tensor_conversion_function(Layer, _convert_layer_to_tensor)

def convert_to_layers(in_layers):
  """Wrap all inputs into tensors if necessary."""
  layers = []
  for in_layer in in_layers:
    if isinstance(in_layer, Layer):
      layers.append(in_layer)
    elif isinstance(in_layer, tf.Tensor):
      layers.append(TensorWrapper(in_layer))
    else:
      raise ValueError("convert_to_layers must be invoked on layers or tensors")
  return layers

class Dense(Layer):

  def __init__(
      self,
      out_channels,
      activation_fn=None,
      biases_initializer=tf.zeros_initializer,
      weights_initializer=tf.contrib.layers.variance_scaling_initializer,
      **kwargs):
    """Create a dense layer.

    The weight and bias initializers are specified by callable objects that construct
    and return a Tensorflow initializer when invoked with no arguments.  This will typically
    be either the initializer class itself (if the constructor does not require arguments),
    or a TFWrapper (if it does).

    Parameters
    ----------
    out_channels: int
      the number of output values
    activation_fn: object
      the Tensorflow activation function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    super(Dense, self).__init__(**kwargs)
    self.out_channels = out_channels
    self.out_tensor = None
    self.activation_fn = activation_fn
    self.biases_initializer = biases_initializer
    self.weights_initializer = weights_initializer
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = (parent_shape[0], out_channels)
    except:
      pass
    self._shared_with = None

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Dense layer can only have one input")
    parent = inputs[0]
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer()
    out_tensor = tf.contrib.layers.fully_connected(parent,
                                                   num_outputs=self.out_channels,
                                                   activation_fn=self.activation_fn,
                                                   biases_initializer=biases_initializer,
                                                   weights_initializer=self.weights_initializer(),
                                                   scope=self._get_scope_name(),
                                                   reuse=False,
                                                   trainable=True)
    if set_tensors:
      self._record_variable_scope(self._get_scope_name())
      self.out_tensor = out_tensor
    return out_tensor

  def _get_scope_name(self):
    if self._shared_with is None:
      return self.name
    else:
      return self._shared_with._get_scope_name()

class Squeeze(Layer):

  def __init__(self, in_layers=None, squeeze_dims=None, **kwargs):
    self.squeeze_dims = squeeze_dims
    super(Squeeze, self).__init__(in_layers, **kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      if squeeze_dims is None:
        self._shape = [i for i in parent_shape if i != 1]
      else:
        self._shape = [
            parent_shape[i] for i in range(len(parent_shape))
            if i not in squeeze_dims
        ]
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.squeeze(parent_tensor, squeeze_dims=self.squeeze_dims)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class BatchNorm(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(BatchNorm, self).__init__(in_layers, **kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.layers.batch_normalization(parent_tensor)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class Flatten(Layer):
  """Flatten every dimension except the first"""

  def __init__(self, in_layers=None, **kwargs):
    super(Flatten, self).__init__(in_layers, **kwargs)
    try:
      parent_shape = self.in_layers[0].shape
      s = list(parent_shape[:2])
      for x in parent_shape[2:]:
        s[1] *= x
      self._shape = tuple(s)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Only One Parent to Flatten")
    parent = inputs[0]
    parent_shape = parent.get_shape()
    vector_size = 1
    for i in range(1, len(parent_shape)):
      vector_size *= parent_shape[i].value
    parent_tensor = parent
    out_tensor = tf.reshape(parent_tensor, shape=(-1, vector_size))
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class SoftMax(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(SoftMax, self).__init__(in_layers, **kwargs)
    try:
      self._shape = tuple(self.in_layers[0].shape)
    except:
      pass

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Must only Softmax single parent")
    parent = inputs[0]
    out_tensor = tf.contrib.layers.softmax(parent)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor

class Input(Layer):

  def __init__(self, shape, dtype=tf.float32, **kwargs):
    self._shape = tuple(shape)
    self.dtype = dtype
    super(Input, self).__init__(**kwargs)
    self.op_type = "cpu"

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    out_tensor = tf.placeholder(dtype=self.dtype, shape=self._shape)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class A3CLoss(Layer):
  """This layer computes the loss function for A3C."""

  def __init__(self, value_weight, entropy_weight, **kwargs):
    super(A3CLoss, self).__init__(**kwargs)
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight

  def create_tensor(self, **kwargs):
    reward, action, prob, value, advantage = [
        layer.out_tensor for layer in self.in_layers
    ]
    prob = prob + np.finfo(np.float32).eps
    log_prob = tf.log(prob)
    policy_loss = -tf.reduce_mean(
        advantage * tf.reduce_sum(action * log_prob, axis=1))
    value_loss = tf.reduce_mean(tf.square(reward - value))
    entropy = -tf.reduce_mean(tf.reduce_sum(prob * log_prob, axis=1))
    self.out_tensor = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
    return self.out_tensor


class A3C(object):
  """
  Implements the Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning.

  The algorithm is described in Mnih et al, "Asynchronous Methods for Deep
  Reinforcement Learning" (https://arxiv.org/abs/1602.01783).  This class
  requires the policy to output two quantities: a vector giving the probability
  of taking each action, and an estimate of the value function for the current
  state.  It optimizes both outputs at once using a loss that is the sum of three
  terms:

  1. The policy loss, which seeks to maximize the discounted reward for each action.
  2. The value loss, which tries to make the value estimate match the actual
     discounted reward that was attained at each step.
  3. An entropy term to encourage exploration.

  This class only supports environments with discrete action spaces, not
  continuous ones.  The "action" argument passed to the environment is an
  integer, giving the index of the action to perform.

  This class supports Generalized Advantage Estimation as described in Schulman
  et al., "High-Dimensional Continuous Control Using Generalized Advantage
  Estimation" (https://arxiv.org/abs/1506.02438).  This is a method of trading
  off bias and variance in the advantage estimate, which can sometimes improve
  the rate of convergance.  Use the advantage_lambda parameter to adjust the
  tradeoff.
  """

  def __init__(self,
               env,
               max_rollout_length=20,
               discount_factor=0.99,
               advantage_lambda=0.98,
               value_weight=1.0,
               entropy_weight=0.01,
               optimizer=None,
               model_dir=None):
    """Create an object for optimizing a policy.

    Parameters
    ----------
    env: Environment
      the Environment to interact with
    max_rollout_length: int
      the maximum length of rollouts to generate
    discount_factor: float
      the discount factor to use when computing rewards
    advantage_lambda: float
      the parameter for trading bias vs. variance in Generalized Advantage Estimation
    value_weight: float
      a scale factor for the value loss term in the loss function
    entropy_weight: float
      a scale factor for the entropy term in the loss function
    optimizer: Optimizer
      the optimizer to use.  If None, a default optimizer is used.
    model_dir: str
      the directory in which the model will be saved.  If None, a temporary
      directory will be created.
    """
    self._env = env
    self.max_rollout_length = max_rollout_length
    self.discount_factor = discount_factor
    self.advantage_lambda = advantage_lambda
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    self._optimizer = None
    (self._graph, self._features, self._rewards, self._actions,
     self._action_prob, self._value, self._advantages) = self.build_graph(
         None, "global", model_dir)
    with self._graph._get_tf("Graph").as_default():
      self._session = tf.Session()

  def build_graph(self, tf_graph, scope, model_dir):
    """Construct a TensorGraph containing the policy and loss calculations."""
    state_shape = self._env.state_shape
    features = []
    for s in state_shape:
      features.append(Input(shape=[None] + list(s), dtype=tf.float32))
    d1 = Flatten(in_layers=features)
    d2 = Dense(
        in_layers=[d1],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=64)
    d3 = Dense(
        in_layers=[d2],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=32)
    d4 = Dense(
        in_layers=[d3],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=16)
    d4 = BatchNorm(in_layers=[d4])
    d5 = Dense(in_layers=[d4], activation_fn=None, out_channels=9)
    value = Dense(in_layers=[d4], activation_fn=None, out_channels=1)
    value = Squeeze(squeeze_dims=1, in_layers=[value])
    action_prob = SoftMax(in_layers=[d5])

    rewards = Input(shape=(None,))
    advantages = Input(shape=(None,))
    actions = Input(shape=(None, self._env.n_actions))
    loss = A3CLoss(
        self.value_weight,
        self.entropy_weight,
        in_layers=[rewards, actions, action_prob, value, advantages])
    graph = TensorGraph(
        batch_size=self.max_rollout_length,
        graph=tf_graph,
        model_dir=model_dir)
    for f in features:
      graph._add_layer(f)
    graph.add_output(action_prob)
    graph.add_output(value)
    graph.set_loss(loss)
    graph.set_optimizer(self._optimizer)
    with graph._get_tf("Graph").as_default():
      with tf.variable_scope(scope):
        graph.build()
    return graph, features, rewards, actions, action_prob, value, advantages

  def fit(self,
          total_steps,
          max_checkpoints_to_keep=5,
          checkpoint_interval=600,
          restore=False):
    """Train the policy.

    Parameters
    ----------
    total_steps: int
      the total number of time steps to perform on the environment, across all
      rollouts on all threads
    max_checkpoints_to_keep: int
      the maximum number of checkpoint files to keep.  When this number is
      reached, older files are deleted.
    checkpoint_interval: float
      the time interval at which to save checkpoints, measured in seconds
    restore: bool
      if True, restore the model from the most recent checkpoint and continue
      training from there.  If False, retrain the model from scratch.
    """
    with self._graph._get_tf("Graph").as_default():
      step_count = [0]
      workers = []
      threads = []
      for i in range(multiprocessing.cpu_count()):
        workers.append(Worker(self, i))
      self._session.run(tf.global_variables_initializer())
      if restore:
        self.restore()
      for worker in workers:
        thread = threading.Thread(
            name=worker.scope,
            target=lambda: worker.run(step_count, total_steps))
        threads.append(thread)
        thread.start()
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope="global")
      saver = tf.train.Saver(variables, max_to_keep=max_checkpoints_to_keep)
      checkpoint_index = 0
      while True:
        threads = [t for t in threads if t.isAlive()]
        if len(threads) > 0:
          threads[0].join(checkpoint_interval)
        checkpoint_index += 1
        saver.save(
            self._session, self._graph.save_file, global_step=checkpoint_index)
        if len(threads) == 0:
          break

  def predict(self, state, use_saved_states=True, save_states=True):
    """Compute the policy's output predictions for a state.

    If the policy involves recurrent layers, this method can preserve their internal
    states between calls.  Use the use_saved_states and save_states arguments to specify
    how it should behave.

    Parameters
    ----------
    state: array
      the state of the environment for which to generate predictions
    use_saved_states: bool
      if True, the states most recently saved by a previous call to predict()
      or select_action() will be used as the initial states.  If False, the
      internal states of all recurrent layers will be set to all zeros before
      computing the predictions.
    save_states: bool
      if True, the internal states of all recurrent layers at the end of the
      calculation will be saved, and any previously saved states will be discarded.
      If False, the states at the end of the calculation will be discarded, and any
      previously saved states will be kept.

    Returns
    -------
    the array of action probabilities, and the estimated value function
    """
    with self._graph._get_tf("Graph").as_default():
      feed_dict = self.create_feed_dict(state, use_saved_states)
      tensors = [self._action_prob.out_tensor, self._value.out_tensor]
      results = self._session.run(tensors, feed_dict=feed_dict)
      return results[:2]

  def select_action(self,
                    state,
                    deterministic=False,
                    use_saved_states=True,
                    save_states=True):
    """Select an action to perform based on the environment's state.

    If the policy involves recurrent layers, this method can preserve their
    internal states between calls.  Use the use_saved_states and save_states
    arguments to specify how it should behave.

    Parameters
    ----------
    state: array
      the state of the environment for which to select an action
    deterministic: bool
      if True, always return the best action (that is, the one with highest
      probability).  If False, randomly select an action based on the computed
      probabilities.
    use_saved_states: bool
      if True, the states most recently saved by a previous call to predict()
      or select_action() will be used as the initial states.  If False, the
      internal states of all recurrent layers will be set to all zeros before
      computing the predictions.
    save_states: bool
      if True, the internal states of all recurrent layers at the end of the
      calculation will be saved, and any previously saved states will be
      discarded.  If False, the states at the end of the calculation will be
      discarded, and any previously saved states will be kept.

    Returns
    -------
    the index of the selected action
    """
    with self._graph._get_tf("Graph").as_default():
      feed_dict = self.create_feed_dict(state, use_saved_states)
      tensors = [self._action_prob.out_tensor]
      results = self._session.run(tensors, feed_dict=feed_dict)
      probabilities = results[0]
      if deterministic:
        return probabilities.argmax()
      else:
        return np.random.choice(
            np.arange(self._env.n_actions), p=probabilities[0])

  def restore(self):
    """Reload the model parameters from the most recent checkpoint file."""
    last_checkpoint = tf.train.latest_checkpoint(self._graph.model_dir)
    if last_checkpoint is None:
      raise ValueError("No checkpoint found")
    with self._graph._get_tf("Graph").as_default():
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope="global")
      saver = tf.train.Saver(variables)
      saver.restore(self._session, last_checkpoint)

  def create_feed_dict(self, state, use_saved_states):
    """Create a feed dict for use by predict() or select_action()."""
    feed_dict = dict((f.out_tensor, np.expand_dims(s, axis=0))
                     for f, s in zip(self._features, state))
    return feed_dict


class Worker(object):
  """A Worker object is created for each training thread."""

  def __init__(self, a3c, index):
    self.a3c = a3c
    self.index = index
    self.scope = "worker%d" % index
    self.env = copy.deepcopy(a3c._env)
    self.env.reset()
    (self.graph, self.features, self.rewards, self.actions, self.action_prob,
     self.value, self.advantages) = a3c.build_graph(
        a3c._graph._get_tf("Graph"), self.scope, None)
    with a3c._graph._get_tf("Graph").as_default():
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope)
      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      "global")
      gradients = tf.gradients(self.graph.loss.out_tensor, local_vars)
      grads_and_vars = list(zip(gradients, global_vars))
      self.train_op = a3c._graph._get_tf("Optimizer").apply_gradients(
          grads_and_vars)
      self.update_local_variables = tf.group(
          * [tf.assign(v1, v2) for v1, v2 in zip(local_vars, global_vars)])
      self.global_step = self.graph.get_global_step()

  def run(self, step_count, total_steps):
    with self.graph._get_tf("Graph").as_default():
      while step_count[0] < total_steps:
        self.a3c._session.run(self.update_local_variables)
        states, actions, rewards, values = self.create_rollout()
        self.process_rollout(states, actions, rewards, values, step_count[0])
        step_count[0] += len(actions)

  def create_rollout(self):
    """Generate a rollout."""
    n_actions = self.env.n_actions
    session = self.a3c._session
    states = []
    actions = []
    rewards = []
    values = []

    # Generate the rollout.

    for i in range(self.a3c.max_rollout_length):
      if self.env.terminated:
        break
      state = self.env.state
      states.append(state)
      feed_dict = self.create_feed_dict(state)
      results = session.run(
          [self.action_prob.out_tensor, self.value.out_tensor],
          feed_dict=feed_dict)
      probabilities, value = results[:2]
      action = np.random.choice(np.arange(n_actions), p=probabilities[0])
      actions.append(action)
      values.append(float(value))
      rewards.append(self.env.step(action))

    # Compute an estimate of the reward for the rest of the episode.
    if not self.env.terminated:
      feed_dict = self.create_feed_dict(self.env.state)
      final_value = self.a3c.discount_factor * float(
          session.run(self.value.out_tensor, feed_dict))
    else:
      final_value = 0.0
    values.append(final_value)
    if self.env.terminated:
      self.env.reset()
    return states, actions, np.array(rewards), np.array(values)

  def process_rollout(self, states, actions, rewards, values, step_count):
    """Train the network based on a rollout."""

    # Compute the discounted rewards and advantages.
    if len(states) == 0:
      # Rollout creation sometimes fails in multithreaded environment.
      # Don't process if malformed
      print("Rollout creation failed. Skipping")    
      return

    discounted_rewards = rewards.copy()
    discounted_rewards[-1] += values[-1]
    advantages = rewards - values[:-1] + self.a3c.discount_factor * np.array(
        values[1:])
    for j in range(len(rewards) - 1, 0, -1):
      discounted_rewards[j -
                         1] += self.a3c.discount_factor * discounted_rewards[j]
      advantages[
          j -
          1] += self.a3c.discount_factor * self.a3c.advantage_lambda * advantages[
              j]

    # Convert the actions to one-hot.

    n_actions = self.env.n_actions
    actions_matrix = []
    for action in actions:
      a = np.zeros(n_actions)
      a[action] = 1.0
      actions_matrix.append(a)

    # Rearrange the states into the proper set of arrays.
    state_arrays = [[] for i in range(len(self.features))]
    for state in states:
      for j in range(len(state)):
        state_arrays[j].append(state[j])
    
    # Build the feed dict and apply gradients.
    feed_dict = {}
    for f, s in zip(self.features, state_arrays):
      feed_dict[f.out_tensor] = s
    feed_dict[self.rewards.out_tensor] = discounted_rewards
    feed_dict[self.actions.out_tensor] = actions_matrix
    feed_dict[self.advantages.out_tensor] = advantages
    feed_dict[self.global_step] = step_count
    self.a3c._session.run(self.train_op, feed_dict=feed_dict)

  def create_feed_dict(self, state):
    """Create a feed dict for use during a rollout."""
    feed_dict = dict((f.out_tensor, np.expand_dims(s, axis=0))
                     for f, s in zip(self.features, state))
    return feed_dict
