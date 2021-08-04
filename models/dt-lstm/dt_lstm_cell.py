from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops, array_ops, nn_ops, init_ops
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DT_LSTMCell(tf.nn.rnn_cell.BasicLSTMCell): # Based on Tensorflow's BasicLSTMCell
  def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, transition_activation=None, name=None, dtype=None):
    super(DT_LSTMCell, self).__init__(num_units=num_units, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple

    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if transition_activation:
      self._transition_activation = activations.get(activation)
    else:
      self._transition_activation = tf.nn.relu

  @property
  def state_size(self):
    return (tf.nn.rnn_cell.LSTMStateTuple(self._num_units[-1], self._num_units[-1])
            if self._state_is_tuple else 2 * self._num_units[-1])

  @property
  def output_size(self):
    return self._num_units[-1]

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % str(inputs_shape))

    input_depth = inputs_shape[-1]
    h_depth = self._num_units[0]

    # Kernel list for deep transition
    self._kernel = [self.add_variable(_WEIGHTS_VARIABLE_NAME+"_0", shape=[input_depth + h_depth, 4 * self._num_units[0]], initializer=tf.keras.initializers.glorot_normal())]
    for i in range(1, len(self._num_units)):
      self._kernel.append(self.add_variable(_WEIGHTS_VARIABLE_NAME+"_"+str(i), shape=[4 * self._num_units[i] + h_depth, 4 * self._num_units[i]], initializer=tf.keras.initializers.glorot_normal()))

    # Bias list for deep transition
    self._bias = [self.add_variable(_BIAS_VARIABLE_NAME+"_0", shape=[4 * self._num_units[0]], initializer=init_ops.zeros_initializer(dtype=self.dtype))]
    for i in range(1, len(self._num_units)):
      self._bias.append(self.add_variable(_BIAS_VARIABLE_NAME+"_"+str(i), shape=[4 * self._num_units[i]], initializer=init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)

    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel[0])
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias[0])

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(
        multiply(c, sigmoid(add(f, forget_bias_tensor))),
        multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    # Deep transition between states
    for k in range(1, len(self._kernel)):
      gate_inputs = math_ops.matmul(array_ops.concat([i, j, f, o, new_h], 1), self._kernel[k])
      gate_inputs = nn_ops.bias_add(gate_inputs, self._bias[k])

      i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
      forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)

      i = sigmoid(i)
      j = self._transition_activation(j)
      f = sigmoid(add(f, forget_bias_tensor))
      o = sigmoid(o)

      new_c = add(multiply(new_c, f), multiply(i, j))
      new_h = self._transition_activation(multiply(new_c, o))

    if self._state_is_tuple:
      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)

    return new_h, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation)
    }

    base_config = super(DT_LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))