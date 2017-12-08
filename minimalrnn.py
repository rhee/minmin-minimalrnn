import tensorflow as tf

from tensorflow.python.ops import math_ops, init_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell, _linear, _Linear


class MinimalRNNCell(RNNCell):
    """Minimal RNN.
       This implementation is based on:
       Minmin Chen (2017)
       "MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks"
       https://arxiv.org/abs/1711.06788.pdf
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 phi_initializer=None):
      """Initialize the parameters for a cell.
        Args:
          num_units: int, number of units in the cell
          kernel_initializer: (optional) The initializer to use for the weight and
            projection matrices.
          bias_initializer: (optional) The initializer to use for the bias matrices.
            Default: vectors of ones.
      """
      super(MinimalRNNCell, self).__init__(_reuse=True)

      self._activation = activation or math_ops.tanh
      self._num_units = num_units
      self._kernel_initializer = kernel_initializer
      self._bias_initializer = bias_initializer
      self._phi_initializer = phi_initializer
      self._phi = None
      self._gate_linear = None

    @property
    def state_size(self):
      return self._num_units

    @property
    def output_size(self):
      return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Run one step of minimal RNN.
          Args:
            inputs: input Tensor, 2D, batch x num_units.
            state: a state Tensor, `2-D, batch x state_size`.
          Returns:
            A tuple containing:
            - A `2-D, [batch x num_units]`, Tensor representing the output of the
              cell after reading `inputs` when previous state was `state`.
            - A `2-D, [batch x num_units]`, Tensor representing the new state of cell after reading `inputs` when
              the previous state was `state`.  Same type and shape(s) as `state`.
          Raises:
            ValueError:
            - If input size cannot be inferred from inputs via
              static shape inference.
            - If state is not `2D`.
        """

        # Phi projection to a latent space / candidate
        if self._phi is None:
          with tf.variable_scope("candidate"):
            if self._phi_initializer is not None:
                self._phi = self._phi_initializer(
                    inputs,
                    self._num_units,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
            else:
                self._phi = _Linear(
                    inputs,
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)

        z = self._activation(self._phi(inputs))

        # Update gate
        if self._gate_linear is None:
          bias_ones = self._bias_initializer
          if self._bias_initializer is None:
            bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
          with tf.variable_scope("update_gate"):
            self._gate_linear = _Linear(
                [state, z],
                self._num_units,
                True,
                bias_initializer=bias_ones,
                kernel_initializer=self._kernel_initializer)

        u = math_ops.sigmoid(self._gate_linear([state, z]))

        # Activation step
        new_h = u * state + (1 - u) * z

        return new_h, new_h
