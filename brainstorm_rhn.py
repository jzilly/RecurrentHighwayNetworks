#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_time, \
    flatten_time_and_features


def HighwayRNNCoupledGates(size, activation='tanh', name=None, recurrence_depth=1):
    """Create a Simple Recurrent layer."""
    return ConstructionWrapper.create(HighwayRNNCoupledGatesLayerImpl, size=size,
                                      name=name, activation=activation,
                                      recurrence_depth=recurrence_depth)


class HighwayRNNCoupledGatesLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'size', 'activation', 'recurrence_depth', 'block_size', 'sizes_list'}

    def setup(self, kwargs, in_shapes):
        self.activation = kwargs.get('activation', 'tanh')
        self.size = kwargs.get('size', self.in_shapes['default'].feature_size)
        self.recurrence_depth = kwargs.get('recurrence_depth', 1)
        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))
        if not isinstance(self.recurrence_depth, int):
            raise LayerValidationError('recurrence_depth must be int but was {}'.
                                       format(self.recurrence_depth))
        in_size = self.in_shapes['default'].feature_size

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', self.size,
                                             context_size=1)
        parameters = OrderedDict()
        parameters['W_H'] = BufferStructure(self.size, in_size)
        parameters['W_T'] = BufferStructure(self.size, in_size)
        parameters['R_T'] = BufferStructure(self.recurrence_depth, self.size, self.size)
        parameters['bias_T'] = BufferStructure(self.recurrence_depth, self.size)
        parameters['R_H'] = (BufferStructure(self.recurrence_depth, self.size, self.size))
        parameters['bias_H'] = BufferStructure(self.recurrence_depth, self.size)

        internals = OrderedDict()
        for i in range(self.recurrence_depth):
            internals['H_{}'.format(i)] = BufferStructure('T', 'B', self.size, context_size=1)
            internals['T_{}'.format(i)] = BufferStructure('T', 'B', self.size, context_size=1)
            internals['Y_{}'.format(i)] = BufferStructure('T', 'B', self.size, context_size=1)
            internals['dH_{}'.format(i)] = BufferStructure('T', 'B', self.size, context_size=1,
                                                           is_backward_only=True)
            internals['dT_{}'.format(i)] = BufferStructure('T', 'B', self.size, context_size=1,
                                                           is_backward_only=True)
            internals['dY_{}'.format(i)] = BufferStructure('T', 'B', self.size, context_size=1,
                                                           is_backward_only=True)

        return outputs, parameters, internals
    
    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W_H, W_T, R_T, bias_T, R_H, bias_H = buffers.parameters

        inputs = buffers.inputs.default
        outputs = buffers.outputs.default

        H_list = []
        T_list = []
        Y_list = []

        for i in range(self.recurrence_depth):
            H_list.append(buffers.internals['H_{}'.format(i)])
            T_list.append(buffers.internals['T_{}'.format(i)])
            Y_list.append(buffers.internals['Y_{}'.format(i)])

        flat_inputs = flatten_time_and_features(inputs)

        flat_H = flatten_time(H_list[0][:-1])
        flat_T = flatten_time(T_list[0][:-1])

        _h.dot_mm(flat_inputs, W_H, flat_H, transb=True)
        _h.dot_mm(flat_inputs, W_T, flat_T, transb=True)

        for t in range(inputs.shape[0]):
            for i in range(self.recurrence_depth):
                if i == 0:
                    x = outputs[t-1]
                    _h.dot_add_mm(x, R_T[i], T_list[i][t], transb=True)
                    _h.add_mv(T_list[i][t], bias_T[i].reshape((1, self.size)), T_list[i][t])
                    _h.inplace_act_func['sigmoid'](T_list[i][t])
                    _h.dot_add_mm(x, R_H[i], H_list[i][t], transb=True)
                    _h.add_mv(H_list[i][t], bias_H[i].reshape((1, self.size)), H_list[i][t])
                    _h.inplace_act_func[self.activation](H_list[i][t])
                else:
                    x = Y_list[i-1][t]
                    _h.dot_mm(x, R_T[i], T_list[i][t], transb=True)
                    _h.add_mv(T_list[i][t], bias_T[i].reshape((1, self.size)), T_list[i][t])
                    _h.inplace_act_func['sigmoid'](T_list[i][t])
                    _h.dot_mm(x, R_H[i], H_list[i][t], transb=True)
                    _h.add_mv(H_list[i][t], bias_H[i].reshape((1, self.size)), H_list[i][t])
                    _h.inplace_act_func[self.activation](H_list[i][t])

                if i == 0:
                    _h.mult_tt(T_list[i][t], H_list[i][t], out=Y_list[i][t])
                    tmp = _h.ones(H_list[i][t].shape)
                    _h.subtract_tt(tmp, T_list[i][t], tmp)
                    _h.mult_add_tt(tmp, outputs[t-1], out=Y_list[i][t])
                else:
                    _h.mult_tt(T_list[i][t], H_list[i][t], out=Y_list[i][t])
                    tmp = _h.ones(H_list[i][t].shape)
                    _h.subtract_tt(tmp, T_list[i][t], tmp)
                    _h.mult_add_tt(tmp, Y_list[i-1][t], out=Y_list[i][t])
            _h.copy_to(Y_list[self.recurrence_depth-1][t], outputs[t])

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler

        W_H, W_T, R_T, bias_T, R_H, bias_H = buffers.parameters
        dW_H, dW_T, dR_T, dbias_T, dR_H, dbias_H = buffers.gradients

        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        dinputs = buffers.input_deltas.default
        doutputs = buffers.output_deltas.default

        H_list = []
        T_list = []
        Y_list = []
        dH_list = []
        dT_list = []
        dY_list = []

        for i in range(self.recurrence_depth):
            H_list.append(buffers.internals['H_{}'.format(i)])
            T_list.append(buffers.internals['T_{}'.format(i)])
            Y_list.append(buffers.internals['Y_{}'.format(i)])
            dH_list.append(buffers.internals['dH_{}'.format(i)])
            dT_list.append(buffers.internals['dT_{}'.format(i)])
            dY_list.append(buffers.internals['dY_{}'.format(i)])

        t = inputs.shape[0] - 1
        _h.copy_to(doutputs[t], dY_list[self.recurrence_depth-1][t])

        for i in range(self.recurrence_depth-1, -1, -1):
                if i == 0:
                    _h.mult_tt(dY_list[i][t], T_list[i][t], dH_list[i][t])
                    tmp = _h.ones(dH_list[i][t].shape)
                    _h.subtract_tt(H_list[i][t], outputs[t-1], tmp)
                    _h.mult_tt(dY_list[i][t], tmp, dT_list[i][t])

                    _h.inplace_act_func_deriv['sigmoid'](T_list[i][t], dT_list[i][t])
                    _h.inplace_act_func_deriv[self.activation](H_list[i][t], dH_list[i][t])
                else:
                    _h.mult_tt(dY_list[i][t], T_list[i][t], dH_list[i][t])
                    tmp = _h.ones(dH_list[i][t].shape)
                    _h.subtract_tt(tmp, T_list[i][t], tmp)
                    _h.mult_tt(dY_list[i][t], tmp, dY_list[i-1][t])

                    _h.subtract_tt(H_list[i][t], Y_list[i-1][t], tmp)
                    _h.mult_tt(dY_list[i][t], tmp, dT_list[i][t])

                    _h.inplace_act_func_deriv['sigmoid'](T_list[i][t], dT_list[i][t])
                    _h.inplace_act_func_deriv[self.activation](H_list[i][t], dH_list[i][t])
                    _h.dot_add_mm(dT_list[i][t], R_T[i], dY_list[i-1][t])
                    _h.dot_add_mm(dH_list[i][t], R_H[i], dY_list[i-1][t])

        for t in range(inputs.shape[0] - 2, -1, -1):
            _h.dot_add_mm(dT_list[0][t + 1], R_T[0], doutputs[t])
            _h.dot_add_mm(dH_list[0][t + 1], R_H[0], doutputs[t])
            tmp = _h.ones(dH_list[0][t + 1].shape)
            _h.subtract_tt(tmp, T_list[0][t + 1], tmp)
            _h.mult_add_tt(dY_list[0][t + 1], tmp, doutputs[t])
            _h.copy_to(doutputs[t], dY_list[self.recurrence_depth-1][t])

            for i in range(self.recurrence_depth-1, -1, -1):
                    if i == 0:
                        _h.mult_tt(dY_list[i][t], T_list[i][t], dH_list[i][t])
                        tmp = _h.ones(dH_list[i][t].shape)
                        _h.subtract_tt(H_list[i][t], outputs[t-1], tmp)
                        _h.mult_tt(dY_list[i][t], tmp, dT_list[i][t])

                        _h.inplace_act_func_deriv['sigmoid'](T_list[i][t], dT_list[i][t])
                        _h.inplace_act_func_deriv[self.activation](H_list[i][t], dH_list[i][t])
                    else:
                        _h.mult_tt(dY_list[i][t], T_list[i][t], dH_list[i][t])
                        tmp = _h.ones(dH_list[i][t].shape)
                        _h.subtract_tt(tmp, T_list[i][t], tmp)
                        _h.mult_tt(dY_list[i][t], tmp, dY_list[i-1][t])

                        _h.subtract_tt(H_list[i][t], Y_list[i-1][t], tmp)
                        _h.mult_tt(dY_list[i][t], tmp, dT_list[i][t])

                        _h.inplace_act_func_deriv['sigmoid'](T_list[i][t], dT_list[i][t])
                        _h.inplace_act_func_deriv[self.activation](H_list[i][t], dH_list[i][t])
                        _h.dot_add_mm(dT_list[i][t], R_T[i], dY_list[i-1][t])
                        _h.dot_add_mm(dH_list[i][t], R_H[i], dY_list[i-1][t])

        flat_inputs = flatten_time_and_features(inputs)
        flat_dinputs = flatten_time_and_features(dinputs)
        flat_dH = flatten_time(dH_list[0][:-1])
        flat_dT = flatten_time(dT_list[0][:-1])

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dH, W_H, flat_dinputs)
        _h.dot_add_mm(flat_dH, flat_inputs, dW_H, transa=True)
        _h.dot_add_mm(flat_dT, W_T, flat_dinputs)
        _h.dot_add_mm(flat_dT, flat_inputs, dW_T, transa=True)

        for i in range(self.recurrence_depth):
            dbias_tmp = _h.allocate(dbias_H[i].shape)
            flat_dH = flatten_time(dH_list[i][:-1])
            flat_dT = flatten_time(dT_list[i][:-1])
            _h.sum_t(flat_dT, axis=0, out=dbias_tmp)
            _h.add_tt(dbias_T[i], dbias_tmp, dbias_T[i])
            _h.sum_t(flat_dH, axis=0, out=dbias_tmp)
            _h.add_tt(dbias_H[i], dbias_tmp, dbias_H[i])

        for i in range(self.recurrence_depth):
            if i == 0:
                flat_outputs = flatten_time(outputs[:-2])
                flat_dH = flatten_time(dH_list[i][1:-1])
                flat_dT = flatten_time(dT_list[i][1:-1])
                _h.dot_add_mm(flat_dT, flat_outputs, dR_T[i], transa=True)
                _h.dot_add_mm(dT_list[i][0], outputs[-1], dR_T[i], transa=True)

                _h.dot_add_mm(flat_dH, flat_outputs, dR_H[i], transa=True)
                _h.dot_add_mm(dH_list[i][0], outputs[-1], dR_H[i], transa=True)
            else:
                flat_outputs = flatten_time(Y_list[i-1][:-1])
                flat_dH = flatten_time(dH_list[i][:-1])
                flat_dT = flatten_time(dT_list[i][:-1])
                _h.dot_add_mm(flat_dT, flat_outputs, dR_T[i], transa=True)
                _h.dot_add_mm(flat_dH, flat_outputs, dR_H[i], transa=True)
