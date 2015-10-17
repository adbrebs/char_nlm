import sys

import numpy as np

import theano
from theano import shared, function
from theano.sandbox.rng_mrg import MRG_RandomStreams
floatX = theano.config.floatX
import theano.tensor as t


def relu(x):
    return 0.5 * (x + abs(x))


class FF:
    """
    Feed forward model with an embedding layer, a relu hidden layer and
    a softmax output layer.
    """
    def __init__(self, seq_len, emb_size, n_hidden, size_dict, lr):
        self.seq_len = seq_len

        # Parameters
        w_emb = shared(np.random.normal(
            0, 0.01, size=(size_dict, emb_size)).astype(dtype=floatX))

        w_hidden = shared(np.random.normal(
            0, 0.01, size=(seq_len * emb_size, n_hidden)).astype(dtype=floatX))
        b_hidden = shared(np.random.normal(
            0, 0.01, size=(n_hidden,)).astype(dtype=floatX))

        w_out = shared(np.random.normal(
            0, 0.01, size=(n_hidden, size_dict)).astype(dtype=floatX))
        b_out = shared(np.random.normal(
            0, 0.01, size=(size_dict,)).astype(dtype=floatX))

        # Graph
        x = t.imatrix('x')
        target = t.ivector('y')

        emb = w_emb[x]
        buff = relu(t.dot(emb.reshape((x.shape[0], -1)), w_hidden) +
                     b_hidden)
        y_hat = t.nnet.softmax((t.dot(buff, w_out)) + b_out)

        cost = t.nnet.categorical_crossentropy(y_hat, target).mean()

        params = [w_emb, w_hidden, b_hidden, w_out, b_out]

        grads = t.grad(cost, params)
        updates = [(w, w - lr * p) for w, p in zip(params, grads)]

        self.fun_cost = theano.function([x, target], cost, updates=updates)

        # Sampling function
        rng = MRG_RandomStreams(42)
        next_char = t.argmax(rng.multinomial(pvals=y_hat), axis=1)
        self.fun_predict = theano.function([x], next_char)

    def train(self, batch, target, i):
        return self.fun_cost(batch, target)

    def sample(self, dictionary, inv_dictionary,
               begin="MADAME PERNELLE", seq_print_len=1000):
        """
        Sample from the model.
        seq_print_len is the length of the sequence to be printed.
        """
        begin = begin[:self.seq_len]

        prev = np.array([dictionary[ch] for ch in begin], dtype='int16')
        prev = prev[None, :]

        print begin,
        for i in xrange(seq_print_len):
            a = self.fun_predict(prev)[0]
            sys.stdout.write(inv_dictionary[a])
            prev = np.roll(prev, -1)
            prev[:, -1] = a


class RNN:
    """
    Recurrent neural network with an embedding layer, a hidden layer of
    relus and a softmax output layer.
    """
    def __init__(self, seq_len, emb_size, n_hidden, size_dict, batch_size, lr):

        self.seq_len = seq_len
        self.batch_size = batch_size

        w_emb = shared(np.random.normal(
            0, 0.01, size=(size_dict, emb_size)).astype(dtype=floatX))

        w_in = shared(np.random.normal(
            0, 0.01, size=(emb_size, n_hidden)).astype(dtype=floatX))

        b_in = shared(np.random.normal(
            0, 0.01, size=(n_hidden,)).astype(dtype=floatX))

        # IRNN initialization
        # w_hidden = shared(np.eye(n_hidden).astype(dtype=floatX))
        w_hidden = shared(np.random.normal(
            0, 0.01, size=(n_hidden, n_hidden)).astype(dtype=floatX))

        b_hidden = shared(np.random.normal(
            0, 0.01, size=(n_hidden,)).astype(dtype=floatX))

        w_out = shared(np.random.normal(
            0, 0.01, size=(n_hidden, size_dict)).astype(dtype=floatX))
        b_out = shared(np.random.normal(
            0, 0.01, size=(size_dict,)).astype(dtype=floatX))

        self.params = [w_emb, w_in, b_in, w_hidden, b_hidden, w_out, b_out]

        x = t.imatrix('x')
        y = t.ivector('y')

        self.init_state = shared(np.zeros((batch_size, n_hidden), dtype=floatX))
        buff = self.init_state
        for e in xrange(seq_len):
            emb = w_emb[x[:, e]]
            emb = emb.reshape((x.shape[0], -1))
            buff = relu(t.dot(emb, w_in) + t.dot(buff, w_hidden) + b_hidden)

        y_hat = t.nnet.softmax((t.dot(buff, w_out)) + b_out)

        cost = t.nnet.categorical_crossentropy(y_hat, y).mean()

        params = [w_emb, w_in, w_hidden, b_hidden, w_out, b_out]

        grads = t.grad(cost, params)
        updates = [(self.init_state, buff)] + \
                  [(w, w - lr * p) for w, p in zip(params, grads)]

        self.fun_cost = function([x, y], cost, updates=updates)

        rng = MRG_RandomStreams(42)
        next_char = t.argmax(rng.multinomial(pvals=y_hat), axis=1)
        self.fun_predict = function([x], next_char)

    def train(self, batch, target, i):
        # Once in a while, we reset the initial set
        if i % 1000:
            self.init_state.set_value(
                np.zeros_like(self.init_state.get_value()))
        return self.fun_cost(batch, target)

    def sample(self, dictionary, inv_dictionary,
               begin="MADAME PERNELLE", seq_print_len=1000):
        """
        Sample from the model.
        seq_print_len is the length of the sequence to be printed.
        """

        begin = begin[:self.seq_len]

        # Save the state of the model and set it to zero
        saved_state = self.init_state.get_value()
        self.init_state.set_value(np.zeros_like(saved_state))

        prev = np.array([dictionary[ch] for ch in begin], dtype='int16')
        prev = np.repeat(prev[None,:], self.batch_size, axis=0)

        print begin,
        for i in xrange(seq_print_len):
            a = self.fun_predict(prev)[0]
            sys.stdout.write(inv_dictionary[a])
            prev = np.roll(prev, -1, axis=1)
            prev[:, -1] = a

        # Restore training state
        self.init_state.set_value(saved_state)

