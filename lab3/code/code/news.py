"""
Dataset iterator for news 2007 corpora
"""
import cPickle as pkl

import numpy
import os


def prepare_data(seqs_x, maxlen=None, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        s_x = numpy.asarray(s_x, dtype=numpy.int64)
        s_x[numpy.where(s_x >= n_words - 1)] = 1
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx] + 1, idx] = 1.

    return x, x_mask


def load_data(path=None):
    ''' 
    Loads the dataset
    '''
    #############
    # LOAD DATA #
    #############

    print 'Loading training'
    if not os.path.isfile(path):
        raise ValueError("The path provided from the trainset doesn't exist")

    with open(path) as f:
        train = pkl.load(f)

    return numpy.asarray(train)
