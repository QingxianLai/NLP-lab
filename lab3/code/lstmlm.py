'''
Language Modelling with Theano
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import argparse
import copy
import cPickle as pkl
import os
import numpy
import yaml
import time

from scipy import optimize, stats
from collections import OrderedDict
from sklearn.cross_validation import KFold

import news

datasets = {'news': (news.load_data, news.prepare_data)}


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         state_before * trng.binomial(state_before.shape,
                                                      p=0.5,
                                                      n=1,
                                                      dtype=state_before.dtype),
                         state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm': ('param_init_lstm', 'lstm_layer')}


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, orth=True):
    if nout is None:
        nout = nin
    if nout == nin and orth:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def rectifier(x):
    return tensor.maximum(0., x)


def linear(x):
    return x


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options,
                       params,
                       prefix='ff',
                       nin=None,
                       nout=None,
                       orth=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, orth=orth)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams,
            state_below,
            options,
            prefix='rconv',
            activ='lambda x: tensor.tanh(x)',
            **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                       tparams[_p(prefix, 'b')])


# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    W = numpy.concatenate(
        [norm_weight(nin, dim), norm_weight(nin, dim), norm_weight(nin, dim),
         norm_weight(nin, dim)],
        axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate(
        [ortho_weight(dim), ortho_weight(dim), ortho_weight(dim),
         ortho_weight(dim)],
        axis=1)
    params[_p(prefix, 'U')] = U
    params[_p(prefix, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params


def lstm_layer(tparams,
               state_below,
               options,
               prefix='lstm',
               mask=None,
               one_step=False,
               init_state=None,
               init_memory=None,
               **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'U')].shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)
    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_, U, b):
        preact = tensor.dot(h_, U)
        preact += x_
        preact += b

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))

        c = tensor.tanh(_slice(preact, 3, dim))
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = tensor.dot(
        state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    if one_step:
        h, c = _step(mask,
                     state_below,
                     init_state,
                     init_memory,
                     tparams[_p(prefix, 'U')],
                     tparams[_p(prefix, 'b')])
        rval = [h, c]
    else:
        rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below],
            outputs_info=[init_state, init_memory],
            non_sequences=[tparams[_p(prefix, 'U')], tparams[_p(prefix, 'b')]],
            name=_p(prefix, '_layers'),
            n_steps=nsteps)
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    # LSTM
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    # readout
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit_lstm',
                                nin=options['dim'],
                                nout=options['dim_word'],
                                orth=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'],
                                orth=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, options):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options[
        'dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    # encoder
    proj = get_layer(options['encoder'])[1](tparams,
                                            emb,
                                            options,
                                            prefix='encoder',
                                            mask=x_mask)
    proj_h = proj[0]  # use hidden state
    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams,
                                    proj_h,
                                    options,
                                    prefix='ff_logit_lstm',
                                    activ='linear')
    logit_prev = get_layer('ff')[1](tparams,
                                    emb,
                                    options,
                                    prefix='ff_logit_prev',
                                    activ='linear')
    logit = rectifier(logit_lstm + logit_prev)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams,
                               logit,
                               options,
                               prefix='ff_logit',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                               logit_shp[2]]))
    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx] + 1e-8)
    cost = cost.reshape([x.shape[0], x.shape[1]])
    cost = (cost * x_mask).sum(0)
    cost = cost.mean()

    return trng, use_noise, x, x_mask, cost


# build a sampler
def build_sampler(tparams, options, trng, use_noise):
    x = tensor.vector('x', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    if options['encoder'] == 'lstm':
        init_memory = tensor.matrix('init_memory', dtype='float32')
    else:
        init_memory = None
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source)
    emb = tensor.switch(x[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][x])
    # encoder
    rval = get_layer(options['encoder'])[1](tparams,
                                            emb,
                                            options,
                                            prefix='encoder',
                                            one_step=True,
                                            init_state=init_state,
                                            init_memory=init_memory)
    next_state = rval[0]
    if options['encoder'] == 'lstm':
        next_memory = rval[1]

    logit_lstm = get_layer('ff')[1](tparams,
                                    next_state,
                                    options,
                                    prefix='ff_logit_lstm',
                                    activ='linear')
    logit_prev = get_layer('ff')[1](tparams,
                                    emb,
                                    options,
                                    prefix='ff_logit_prev',
                                    activ='linear')
    logit = rectifier(logit_lstm + logit_prev)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams,
                               logit,
                               options,
                               prefix='ff_logit',
                               activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..',
    inps = [x, init_state]
    if options['encoder'] == 'lstm':
        inps += [init_memory]
    outs = [next_probs, next_sample, next_state]
    if options['encoder'] == 'lstm':
        outs += [next_memory]

    f_next = theano.function(inps, outs, name='f_next')
    print 'Done'

    return f_next


# generate sample
def gen_sample(tparams, f_next, options, trng, maxlen=30, cue=None):
    # no beam search
    sample = []
    sample_score = 0

    next_state = numpy.zeros((1, options['dim'])).astype('float32')
    next_memory = numpy.zeros((1, options['dim'])).astype('float32')

    next_w = (-1 * numpy.ones((1,))).astype('int64')

    cidx = 0

    for ii in xrange(maxlen):
        if options['encoder'] == 'lstm':
            next_p, next_w, next_state, next_memory = f_next(next_w, next_state,
                                                             next_memory)
        else:
            next_p, next_w, next_state = f_next(next_w, next_state)

        if cue is not None and cidx < len(cue):
            next_w[0] = cue[cidx]
            cidx += 1

        sample.append(next_w[0])
        sample_score -= next_p[0, next_w[0]]
        if next_w[0] == 0:
            break

    return sample, sample_score


def pred_probs(f_log_probs, prepare_data, data, iterator, verbose=False):
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 1)).astype('float32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask = prepare_data([valid[0][t] for t in valid_index])
        pred_probs = f_log_probs(x, mask)
        probs[valid_index] = pred_probs[:, None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed' % (n_done, n_samples)

    return probs


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g**2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup + rg2up)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud**2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr],
                               [],
                               updates=ru2up + param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr],
                               [],
                               updates=updates,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g**2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup + rgup + rg2up)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg**2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr],
                               [],
                               updates=updir_new + param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def perplexity(f_cost, lines, worddict, options, verbose=False):
    n_lines = len(lines)
    cost = 0.
    n_words = 0.

    for i, line in enumerate(lines):
        # get array from line
        seq = []
        for w in line.strip().split():
            if w in worddict:
                seq.append(worddict[w])
            else:
                seq.append(1)  # unknown
        seq = [s if s < options['n_words'] else 1 for s in seq]
        n_words += len(seq) + 1
        x = numpy.array(seq + [0]).astype('int64').reshape([len(seq) + 1, 1])
        x_mask = numpy.ones((len(seq) + 1, 1)).astype('float32')

        cost_one = f_cost(x, x_mask) * (len(seq) + 1)
        cost += cost_one
        if verbose:
            print 'Sentence ', i, '/', n_lines, ' (', seq.mean(), '):', 2**(
                cost_one / len(seq) / numpy.log(2)), ', ', cost_one / len(seq)
    cost = cost / n_words
    return cost


def train(opts):
    # opts - model options

    file_name = "start_at_" + time.strftime("%H_%M_%S") + ".output"
    f = open(file_name, 'w+')


    if opts['dictionary']:
        with open(opts['dictionary'], 'rb') as f:
            worddict = pkl.load(f)
        word_idict = dict()
        for kk, vv in worddict.iteritems():
            word_idict[vv] = kk

    # reload options
    if opts['reload_'] and os.path.exists(opts['saveto']):
        with open('%s.pkl' % opts['saveto'], 'rb') as f:
            reloaded_options = pkl.load(f)
            opts.update(reloaded_options)

    # get dataset
    print 'Loading data'
    print >> f,'Loading data'

    load_data, prepare_data =get_dataset(opts['dataset'])
    train = load_data(path=opts['train_path'])

    # build computational graph
    print 'Building model'
    print >> f,'Building model'
    params = init_params(opts)
    # reload parameters
    if opts['reload_'] and os.path.exists(opts['saveto']):
        params = load_params(opts['saveto'], params)

    tparams = init_tparams(params)
    trng, use_noise, \
          x, x_mask, \
          cost = \
          build_model(tparams, opts)
    inps = [x, x_mask]

    # weight noise
    param_noise = opts['param_noise']
    if param_noise > 0.:
        noise_update = []
        noise_tparams = OrderedDict()
        for kk, vv in tparams.iteritems():
            noise_tparams[kk] = theano.shared(vv.get_value() * 0.)
            noise_update.append((noise_tparams[kk], param_noise * trng.normal(
                vv.shape)))
        f_noise = theano.function([], [], updates=noise_update)
        add_update = []
        rem_update = []
        for vv, nn in zip(tparams.values(), noise_tparams.values()):
            add_update.append((vv, vv + nn))
            rem_update.append((vv, vv - nn))
        f_add_noise = theano.function([], [], updates=add_update)
        f_rem_noise = theano.function([], [], updates=rem_update)

    print 'Building sampler'
    print >> f, 'Building sampler'
    f_next = build_sampler(tparams, opts, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    print  >> f, 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost)
    print 'Done'
    print  >> f, 'Done'

    decay_c = opts['decay_c']
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv**2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    print  >> f, 'Building f_cost...',
    f_cost = theano.function(inps, cost)
    print 'Done'
    print  >> f, 'Done'

    # get gradients
    print 'Computing gradient...',
    print  >> f, 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))

    # f_grad = theano.function(inps, grads)
    print 'Done'
    print  >> f, 'Done'

    # TODO: ASSIGNMENT: Implement gradient clipping here.
    ita = 1
    print grads
    print  >> f, grads
    grad_norm = tensor.sqrt(tensor.sum([(g ** 2).sum() for g in grads]))
    grads = [tensor.switch(tensor.ge(grad_norm, ita), g/grad_norm * ita,  g) for g in grads]

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    print  >> f, 'Building optimizers...',
    f_grad_shared, f_update = eval(opts['optimizer'])(lr, tparams, grads, inps, cost)
    print 'Done'
    print  >> f, 'Done'

    print 'Optimization'
    print  >> f, 'Optimization'

    history_errs = []
    # reload history
    if opts['reload_'] and os.path.exists(opts['saveto']):
        history_errs = list(numpy.load(opts['saveto'])['history_errs'])
    best_p = None
    bad_count = 0

    batch_size = opts['batch_size']
    if opts['validFreq'] == -1:
        validFreq = len(train) / batch_size
    if opts['saveFreq'] == -1:
        saveFreq = len(train) / batch_size
    if opts['sampleFreq'] == -1:
        sampleFreq = len(train) / batch_size

    # training iterator
    kf_train = KFold(len(train), n_folds=len(train)/batch_size, shuffle=False)

    # load validation and test
    if opts['valid_text']:
        valid_lines = []
        with open(opts['valid_text'], 'r') as f:
            for l in f:
                valid_lines.append(l.lower())
        n_valid_lines = len(valid_lines)
    if opts['test_text']:
        test_lines = []
        with open(opts['test_text'], 'r') as f:
            for l in f:
                test_lines.append(l.lower())
        n_test_lines = len(test_lines)

    f =

    uidx = 0
    estop = False
    Valid_pplx = []
    Test_pplx = []
    for eidx in xrange(opts['max_epochs']):
        n_samples = 0

        for _, index in kf_train:
            n_samples += len(index)
            uidx += 1
            use_noise.set_value(1.)

            x = train[index]
            x, x_mask = prepare_data(x.tolist(), maxlen=opts['maxlen'], n_words=opts['n_words'])

            #if x is None:
            #    print 'Minibatch with zero sample under length ', opts['maxlen']
            #    continue

            if opts['param_noise'] > 0.:
                f_noise()
                f_add_noise()
            cost = f_grad_shared(x, x_mask)
            if opts['param_noise'] > 0.:
                f_rem_noise()
            f_update(opts['lrate'])

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, opts['dispFreq']) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

            if numpy.mod(uidx, opts['saveFreq']) == 0:
                print 'Saving...',
                print  >> f, 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(opts['saveto'], history_errs=history_errs, **params)
                pkl.dump(opts, open('%s.pkl' % opts['saveto'], 'wb'))
                print 'Done'
                print  >> f, 'Done'

            if numpy.mod(uidx, opts['sampleFreq']) == 0:
                for jj in xrange(5):
                    sample, score = gen_sample(tparams,
                                               f_next,
                                               opts,
                                               trng=trng,
                                               maxlen=30)
                    print 'Sample ', jj, ': ',
                    print  >> f, 'Sample ', jj, ': ',
                    for vv in sample:
                        if vv in word_idict:
                            print word_idict[vv],
                            print  >> f, word_idict[vv],
                        else:
                            print 'UNK',
                            print  >> f, 'UNK',
                    print

            if numpy.mod(uidx, opts['validFreq']) == 0:
                print "Computing Dev/Test Perplexity"
                print  >> f, "Computing Dev/Test Perplexity"
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0

                if opts['valid_text'] is not None:
                    valid_err = perplexity(f_cost, valid_lines, worddict,
                                           opts)
                if opts['test_text'] is not None:
                    test_err = perplexity(f_cost, test_lines, worddict,
                                          opts)

                history_errs.append([valid_err, test_err])

                if len(history_errs) > 1:
                    if uidx == 0 or valid_err <= numpy.array(
                            history_errs)[:, 0].min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                    if eidx > opts['patience'] and valid_err >= numpy.array(
                            history_errs)[:-opts['patience'], 0].min():
                        bad_counter += 1
                        if bad_counter > opts['patience']:
                            print 'Early Stop!'
                            print  >> f, 'Early Stop!'
                            estop = True
                            break

                Valid_pplx.append(valid_err)
                Test_pplx.append(test_err)
                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
                print  >> f, 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        print 'Seen %d samples' % n_samples
        print  >> f, 'Seen %d samples' % n_samples

        if estop:
            break
    print "====== Valid pplx: ===========\n", Valid_pplx, "\n========== Test pplx: ==========\n", Test_pplx, "\n================================\n"
    print  >> f, "====== Valid pplx: ===========\n", Valid_pplx, "\n========== Test pplx: ==========\n", Test_pplx, "\n================================\n"
    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    if opts['valid_text'] is not None:
        valid_err = perplexity(f_cost, valid_lines, worddict, opts)
    if opts['test_text'] is not None:
        test_err = perplexity(f_cost, test_lines, worddict, opts)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    print  >> f, 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    params = copy.copy(best_p)
    numpy.savez(opts['saveto'],
                zipped_params=best_p,
                train_err=train_err,
                valid_err=valid_err,
                test_err=test_err,
                history_errs=history_errs,
                **params)
    f.close()
    return train_err, valid_err, test_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("yml_location",
                        help="Location of the yml file",
                        type=argparse.FileType('r'))
    args = parser.parse_args()
    options = yaml.load(args.yml_location)
    best_train, best_valid, best_test = train(options)
