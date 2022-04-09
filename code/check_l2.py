import argparse, time, logging, os, math, copy

import numpy as np
import mxnet as mx

from dbt import *
in_shapes = [(1, 3, 224, 224)]
in_types = [np.float32]
pdir = '../model/params_n1_training_clean_dbt'
parfile = "imagenet-resnet50-0009.params"
symfile = "imagenet-resnet50-symbol.json"
sf = os.path.join(pdir, symfile)
pf = os.path.join(pdir, parfile)

def diag_rep(ngroups):
  return mx.sym.linalg_makediag(mx.sym.ones(ngroups)).reshape((1, 1, ngroups, ngroups))

def check_l2(array, width, num_gpus=0):
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    arr = mx.nd.array(array)
    tmp = mx.sym.Variable('tmp')
    tmp1 = mx.sym.Variable('tmp1')
    tmp1 = tmp + 1.0e-12
    tmp = mx.sym.L2Normalization(tmp.reshape((-1,width*width)), mode='instance')
    resh = tmp1.reshape((-1, width*width))
    div1 = resh.norm(ord=2, axis=1) + 1.0e-10
    mult = 1./div1.reshape((1, -1)).transpose()
    tmp1 = mx.sym.broadcast_mul(tmp1.reshape((-1,width*width)), mult)
    diff = tmp - tmp1
    e = diff.bind(ctx[0], {'tmp':arr})

    groups = 16
    channels = 100
    cdg = np.int((channels/groups)*(channels/groups))
    
    d1 = mx.sym.ones(groups).diag().reshape((1, 1, groups, groups))
    d11 = mx.sym.tile(d1, (1, cdg, 1, 1))
    e11 = d11.bind(ctx[0], {})
    y2 = e11.forward()

    d2 = diag_rep(groups)
    d21 = mx.sym.tile(d2, (1, cdg, 1, 1))
    diff1 = d11 - d21
    e1 = diff1.bind(ctx[0], {})
    y = e.forward()
    y1 = e1.forward()
    return y, y1, y2

def main(ntrials = 1000, width=224):
    passes = 0
    fail = 0
    eps = 1.0e-5
    for i in range(ntrials):
        next = mx.nd.random.uniform(0, 1., shape=(1, 3, width, width))
        y, y1, y2 = check_l2(next, width)
        n = y[0].norm()
        n1 = y1[0].norm()
        if n[0] > eps:
            print('Fail: {} -> norm {}'.format(next, n[0]))
            fail += 1
        elif n1[0] > eps:
            print('Fail: diag diff -> norm {}'.format(n1[0]))
        else:
            passes += 1
    print('Passes {} Failures {}'.format(passes, fail))

# symbol, net = load_model(sf, pf, 0)

if __name__ == '__main__':
    main()

