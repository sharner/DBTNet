import argparse, time, logging, os, math, copy

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

from mxnet import cpu
from mxnet.gluon.block import HybridBlock
import warnings

from dbt import *
in_shapes = [(1, 3, 224, 224)]
in_types = [np.float32]
pdir = '../model/params_n1_training_clean_dbt'
parfile = "imagenet-resnet50-0009.params"
symfile = "imagenet-resnet50-symbol.json"
sf = os.path.join(pdir, symfile)
pf = os.path.join(pdir, parfile)

def load_model(symbols, params, num_gpus):
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    symbol = mx.sym.load(symbols)
    inputs = mx.sym.var('data', dtype='float16')
    net = gluon.SymbolBlock(symbol, inputs)
    net.collect_params().load(params, ctx)
    return symbol, net
    #epoch = opt.num_epochs-1
    #net.export(bfn, epoch)

def main():
    pass

symbol, net = load_model(sf, pf, 0)

if __name__ == '__main__':
    main()

