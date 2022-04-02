import argparse, time, logging, os, math, copy

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

from mxnet import cpu
from mxnet.gluon.block import HybridBlock
import warnings

from dbt import *

# CLI
parser = argparse.ArgumentParser(description='Export given model/paramerters to ONNX')
parser.add_argument('--load-params', type=str, required=True,
                    help='path of parameters to load from.')
parser.add_argument('--load-symbols', type=str, required=True,
                    help='path of symbols to load from.')
parser.add_argument('--export-dir', type=str,
                    default=None, help='export directory');
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--logging-file', type=str, default='export_imagenet.log',
                    help='name of training log file')
opt = parser.parse_args()


filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt)

num_gpus = opt.num_gpus
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

if not opt.export_dir:
    opt.export_dir, _ = os.split(opt.load_params)

if not os.path.exists(opt.export_dir):
    os.makedirs(opt.export_dir)

def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = gluon.nn.SymbolBlock.imports(opt.load_symbols,
                                           ['data'], opt.load_params,
                                           ctx=context)
    return net
    #epoch = opt.num_epochs-1
    #net.export(bfn, epoch)

def main():
    pass

if __name__ == '__main__':
    main()

