import argparse, time, logging, os, math, copy

import numpy as np
import mxnet as mx
import logging
logging.basicConfig(level=logging.INFO)

# CLI
parser = argparse.ArgumentParser(description='Export given model/paramerters to ONNX')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--load-params', type=str, required=True,
                    help='path of parameters to load from.')
parser.add_argument('--load-syms', type=str, required=True,
                    help='path of symbols to load from.')
parser.add_argument('--onnx-file', type=str, default=None,
                    help='onnx', required=True);
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--nclasses', type=int, default=1000, help='number of classes')
opts = parser.parse_args()

def main():
    in_shapes = [(opts.nclasses, 3, opts.input_size, opts.input_size)]
    in_types = [np.float16]

    coverted_file = mx.onnx.export_model(opts.load_syms,
                                         opts.load_params,
                                         in_shapes,
                                         in_types,
                                         opts.onnx_file)

if __name__ == '__main__':
    main()

