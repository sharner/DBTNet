import argparse, time, logging, os, math, copy

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

from mxnet import cpu
from mxnet.gluon.block import HybridBlock

from dbt import *

# CLI
parser = argparse.ArgumentParser(description='Export given model/paramerters to ONNX')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--load-params', type=str, required=True,
                    help='path of parameters to load from.')
parser.add_argument('--num-epochs', type=int, required=True,
                    help='number training epochs')
parser.add_argument('--export-dir', type=str,
                    default=None,
                    help='export directory');
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to init gamma of the last BN layer in each bottleneck to 0.')
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--logging-file', type=str, default='export_imagenet.log',
                    help='name of training log file')
parser.add_argument('--nclasses', type=int, default=1000, help='number of classes')
opt = parser.parse_args()


filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt)

batch_size_per_gpu = opt.batch_size
input_size = opt.input_size
classes = 1000 # number of classes in imagenet parameters
batch_size = opt.batch_size

model_name = opt.model
num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

if opt.last_gamma:
    kwargs['last_gamma'] = True

net = dbt(num_layers = 50, batch_size=batch_size_per_gpu, width=input_size, **kwargs)
#net.collect_params().cast('float16')

net.cast('float16')

#for p in net.collect_params().values():
#    p.grad_req = 'add'

ft_params = '../model/params_imagenet_dbt/dbt_imagenet.params'
#net.load_parameters(ft_params, ctx=context, allow_missing=True,  ignore_extra=True)
classes = opt.nclasses # number of classes for fine-tuning
# classes = 200 

with net.name_scope():
    newoutput = nn.HybridSequential(prefix='')
    newoutput.add(nn.Conv2D(classes, kernel_size=1, padding=0, use_bias=True))
    net.myoutput = newoutput#nn.Conv2D(classes, kernel_size=3, padding=1, use_bias=True)#nn.Dense(classes)
net.myoutput[0].initialize(mx.init.Xavier(), ctx = context)
#net.collect_params().reset_ctx(context)
net.hybridize()

net.cast(opt.dtype)
net.load_parameters(opt.load_params, ctx = context)

if not opt.export_dir:
    opt.export_dir, _ = os.split(opt.load_params)

if not os.path.exists(opt.export_dir):
    os.makedirs(opt.export_dir)
    
def main():
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    bfn = os.path.join(opt.export_dir, "imagenet-{}".format(model_name))
    epoch = opt.num_epochs-1
    net.export(bfn, epoch)

if __name__ == '__main__':
    main()

