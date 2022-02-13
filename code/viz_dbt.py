import argparse, time, logging, os, math, copy

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import matplotlib.pyplot as plt

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import viz
from mxnet import cpu

from dbt import *

# CLI
parser = argparse.ArgumentParser(description='vizualize given data')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                    help='the training data')
parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                    help='the index of training data')
parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                    help='the index of validation data')
parser.add_argument('--use-rec', action='store_true',
                    help='use image record iter for data input. default is false.')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--parameters', type=str, required=True,
                    help='parameters to load for vizualizer')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to init gamma of the last BN layer in each bottleneck to 0.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
parser.add_argument('--nclasses', type=int, default=1000, help='number of classes')
parser.add_argument('--num-training-samples', type=int, default=1281167, help='number of training samples')

opt = parser.parse_args()

model_name = opt.model
input_size = opt.input_size
classes = opt.nclasses
classes = 200 # workaround
num_training_samples = opt.num_training_samples
batch_size_per_gpu = opt.batch_size
batch_size = opt.batch_size
num_workers = opt.num_workers

num_gpus=opt.num_gpus
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

net.load_parameters(opt.parameters, ctx=context, allow_missing=True,  ignore_extra=True)
# classes = 200 

with net.name_scope():
    newoutput = nn.HybridSequential(prefix='')
    newoutput.add(nn.Conv2D(classes, kernel_size=1, padding=0, use_bias=True))
    net.myoutput = newoutput#nn.Conv2D(classes, kernel_size=3, padding=1, use_bias=True)#nn.Dense(classes)
net.myoutput[0].initialize(mx.init.Xavier(), ctx = context)
net.collect_params().reset_ctx(context)
net.hybridize()

net.cast(opt.dtype)

# Two functions for reading data from record file or raw images
def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, batch_fn

def get_data_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, val_data, batch_fn

if opt.use_rec:
    train_data, val_data, batch_fn = get_data_rec(opt.rec_train, opt.rec_train_idx,
                                                  opt.rec_val, opt.rec_val_idx,
                                                  batch_size, num_workers)
else:
    train_data, val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

def test(ctx, val_data):
    if opt.use_rec:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False))[0] for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

def train(ctx):
    #train_metric_name, train_metric_score = train_metric.get()
    #err_top1_val, err_top5_val = test(ctx, val_data)
    pass

def display(context):
    for batch in val_data:
        for idx in range(batch.data[0].shape[0]):
            matrix = nd.NDArray.asnumpy(batch.label[0][idx])
            print("index {} label is {}".format(idx, matrix.item()))
            img = nd.NDArray.asnumpy(batch.data[0][idx])
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.savefig("figure{}.png".format(idx))
        break

def main():
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    display(context)

if __name__ == '__main__':
    main()

