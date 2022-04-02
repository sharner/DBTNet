FROM nvcr.io/nvidia/mxnet:21.09-py3

RUN git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git \
    && cd onnx-tensorrt \
    && git checkout release/8.0 \
    && mkdir build && cd build \
    && cmake .. && make install \
    && cd ../.. && rm -rf onnx-tensorrt

RUN pip install \
    gluoncv \
    onnx


