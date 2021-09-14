FROM nvcr.io/nvidia/mxnet:21.08-py3

RUN git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git \
    && cd onnx-tensorrt && mkdir build && cd build \
    && cmake .. && make install \
    && cd ../.. && rm -rf onnx-tensorrt

RUN pip install \
    gluoncv


