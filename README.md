# qResNet

qResNet, which stands for quantized ResNet, facilitates the quantization of the [ResNet-18](https://huggingface.co/microsoft/resnet-18) model using arbitrary-precision floating-point numbers.

## Example
In the example below, the ResNet-18 model was quantized using a 15-bit floating-point representation (1 bit for the sign, 10 bits for the mantissa, and 4 bits for the exponent). We used a 128x128 tiger image for inference, and the results were as follows:

<img src="tiger128.jpg">

```
root@1723a04c0176:~/qresnet# ./bin/main 
[util.hpp L24] image.rows : 128, image.cols : 128
Current Precision : total 15-bit
=== Predicted Result ===
Top 1 : tiger, Panthera tigris with logit 41.00000000000000000000
Top 2 : tiger cat with logit 34.37500000000000000000
Top 3 : jaguar, panther, Panthera onca, Felis onca with logit 29.09375000000000000000
Top 4 : leopard, Panthera pardus with logit 26.48437500000000000000
Top 5 : snow leopard, ounce, Panthera uncia with logit 25.06250000000000000000
Elapsed time : 4362ms
```
## Dependency
- [fmt](https://github.com/fmtlib/fmt) : ```sudo apt install libfmt-dev```
- [protobuf(protoc)](https://github.com/protocolbuffers/protobuf)
- CMake
- OpenCV : ```sudo apt install libopencv-dev```
- [GMP & MPFR](https://www.mpfr.org/) (For Korean readers, a guide is available [here](https://bgreat.tistory.com/52))

The base used is the ```nvcr.io/nvidia/pytorch:23.06-py3``` docker image, chosen for its inclusion of g++-11, protoc, and CMake. GMP and MPFR were built manually.

## How to Build
Ensure all dependencies are installed before proceeding with the build.
```
mkdir -p build
cd build
cmake ..
cmake --build . -j
cd ..
```


## Operator Support Matrix

| Operator                           | Supported  |         Details        |
|-------------------------------------|---|--------------------------------|
| Add                                 | Y | Implemented using a threadpool |
| BatchNorm2D                         | Y | Implemented using a threadpool |
| Convolution2D                       | Y | Implemented using a threadpool |
| ConvTranspose2D                     | Y | Implemented using a threadpool |
| Flatten                             | Y |                                |
| GAP(Global Average Pooling)         | Y | Implemented using a threadpool |
| Gemm(General Matrix Multiplication) | Y | Implemented using a threadpool |
| Identity                            | Y |                                |
| InstanceNorm2D                      | Y | Implemented using a threadpool |
| Leaky ReLU                          | Y | Implemented using a threadpool |
| MaxPool2D                           | Y | Implemented using a threadpool |
| ReLU                                | Y | Implemented using a threadpool |
| Tanh                                | Y | Implemented using a threadpool |
