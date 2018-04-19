# CNNIQAplusplus
PyTorch implementation of the following paper:
[Kang, Le, et al. "Simultaneous estimation of image quality and distortion via multi-task convolutional neural networks." IEEE International Conference on Image Processing IEEE, 2015:2791-2795.](https://ieeexplore.ieee.org/document/7351311/)

### Note
The optimizer is chosen as Adam here, instead of the SGD with momentum in the paper.

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python CNNIQAplusplus.py 0 config.yaml LIVE CNNIQAplusplus
```
Before training, the `im_dir` in `config.yaml` must to be specified.

### Visualization
```bash
tensorboard --logdir='./logs' --port=6006
```
## Requirements
- PyTorch 
- TensorFlow-TensorBoard if `enableTensorboard` in `config.yaml` is `True`.