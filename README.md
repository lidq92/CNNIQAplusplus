# CNNIQAplusplus
PyTorch 0.4 implementation of the following paper:
[Kang, Le, et al. "Simultaneous estimation of image quality and distortion via multi-task convolutional neural networks." IEEE International Conference on Image Processing IEEE, 2015:2791-2795.](https://ieeexplore.ieee.org/document/7351311/)

### Note
The optimizer is chosen as Adam here, instead of the SGD with momentum in the paper.

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0 --database=LIVE --model=CNNIQAplusplus
```
Before training, the `im_dir` in `config.yaml` must to be specified.

### Visualization
```bash
tensorboard --logdir=tensorboard_logs --port=6006
```
## Requirements
- PyTorch 0.4
- TensorboardX 1.2, TensorFlow-TensorBoard
- [pytorch/ignite](https://github.com/pytorch/ignite)