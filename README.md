# CNNIQAplusplus
PyTorch 1.3 implementation of the following paper:
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
tensorboard --logdir=tensorboard_logs --port=6006 # in the server (host:port)
ssh -p port -L 6006:localhost:6006 user@host # in your PC. See the visualization in your PC
```
## Requirements
```bash
conda create -n reproducibleresearch pip python=3.6
source activate reproducibleresearch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
source deactive
```
- Python 3.6.8
- PyTorch 1.3.0
- TensorboardX 1.9, TensorFlow 2.0.0
- [pytorch/ignite 0.2.1](https://github.com/pytorch/ignite)

Note: You need to install the right CUDA version.

## TODO (If I have free time)
- Simplify the code
- Report results on some common databases
- etc.