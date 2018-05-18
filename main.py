"""
PyTorch 0.4 implementation of the following paper:
Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.
 Requirements:
    TensorboardX (https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
    Tensorboard: `pip install tensorflow` (or just install tensorboard without the rest of tensorflow)
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=tensorboard_logs --port=6006
    ```
    Run the main.py:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0
    ```

 Implemented by Dingquan Li
 Email: dingquanli@pku.edu.cn
 Date: 2018/5/18
"""

from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from IQADataset import IQADataset
import numpy as np
from scipy import stats
import os, yaml

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loss_fn(y_pred, y):
    return config['alpha_q'] * F.l1_loss(y_pred[0], y[0]) + \
           config['alpha_d'] * F.cross_entropy(y_pred[1], y[2].long().squeeze(1))


class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._y_pred = []
        self._y      = []
        self._y_std  = []

    def update(self, output):
        pred, y = output

        self._y.append(y[0])
        self._y_std.append(y[1])
        self._y_pred.append(torch.mean(pred[0]))

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
        outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

        return srocc, krocc, plcc, rmse, mae, outlier_ratio

class IDCPerformance(Metric):
    """
    Accuracy of image distortion classification.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._d_pred = []
        self._d      = []

    def update(self, output):
        pred, y = output

        self._d.append(y[2])
        self._d_pred.append(torch.max(torch.mean(pred[1], 0), 0)[1])

    def compute(self):
        acc = np.mean([self._d[i] == self._d_pred[i].float() for i in range(len(self._d))])

        return acc


class CNNIQAplusnet(nn.Module):
    def __init__(self, n_classes, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAplusnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1   = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2   = nn.Linear(n1_nodes, n2_nodes)
        self.fc3_q = nn.Linear(n2_nodes, 1)
        self.fc3_d = nn.Linear(n2_nodes, n_classes)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #
        h = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  #
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3_q(h)
        d = self.fc3_d(h)

        return q, d


class CNNIQAplusplusnet(nn.Module):
    def __init__(self, n_distortions, ker_size=3, n1_kers=8, pool_size=2, n2_kers=32, n1_nodes=128, n2_nodes=512):
        super(CNNIQAplusplusnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n1_kers, ker_size)
        self.pool1 = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(n1_kers, n2_kers, ker_size)
        self.fc1   = nn.Linear(2 * n2_kers, n1_nodes)
        self.fc2   = nn.Linear(n1_nodes, n2_nodes)
        self.fc3_q = nn.Linear(n2_nodes, 1)
        self.fc3_d = nn.Linear(n2_nodes, n_distortions)

    def forward(self, x):
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #
        h  = self.conv1(x)
        h  = self.pool1(h)

        h  = self.conv2(h)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  #
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = F.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3_q(h)
        d  = self.fc3_d(h)

        return q, d


def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def create_summary_writer(model, data_loader, log_dir='tensorboard_logs'):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(train_batch_size, epochs, lr, weight_decay, model_name, config, exp_id, log_dir, trained_model_file, save_result_file, disable_gpu=False):
    if config['test_ratio']:
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")
    if model_name == 'CNNIQAplus' or model_name == 'CNNIQA':
        model = CNNIQAplusnet(n_distortions=config['n_distortions'],
                            ker_size=config['kernel_size'],
                            n_kers=config['n_kernels'],
                            n1_nodes=config['n1_nodes'],
                            n2_nodes=config['n2_nodes'])
    else:
        model = CNNIQAplusplusnet(n_distortions=config['n_distortions'],
                                ker_size=config['kernel_size'],
                                n1_kers=config['n1_kernels'],
                                pool_size=config['pool_size'],
                                n2_kers=config['n2_kernels'],
                                n1_nodes=config['n1_nodes'],
                                n2_nodes=config['n2_nodes'])
    writer = create_summary_writer(model, train_loader, log_dir)
    model = model.to(device)
    print(model)
    # if multi_gpu and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     train_batch_size *= torch.cuda.device_count()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    global best_criterion
    best_criterion = -1  # SROCC>=-1
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance(),
                                                     'IDC_performance': IDCPerformance()},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        Acc = metrics['IDC_performance']
        print("Validation Results - Epoch: {} Acc:  {:.2f}% SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(engine.state.epoch, 100 * Acc, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)
        writer.add_scalar("validation/OR", OR, engine.state.epoch)
        writer.add_scalar("validation/Acc", Acc, engine.state.epoch)
        global best_criterion
        global best_epoch
        if SROCC > best_criterion:
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            torch.save(model.state_dict(), trained_model_file)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            Acc = metrics['IDC_performance']
            print("Testing Results    - Epoch: {} Acc:  {:.2f}% SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(engine.state.epoch, 100 * Acc, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)
            writer.add_scalar("testing/OR", OR, engine.state.epoch)
            writer.add_scalar("testing/Acc", Acc, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        if config["test_ratio"] > 0:
            model.load_state_dict(torch.load(trained_model_file))
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            Acc = metrics['IDC_performance']
            global best_epoch
            print("Final Test Results - Epoch: {} Acc:  {:.2f}% SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                .format(best_epoch, 100 * Acc, SROCC, KROCC, PLCC, RMSE, MAE, 100 * OR))
            np.save(save_result_file, (Acc, SROCC, KROCC, PLCC, RMSE, MAE, OR))

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='LIVE', type=str,
                        help='database name (default: LIVE)')
    parser.add_argument('--model', default='CNNIQAplusplus', type=str,
                        help='model name (default: CNNIQAplusplus)')
    # parser.add_argument('--resume', default=None, type=str,
    #                     help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    # parser.add_argument('--multi_gpu', action='store_true',
    #                     help='flag whether to use multiple GPUs')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('model: ' + args.model)
    config.update(config[args.database])
    config.update(config[args.model])

    log_dir = args.log_dir + '/EXP{}-{}-{}-lr={}-train'.format(args.exp_id, args.database, args.model, args.lr)
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)
    ensure_dir('results')
    save_result_file = 'results/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, args.model, config, args.exp_id,
        log_dir, trained_model_file, save_result_file, args.disable_gpu)