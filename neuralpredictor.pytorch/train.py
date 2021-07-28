import logging
import random
from argparse import ArgumentParser
from loss import MSELoss_and_TripletLoss, select_positive, select_negative

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Nb101Dataset, Nb101DatasetAug
from model import NeuralPredictor
from utils import AverageMeter, AverageMeterGroup, get_logger, reset_seed, to_cuda

from scipy.stats import kendalltau


def accuracy_mse(predict, target, scale=100.):
    predict = Nb101Dataset.denormalize(predict.detach()) * scale
    target = Nb101Dataset.denormalize(target) * scale
    return F.mse_loss(predict, target)


def visualize_scatterplot(predict, target, scale=100.):
    def _scatter(x, y, subplot, threshold=None):
        plt.subplot(subplot)
        plt.grid(linestyle="--")
        plt.xlabel("Validation Accuracy")
        plt.ylabel("Prediction")
        plt.scatter(target, predict, s=1)
        if threshold:
            ax = plt.gca()
            ax.set_xlim(threshold, 95)
            ax.set_ylim(threshold, 95)

    predict = Nb101Dataset.denormalize(predict) * scale
    target = Nb101Dataset.denormalize(target) * scale
    plt.figure(figsize=(12, 6))
    _scatter(predict, target, 121)
    _scatter(predict, target, 122, threshold=90)
    plt.savefig("assets/scatterplot.png", bbox_inches="tight")
    plt.close()


def main():
    valid_splits = ["172", "334", "424", "860", "91-172", "91-334", "91-860", "denoise-91", "denoise-80", "all"]
    parser = ArgumentParser()
    parser.add_argument("--train_split", choices=valid_splits, default="424")
    parser.add_argument("--eval_split", choices=valid_splits, default="all")
    parser.add_argument("--gcn_hidden", type=int, default=144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", default=100, type=int)
    parser.add_argument("--eval_batch_size", default=10000, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", "--learning_rate", default=5e-4, type=float)
    parser.add_argument("--wd", "--weight_decay", default=1e-3, type=float)
    parser.add_argument("--train_print_freq", default=None, type=int)
    parser.add_argument("--eval_print_freq", default=10, type=int)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--arch_aug", default=True, type=bool)
    parser.add_argument("--triploss", default=False, type=bool)
    parser.add_argument("--visualize", default=False, action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    reset_seed(args.seed)

    if args.arch_aug:
        dataset = Nb101DatasetAug(split=args.train_split)
    else:
        dataset = Nb101Dataset(split=args.train_split)
    dataset_test = Nb101Dataset(split=args.eval_split, arch_aug=args.arch_aug)
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=False)
    test_data_loader = DataLoader(dataset_test, batch_size=args.eval_batch_size)
    net = NeuralPredictor(gcn_hidden=args.gcn_hidden)
    net.cuda()
    if args.triploss:
        criterion = MSELoss_and_TripletLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    logger = get_logger()

    net.train()
    for epoch in range(args.epochs):
        meters = AverageMeterGroup()
        lr = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(data_loader):
            batch = to_cuda(batch)
            target = batch["val_acc"].to(torch.float)
            predict = net(batch)
            optimizer.zero_grad()
            if args.triploss:
                positive_batch = to_cuda(select_positive(batch, dataset))
                negative_batch = to_cuda(select_negative(batch, dataset))
                positive = net(positive_batch)
                negative = net(negative_batch)
                loss = criterion(predict, target, positive, negative)
            else:
                loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            mse = accuracy_mse(predict, target)
            meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))
            if (args.train_print_freq and step % args.train_print_freq == 0) or \
                    step + 1 == len(data_loader):
                logger.info("Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s",
                            epoch + 1, args.epochs, step + 1, len(data_loader), lr, meters)
        lr_scheduler.step()

    net.eval()
    meters = AverageMeterGroup()
    predict_, target_ = [], []
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch = to_cuda(batch)
            target = batch["val_acc"]
            predict = net(batch)
            predict_.append(predict.cpu().numpy())
            target_.append(target.cpu().numpy())
            meters.update({"mse": accuracy_mse(predict, target).item()}, n=target.size(0))

            if (args.eval_print_freq and step % args.eval_print_freq == 0) or \
                    step % 10 == 0 or step + 1 == len(test_data_loader):
                logger.info("Evaluation Step [%d/%d]  %s", step + 1, len(test_data_loader), meters)
    predict_ = np.concatenate(predict_)
    target_ = np.concatenate(target_)
    predict_ = torch.from_numpy(Nb101Dataset.denormalize(predict_))
    target_ = torch.from_numpy(Nb101Dataset.denormalize(target_))
    # random choose 5000 test
    random.shuffle(predict_, random.seed(0))
    random.shuffle(target_, random.seed(0))
    KTau_list = []
    MSE_list = []
    for _ in range(10):
        random_int = random.randint(0, 410000)
        KTau_list.append(kendalltau(predict_[random_int:random_int+5000], target_[random_int:random_int+5000])[0])
        MSE_list.append(F.mse_loss(predict_[random_int:random_int+5000], target_[random_int:random_int+5000]).item())
    logger.info("Kendalltau: mean: %.6f, std: %.6f", np.mean(KTau_list), np.std(KTau_list))
    logger.info("MSE: mean: %.6f, std: %.6f", np.mean(MSE_list), np.std(MSE_list))
    # logger.info("mean predict: %.6f, mean target: %.6f", np.mean(Nb101Dataset.denormalize(predict_)),
    #             np.mean(Nb101Dataset.denormalize(target_)))
    if args.visualize:
        visualize_scatterplot(predict_, target_)


if __name__ == "__main__":
    main()
