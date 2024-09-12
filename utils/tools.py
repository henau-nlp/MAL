import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience：自上次模型在验证集上损失降低之后等待的epoch
            verbose：当为False时，运行的时候将不显示详细信息
            counter：计数器，当其值超过patience时候，使用early stopping
            best_score：记录模型评估的最好分数
            early_step：决定模型要不要early stop，为True则停
            val_loss_min：模型评估损失函数的最小值，默认为正无穷(np.Inf)
            save_path：保存路径，我们使用该函数时不用保存模型
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, entity_f1_score, model, model_path_prefix, strategy, choose_fraction):

        score = entity_f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(entity_f1_score, model, model_path_prefix, strategy, choose_fraction)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(entity_f1_score, model, model_path_prefix, strategy, choose_fraction)
            self.counter = 0

    def save_checkpoint(self, entity_f1_score, model, model_path_prefix, strategy, choose_fraction):

        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation entity_f1_score increase ({self.val_loss_min:.6f} --> {entity_f1_score:.6f}).  Saving '
                  f'model ...')
        # 这里会存储迄今最优模型的参数
        torch.save(model.state_dict(),
                   model_path_prefix + strategy + "_" + str(choose_fraction) + '.pth')
        self.val_loss_min = entity_f1_score
