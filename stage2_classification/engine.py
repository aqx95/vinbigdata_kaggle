import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import datetime
import time
import logging
import sklearn
from sklearn.metrics import roc_auc_score

from loss import loss_fn
from commons import Meter


class Fitter:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logging.getLogger('training')

        self.best_auc = 0
        self.epoch = 0
        self.best_loss = np.inf
        self.oof = None
        self.monitored_metrics = None

        if not os.path.exists(self.config.SAVE_PATH):
            os.makedirs(self.config.SAVE_PATH)
        if not os.path.exists(self.config.LOG_PATH):
            os.makedirs(self.config.LOG_PATH)

        self.loss = loss_fn(config.criterion, config).to(self.device)
        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(),
                                **config.optimizer_params[config.optimizer])
        self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(optimizer=self.optimizer,
                                **config.scheduler_params[config.scheduler])


    def fit(self, train_loader, valid_loader, fold):
        self.logger.info("Starts Training with {} on Device: {}".format(self.config.model_name, self.device))

        for epoch in range(self.config.num_epochs):
            self.logger.info("LR: {}".format(self.optimizer.param_groups[0]['lr']))
            train_loss = self.train_one_epoch(train_loader)
            self.logger.info("[RESULTS] Train Epoch: {} | Train Loss: {}".format(self.epoch, train_loss))
            valid_loss, auc_roc, val_pred = self.validate_one_epoch(valid_loader)
            self.logger.info("[RESULTS] Validation Epoch: {} | Valid Loss: {} | AUC: {:.3f}".format(self.epoch, valid_loss, auc_roc))

            self.monitored_metrics = auc_roc
            self.oof = val_pred

            if self.best_loss > valid_loss:
                self.best_loss = valid_loss
            if self.best_auc < auc_roc:
                self.logger.info(f"Epoch {self.epoch}: Saving model... | AUC improvement {self.best_auc} -> {auc_roc}")
                self.save(os.path.join(self.config.SAVE_PATH, '{}_fold{}.pt').format(self.config.model_name, fold))
                self.best_auc = auc_roc

            #Update Scheduler
            if self.config.val_step_scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.monitored_metrics)
                else:
                    self.scheduler.step()

            self.epoch += 1

        fold_checkpoint = self.load(os.path.join(self.config.SAVE_PATH, '{}_fold{}.pt').format(self.config.model_name, fold))
        return fold_checkpoint


    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = Meter()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (img, labels) in pbar:
            img, labels = img.to(self.device), labels.to(self.device)
            batch_size = labels.shape[0]

            logits = self.model(img)
            loss = self.loss(logits, labels.unsqueeze(1).float())
            summary_loss.update(loss.item(), batch_size)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.config.train_step_scheduler:
                self.scheduler.step(self.epoch+step/len(train_loader))

            description = f"Train Steps: {step}/{len(train_loader)} Summary Loss: {summary_loss.avg:.3f}"
            pbar.set_description(description)

        return summary_loss.avg


    def validate_one_epoch(self, valid_loader):
        self.model.eval()
        full_label = []
        full_pred = []
        summary_loss = Meter()
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        val_prediction = []

        with torch.no_grad():
            for step, (img, labels) in pbar:
                img, labels = img.to(self.device), labels.to(self.device)
                batch_size = labels.shape[0]

                logits = self.model(img)
                loss = self.loss(logits, labels.unsqueeze(1).float())
                summary_loss.update(loss.item(), batch_size)
                output = torch.sigmoid(logits)
                full_pred.append(output.cpu().detach())
                full_label.append(labels.cpu().detach())

                description = f"Valid Steps: {step}/{len(valid_loader)} Summary Loss: {summary_loss.avg:.3f}"
                pbar.set_description(description)

        full_pred = np.concatenate(full_pred, axis=0)
        full_label = np.concatenate(full_label, axis=0)
        auc_roc = sklearn.metrics.roc_auc_score(full_label, full_pred)
        return summary_loss.avg, auc_roc, full_pred


    def save(self, path):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_auc": self.best_auc,
                "epoch": self.epoch,
                "oof_pred": self.oof
            }, path
        )


    def load(self, path):
        checkpoint = torch.load(path)
        return checkpoint
