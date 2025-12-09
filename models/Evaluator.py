from typing import Tuple
import copy
import torch
import torchmetrics
import pytorch_lightning as pl

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
import numpy as np
import quadprog

class Evaluator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.datatype == 'imaging' or self.hparams.datatype == 'multimodal':
            self.model = ImagingModel(self.hparams)
        if self.hparams.datatype == 'charms':
            self.model = ImageModelPetFinderWithRTDL(self.hparams)
        if self.hparams.datatype == 'tabular':
            self.model = TabularModel(self.hparams)
        if self.hparams.datatype == 'imaging_and_tabular':
            self.model = MultimodalModel(self.hparams)
            # self.model = DAFT(self.hparams)

        task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_testi = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_testt = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_testi = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_testt = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

        self.acc_i = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_t = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.epoi = 0.5
        self.epot = 0.5

        self.criterion = torch.nn.CrossEntropyLoss()

        self.best_val_score = 0
        self.grad_dims = []
        for n, param in self.model.head.named_parameters():
            self.grad_dims.append(param.data.numel())

        grads = torch.Tensor(sum(self.grad_dims), 2)
        self.grads = grads.cuda()
        self.automatic_optimization = False
        # print(self.model1)

    def forward(self, x: torch.Tensor):
        """
    Generates a prediction from a data point
    """
        o_i, o_t = self.model(x)

        return o_i, o_t

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
    Runs test step
    """
        x, y = batch
        o = self.forward(x)
        o_i, o_t = o

        y_hati = self.model.head(o_i)
        y_hatt = self.model.head(o_t)
        y_hat = (y_hati + y_hatt) * 0.5
        if len(y_hati.shape) == 1:
            y_hati = torch.unsqueeze(y_hati, 0)
        if len(y_hatt.shape) == 1:
            y_hatt = torch.unsqueeze(y_hatt, 0)
        if len(y_hat.shape) == 1:
            y_hat = torch.unsqueeze(y_hat, 0)

        y_hati = torch.softmax(y_hati.detach(), dim=1)
        y_hatt = torch.softmax(y_hatt.detach(), dim=1)
        y_hat = torch.softmax(y_hat.detach(), dim=1)
        # y_hat = (y_hati + y_hatt) * 0.5
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]
            y_hati = y_hati[:,1]
            y_hatt = y_hatt[:, 1]

        self.acc_testi(y_hati, y)
        self.auc_testi(y_hati, y)

        self.acc_testt(y_hatt, y)
        self.auc_testt(y_hatt, y)

        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        """
    Test epoch end
    """
        test_acc = self.acc_test.compute()
        test_auc = self.auc_test.compute()

        self.log('test.acc', test_acc)
        self.log('test.auc', test_auc)

        test_acci = self.acc_testi.compute()
        test_auci = self.auc_testi.compute()

        self.log('test.acci', test_acci)
        self.log('test.auci', test_auci)

        test_acct = self.acc_testt.compute()
        test_auct = self.auc_testt.compute()

        self.log('test.acct', test_acct)
        self.log('test.auct', test_auct)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        """
    Train and log.
    """
        opt = self.optimizers()

        x, y = batch
        o = self.forward(x)
        o_i, o_t = o

        ###############
        out_t = self.model.head(o_t)
        loss_t0 = self.selectsample(out_t, y, self.epot)
        self.manual_backward(loss_t0)
        self.grads[:, 0].fill_(0.0)
        cnt = 0
        for param in self.model.head.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                self.grads[beg:en, 0].copy_(param.grad.data.view(-1))
                cnt += 1
        opt.zero_grad()
        ###############

        y_hati = self.model.head(o_i)

        if len(y_hati.shape) == 1:
            y_hati = torch.unsqueeze(y_hati, 0)

        loss_i = self.criterion(y_hati, y)
        self.manual_backward(loss_i)

        ###########
        self.grads[:, 1].fill_(0.0)
        cnt = 0
        for param in self.model.head.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                self.grads[beg:en, 1].copy_(param.grad.data.view(-1))
                cnt += 1
        dotp = torch.mm(self.grads[:, 1].unsqueeze(0), self.grads[:, :1])
        if (dotp < 0).sum() != 0:
            newgrad = self.project2cone2(self.grads[:, 1].unsqueeze(1), self.grads[:, :1])
            # copy gradients back
            self.model = self.overwrite_grad(newgrad)
        ##############

        opt.step()
        opt.zero_grad()

        #####################
        o_a = self.model.forward_image(x[0])
        out_a = self.model.head(o_a)
        # loss_a = self.criterion(out_a, y)
        loss_a = self.selectsample(out_a, y, self.epoi)
        self.manual_backward(loss_a)
        self.grads[:, 0].fill_(0.0)
        cnt = 0
        for param in self.model.head.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                self.grads[beg:en, 0].copy_(param.grad.data.view(-1))
                cnt += 1
        opt.zero_grad()
        ######################

        o_t = self.model.forward_table(x[1])
        y_hatt = self.model.head(o_t)
        if len(y_hatt.shape) == 1:
            y_hatt = torch.unsqueeze(y_hatt, 0)
        loss_t = self.criterion(y_hatt, y)
        self.manual_backward(loss_t)

        ###########
        self.grads[:, 1].fill_(0.0)
        cnt = 0
        for param in self.model.head.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                self.grads[beg:en, 1].copy_(param.grad.data.view(-1))
                cnt += 1
        dotp = torch.mm(self.grads[:, 1].unsqueeze(0),self.grads[:, :1])
        if (dotp < 0).sum() != 0:
            newgrad = self.project2cone2(self.grads[:, 1].unsqueeze(1), self.grads[:, :1])
            # copy gradients back
            self.model = self.overwrite_grad(newgrad)
        ##############
        opt.step()
        opt.zero_grad()

        for n, p in self.model.named_parameters():
            if p.grad != None:
                del p.grad
        y_hat = (y_hati + y_hatt) * 0.5
        if len(y_hat.shape) == 1:
            y_hat = torch.unsqueeze(y_hat, 0)
        loss = self.criterion(y_hat, y).item()
        y_hat = torch.softmax(y_hat.detach(), dim=1)
        y_hati = torch.softmax(y_hati.detach(), dim=1)
        y_hatt = torch.softmax(y_hatt.detach(), dim=1)
        # y_hat = (y_hati + y_hatt) * 0.5
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]
            y_hati = y_hati[:, 1]
            y_hatt = y_hatt[:, 1]

        self.acc_train(y_hat, y)
        self.auc_train(y_hat, y)

        self.log('eval.train.loss_i', loss_i, on_epoch=True, on_step=False)
        self.log('eval.train.loss_t', loss_t, on_epoch=True, on_step=False)
        self.log('eval.train.loss', loss, on_epoch=True, on_step=False)


        # return loss_t

    def training_epoch_end(self, _) -> None:
        """
    Compute training epoch metrics and check for new best values
    """
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
        self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
    Validate and log
    """
        x, y = batch

        o = self.forward(x)
        o_i, o_t = o
        y_hati = self.model.head(o_i)
        y_hatt = self.model.head(o_t)

        y_hat = (y_hati + y_hatt) * 0.5
        if len(y_hati.shape) == 1:
            y_hati = torch.unsqueeze(y_hati, 0)
        if len(y_hatt.shape) == 1:
            y_hatt = torch.unsqueeze(y_hatt, 0)
        if len(y_hat.shape) == 1:
            y_hat = torch.unsqueeze(y_hat, 0)
        loss = self.criterion(y_hati, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        y_hati = torch.softmax(y_hati.detach(), dim=1)
        y_hatt = torch.softmax(y_hatt.detach(), dim=1)
        # y_hat = (y_hati + y_hatt) * 0.5
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]
            y_hati = y_hati[:, 1]
            y_hatt = y_hatt[:, 1]

        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)

        self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    def validation_epoch_end(self, _) -> None:
        """
    Compute validation epoch metrics and check for new best values
    """
        if self.trainer.sanity_checking:
            return

        epoch_acc_val = self.acc_val.compute()
        epoch_auc_val = self.auc_val.compute()

        self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
        self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)

        self.best_val_score = max(self.best_val_score, epoch_acc_val)
        self.acc_val.reset()
        self.auc_val.reset()

    def configure_optimizers(self):
        """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1) # 15 dvm
        optimizer_config = {
            "optimizer": optimizer,
        }
        print("optimizer_config:\n", optimizer_config)
        if scheduler:
            optimizer_config.update({
                "lr_scheduler": {
                    "name": 'MultiStep_LR_scheduler',
                    "scheduler": scheduler,
                }})
            print("scheduler_config:\n", scheduler.state_dict())
        return optimizer_config

    def overwrite_grad(self, newgrad):
        cnt = 0
        for param in self.model.head.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1
        return self.model

    def project2cone2(self, gradient, memories, margin=0.75, eps=1e-3):
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * (-1)
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np

        return torch.Tensor(x).view(-1, 1)

    def selectsample(self, out_t, y, epo=0.8):
        probs = torch.softmax(out_t, dim=1)
        log_probs = torch.log_softmax(out_t, dim=1)

        entropy = -torch.sum(probs * log_probs, dim=1)

        num_select = max(1, int(epo * out_t.size(0)))

        _, indices = torch.topk(entropy, k=num_select, largest=True)

        out_t_selected = out_t[indices]
        y_selected = y[indices]

        loss = self.criterion(out_t_selected, y_selected)
        return loss
