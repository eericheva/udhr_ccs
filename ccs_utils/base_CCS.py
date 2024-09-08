import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import LogisticRegression

from ccs_utils.base_utils import EarlyStopper, EarlyStopperAcc, get_hidden_states_batch
from setups import model_name, wandb


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)


class CCS(object):
    def __init__(
        self, inner_d, nepochs=100, lr=1e-3, batch_size=-1, device="cuda", linear=True
    ):
        # data
        self.var_normalize = False
        self.d = inner_d

        # training
        self.nepochs = nepochs
        self.lr = lr
        self.device = device
        self.batch_size = batch_size

        # probe
        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)

    def forward(self, x_in):
        x_in = torch.tensor(
            self.normalize(x_in),
            dtype=torch.float,
            requires_grad=False,
            device=self.device,
        )
        with torch.no_grad():
            p_out = self.best_probe(x_in)
        return p_out

    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

    def get_tensor_data(self, x0, x1):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        # x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        # x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        x0 = torch.tensor(
            self.normalize(x0),
            dtype=torch.float,
            requires_grad=False,
            device=self.device,
        )
        x1 = torch.tensor(
            self.normalize(x1),
            dtype=torch.float,
            requires_grad=False,
            device=self.device,
        )
        return x0, x1

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1) ** 2).mean(0)
        consistent_loss = ((p0 - (1 - p1)) ** 2).mean(0)
        return informative_loss + consistent_loss

    def get_acc(self, p0, p1, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        avg_confidence = 0.5 * (p0 + (1 - p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        # acc = max(acc, 1 - acc)
        if acc < 1 - acc:
            acc = 1 - acc
            predictions = 1 - predictions

        return acc, predictions

    def train(self, x0_train, x1_train, y_train, x0_test, x1_test, y_test, runbd):
        """
        Does a single training run of nepochs epochs
        """
        # x0, x1 = self.get_tensor_data(x0_train, x1_train)
        # x0_t, x1_t = self.get_tensor_data(x0_test, x1_test)
        x0, x1 = x0_train, x1_train
        x0_t, x1_t = x0_test, x1_test
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]

        # set up optimizer
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=self.lr, weight_decay=0.01
        )

        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        early_stopper_acc = EarlyStopperAcc(patience=3)
        early_stopper_loss = EarlyStopper(patience=3)
        progress_bar = tqdm.tqdm(
            range(self.nepochs),
            total=self.nepochs,
            desc=f"Train CCS Loss: {np.inf}, ValLoss : {np.inf}",
        )
        for epoch in progress_bar:
            p0, p1 = self.forward(x0), self.forward(x1)
            p0_test, p1_test = self.forward(x0_t), self.forward(x1_t)
            validation_loss = self.get_loss(p0_test, p1_test)
            ccs_acc, _ = self.get_acc(p0, p1, y_train)
            validation_ccs_acc, _ = self.get_acc(p0_test, p1_test, y_test)
            if early_stopper_acc.early_stop(validation_ccs_acc):
                break
            if early_stopper_loss.early_stop(validation_loss):
                break
            for j in range(nbatches):
                x0_batch = x0[j * batch_size : (j + 1) * batch_size]
                x1_batch = x1[j * batch_size : (j + 1) * batch_size]

                # probe
                with torch.enable_grad():
                    x0_batch, x1_batch = self.get_tensor_data(x0_batch, x1_batch)
                    p0, p1 = self.probe(x0_batch), self.probe(x1_batch)
                    loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            progress_bar.set_description(
                desc=f"Train CCS, Loss: {loss.item()}, ValLoss : {validation_loss.item()}",
                refresh=True,
            )
            runbd.log(
                data={
                    "Loss": loss.item(),
                    "ValLoss": validation_loss.item(),
                    "Acc": ccs_acc,
                    "ValAcc": validation_ccs_acc,
                },
                step=epoch,
            )
        runbd.log(
            {"p_neg": p0, "p_pos": p1, "p_test_neg": p0_test, "p_test_pos": p1_test}
        )
        time.sleep(30)

        return loss.detach().cpu().item()

    def repeated_train(self, x0, x1, y, x0_test, x1_test, y_test, layer):
        best_loss = np.inf
        runbd = wandb.init(
            project="UDHR_CCS",
            name=f"{model_name}_l{layer}",
            id=f"{model_name}_l{layer}_0",
            group=f"UDHR_CCS{model_name}",
            config={"post_name": model_name, "group_name": f"{model_name}"},
            reinit=True,
        )
        time.sleep(5)
        self.initialize_probe()
        loss = self.train(x0, x1, y, x0_test, x1_test, y_test, runbd)
        runbd.finish()
        if loss < best_loss:
            self.best_probe = copy.deepcopy(self.probe)
            best_loss = loss

        return best_loss


def check_ccs(model, tokenizer, data_train, data_test, layer, model_type):
    neg_hs_train, pos_hs_train, y_train = get_hidden_states_batch(
        model, tokenizer, data_train, layer, model_type
    )
    neg_hs_test, pos_hs_test, y_test = get_hidden_states_batch(
        model, tokenizer, data_test, layer, model_type
    )

    # Train CCS without any labels
    ccs = CCS(inner_d=neg_hs_train.shape[-1], batch_size=len(data_train[0]))
    ccs.repeated_train(
        neg_hs_train, pos_hs_train, y_train, neg_hs_test, pos_hs_test, y_test, layer
    )

    # Evaluate
    p0, p1 = ccs.forward(neg_hs_test), ccs.forward(pos_hs_test)
    ccs_acc, y_mod = ccs.get_acc(p0, p1, y_test)
    print("CCS Layer {} accuracy: {}".format(layer, ccs_acc))
    return y_mod


def check_lr(model, tokenizer, data_train, data_test, layer, model_type):
    neg_hs_train, pos_hs_train, y_train = get_hidden_states_batch(
        model, tokenizer, data_train, layer, model_type
    )
    neg_hs_test, pos_hs_test, y_test = get_hidden_states_batch(
        model, tokenizer, data_test, layer, model_type
    )

    # for simplicity we can just take the difference between positive and negative hidden states
    # (concatenating also works fine)
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test

    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    y_mod = lr.predict(x_test)
    print(
        "Logistic regression Layer {}  accuracy: {}".format(
            layer, lr.score(x_test, y_test)
        )
    )
    return y_mod
