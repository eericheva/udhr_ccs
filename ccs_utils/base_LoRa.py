import gc
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig
from accelerate.utils import load_and_quantize_model
from natsort import natsorted

from ccs_utils.base_utils import (
    EarlyStopper,
    EarlyStopperAcc,
    get_tokeniser_batch,
    NeedToRestartLORA,
)
from setups import add_name, model_name, wandb


class LoRa(nn.Module):
    def __init__(
        self, inner_d, lora_rank=2, device="cuda", PATH_TO_DICT="", load_ep=None
    ):
        super().__init__()
        self.inner_d = inner_d
        self.lora_rank = lora_rank
        self.lora_alpha = 1
        self.PATH_TO_DICT = PATH_TO_DICT

        self.lora_a = nn.Linear(self.inner_d, self.lora_rank, bias=False)
        self.lora_b = nn.Linear(self.lora_rank, self.inner_d, bias=False)
        torch.nn.init.xavier_uniform_(self.lora_a.weight, gain=1.0 / (2 * self.inner_d))
        torch.nn.init.zeros_(self.lora_b.weight)
        self.to(device)
        if load_ep:
            self.load_lora(load_ep)
            self.eval()
            self.half()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x_in):
        lora_type = self.lora_a.weight.dtype
        x_in_type = x_in.dtype
        x_in = self.lora_a(x_in.to(dtype=lora_type))
        x_in = (self.lora_alpha / self.lora_rank) * self.lora_b(x_in)
        x_in = x_in.to(dtype=x_in_type)
        return x_in

    def save_lora(self, ep):
        torch.save(self, f"{self.PATH_TO_DICT}_{ep}.chkp")

    def load_lora(self, ep):
        if "last" in ep:
            chkp_list = natsorted(glob.iglob(f"{self.PATH_TO_DICT}_*.chkp"))
            ep_chkp_list = [
                int(l.split(f"{self.PATH_TO_DICT}_")[-1].split(".chkp")[0])
                for l in chkp_list
            ]
            ep = np.max(ep_chkp_list)
        self.load_state_dict(torch.load(f"{self.PATH_TO_DICT}_{ep}.chkp").state_dict())


class QLoRa(nn.Module):
    def __init__(
        self, inner_d, lora_rank=2, device="cuda", PATH_TO_DICT="", load_ep=None
    ):
        super().__init__()
        self.inner_d = inner_d
        self.lora_rank = lora_rank
        self.lora_alpha = 1
        self.PATH_TO_DICT = PATH_TO_DICT

        self.lora_a = nn.Linear(self.inner_d, self.lora_rank, bias=False)
        self.lora_b = nn.Linear(self.lora_rank, self.inner_d, bias=False)
        weights_location = self.chkp_path(ep=load_ep)
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True, skip_modules=[], bnb_4bit_compute_dtype=torch.bfloat16
        )

        with init_empty_weights():
            self.lora_a = nn.Linear(self.inner_d, self.lora_rank, bias=False)
            self.lora_b = nn.Linear(self.lora_rank, self.inner_d, bias=False)
        self = load_and_quantize_model(
            self,
            weights_location=weights_location,
            bnb_quantization_config=bnb_quantization_config,
            device_map="auto",
        )

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x_in):
        x_in = self.lora_a(x_in)
        x_in = (self.lora_alpha / self.lora_rank) * self.lora_b(x_in)
        return x_in

    def save_lora(self, ep, state_dict=False):
        if state_dict:
            torch.save(self.state_dict(), f"{self.PATH_TO_DICT}_sd_{ep}.chkp")
        else:
            torch.save(self, f"{self.PATH_TO_DICT}_{ep}.chkp")

    def chkp_path(self, ep):
        # self.load_state_dict(torch.load(f"{self.PATH_TO_DICT}_ep{ep}.chkp"))
        if "last" in ep:
            chkp_list = natsorted(glob.iglob(f"{self.PATH_TO_DICT}_*.chkp"))
            ep_chkp_list = []
            for l in chkp_list:
                if f"{self.PATH_TO_DICT}_sd_" in l:
                    ep_chkp_list.append(
                        int(l.split(f"{self.PATH_TO_DICT}_sd_")[-1].split(".chkp")[0])
                    )
                else:
                    ep_chkp_list.append(
                        int(l.split(f"{self.PATH_TO_DICT}_")[-1].split(".chkp")[0])
                    )
            ep = np.max(ep_chkp_list)
        chkp_path = f"{self.PATH_TO_DICT}_sd_{ep}.chkp"
        if not os.path.isfile(chkp_path):
            self.load_state_dict(
                torch.load(f"{self.PATH_TO_DICT}_{ep}.chkp").state_dict()
            )
            torch.save(self.state_dict(), chkp_path)
        return chkp_path


def get_hook_for(mod, inner_d, lora_rank=2, PATH_TO_DICT="", load_ep=None):
    if load_ep is not None and "sd" in load_ep:
        mod.lora = QLoRa(
            inner_d, lora_rank=lora_rank, PATH_TO_DICT=PATH_TO_DICT, load_ep=load_ep
        )
    else:
        mod.lora = LoRa(
            inner_d, lora_rank=lora_rank, PATH_TO_DICT=PATH_TO_DICT, load_ep=load_ep
        )

    def hook_fn(self, input, output):
        # mod and self are the same here, so you can use any of them
        return (output[0] + self.lora(input[0]), output[1])

    return hook_fn


class LoRa_model(object):
    def __init__(
        self,
        model,
        lora_rank=2,
        lora_layer=-1,
        nepochs=10,
        lr=1e-3,
        batch_size=-1,
        PATH_TO_DICT="",
        load_ep=None,
    ):
        device = "cuda"

        # data
        self.d = model.model.layers[lora_layer - 1].hidden_size

        # training
        self.nepochs = nepochs
        self.lr = lr
        self.device = device
        self.batch_size = batch_size

        # lora
        self.model = model
        self.load_ep = load_ep
        self.lora_rank = lora_rank
        # set model to test regime
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # initialize LORA
        self.PATH_TO_DICT = os.path.join(PATH_TO_DICT, f"lora_model_{lora_layer - 1}")
        self.lora_layer = self.model.model.layers[lora_layer - 1]
        self.register_lora()

    def register_lora(self):
        # register LORA
        self.lora_layer.register_forward_hook(
            get_hook_for(
                self.lora_layer,
                inner_d=self.d,
                lora_rank=self.lora_rank,
                PATH_TO_DICT=self.PATH_TO_DICT,
                load_ep=self.load_ep,
            )
        )
        self.first_prediction = None
        self.last_prediction = None

    def forward_lora_model(self, x_in):
        x_in = (
            x_in.clone().detach().type(torch.int).to(self.device).requires_grad_(False)
        )
        p_logits = self.model(x_in, output_hidden_states=True)
        p_logits = p_logits.logits[:, -3]  # -1 = next, # -2 = eos, # -3 = 0/1
        # get the appropriate hidden states
        # hs_tuple = p_out["hidden_states"]
        # layer = -1
        # hs = hs_tuple[layer][:, -1]  # .detach().cpu().numpy()
        p_out = torch.max(nn.Softmax(dim=1)(p_logits), dim=1)[0]  # .unsqueeze(dim=1)
        return p_out, p_logits

    def get_y_data(self, y1):
        # {"0": 235276,
        # "1": 235274}
        tokenizer = self.model.tokenizer
        y0_logits = torch.zeros(
            (len(y1), tokenizer.vocab_size), dtype=torch.int, device=self.device
        )
        y1_logits = torch.zeros(
            (len(y1), tokenizer.vocab_size), dtype=torch.int, device=self.device
        )
        y1_ids, y0_ids = torch.zeros((len(y1))), torch.zeros((len(y1)))
        for i, _y in enumerate(y1):
            y1_logits[i, tokenizer(str(_y)).input_ids[1]] = 1
            y1_ids[i] = tokenizer(str(_y)).input_ids[1]
            y0_logits[i, tokenizer(str(1 - int(_y))).input_ids[1]] = 1
            y0_ids[i] = tokenizer(str(1 - int(_y))).input_ids[1]

        y0_logits = y0_logits.to(self.device).type(torch.cuda.LongTensor)
        y0_ids = y0_ids.to(self.device).type(torch.cuda.LongTensor)
        y1_logits = y1_logits.to(self.device).type(torch.cuda.LongTensor)
        y1_ids = y1_ids.to(self.device).type(torch.cuda.LongTensor)

        return y0_ids, y0_logits, y1_ids, y1_logits

    def get_loss_proba(self, p0, p1, y0_probs, y1_probs):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        p0 = nn.Softmax(dim=1)(p0)
        p1 = nn.Softmax(dim=1)(p1)
        p0 = torch.max(p0 * y0_probs, dim=1)[0]
        p1 = torch.max(p1 * y1_probs, dim=1)[0]

        informative_loss = (torch.min(p0, p1) ** 2).mean(0)
        consistent_loss = ((p0 - (1 - p1)) ** 2).mean(0)
        return informative_loss + consistent_loss

    def get_loss_spin(self, p0, p1, y0_probs, y1_probs):
        """Compute the SPIN loss for a batch of policy and reference model log probabilities.

        Args:
            y1_probs : policy_real_logps: Log probabilities of the policy model for the real responses. Shape: (
            batch_size,)
            p1 : policy_generated_logps: Log probabilities of the policy model for the generated responses. Shape: (
            batch_size,)
            y0_probs : opponent_real_logps: Log probabilities of the reference model for the real responses. Shape:
            (batch_size,)
            p0: opponent_generated_logps: Log probabilities of the reference model for the generated responses.
            Shape: (batch_size,)
            beta: Temperature parameter for the SPIN loss, typically something in the range of 0.1 to 0.5. We ignore
            the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model
            that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, real_rewards, generated_rewards).
            The losses tensor contains the SPIN loss for each example in the batch.
            The real_rewards and generated_rewards tensors contain the rewards for the real and generated responses,
            respectively.
        """
        loss_type = "hinge"  # 'sigmoid', 'hinge'
        reference_free = False
        # beta = 0.1
        beta = 1

        p0 = nn.Softmax(dim=1)(p0)
        p1 = nn.Softmax(dim=1)(p1)
        pi_logratios = y1_probs - p1
        ref_logratios = y0_probs - p0

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if loss_type == "sigmoid":
            losses = -F.logsigmoid(beta * logits)
        elif loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        real_rewards = beta * (y1_probs - y0_probs).detach()
        generated_rewards = beta * (p1 - p0).detach()

        return losses.mean()

    def get_acc(self, p0, p1, y0_probs, y1_probs, y1_labels):
        """
        Computes accuracy for the current parameters on the given test inputs
        # {"0": 235276,
        # "1": 235274}
        """
        p0 = nn.Softmax(dim=1)(p0)
        p1 = nn.Softmax(dim=1)(p1)
        p0 = torch.max(p0 * y0_probs, dim=1)[0]
        p1 = torch.max(p1 * y1_probs, dim=1)[0]

        avg_confidence = 0.5 * (p0 + (1 - p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(
            int
        )  # [:, 0]
        y1_labels = y1_labels.astype(int)
        acc = (predictions == y1_labels).mean()
        if self.load_ep is None:
            if acc < 1 - acc:
                acc = 1 - acc
                predictions = 1 - predictions
        else:
            if (acc < 1 - acc) and self.first_prediction is None:
                self.first_prediction = "inverse"
            if self.first_prediction == "inverse":
                acc = 1 - acc
                predictions = 1 - predictions

        return acc, predictions

    def save_lora(self, ep):
        self.lora_layer.lora.save_lora(ep)

    def load_lora(self, ep):
        self.lora_layer.lora.load_lora(ep)

    def train(
        self,
        x0_train,
        x1_train,
        y1_train,
        x0_test,
        x1_test,
        y1_test,
        runbd,
        restart_nums,
    ):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = x0_train, x1_train
        permutation = torch.randperm(len(x0))
        x0, x1, y1 = x0[permutation], x1[permutation], y1_train[permutation]

        x0_t, x1_t = x0_test, x1_test

        # set up optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        batch_size_tr = len(x0) if self.batch_size == -1 else self.batch_size
        batch_size_te = 40
        nbatches_tr = (
            (len(x0) // batch_size_tr)
            if (len(x0) / batch_size_tr) == (len(x0) // batch_size_tr)
            else (len(x0) // batch_size_tr + 1)
        )
        nbatches_te = (
            (len(x0_t) // batch_size_te)
            if (len(x0_t) / batch_size_te) == (len(x0_t) // batch_size_te)
            else (len(x0_t) // batch_size_te + 1)
        )

        # Start training (full batch)
        early_stopper_acc = EarlyStopperAcc(patience=2)
        early_stopper_loss = EarlyStopper(patience=2)
        need2restartLORA = NeedToRestartLORA(restart_nums=restart_nums)
        progress_bar_epochs = tqdm.tqdm(
            range(self.nepochs),
            total=self.nepochs,
            desc=f"Train CCS Loss: {np.inf}, ValLoss : {np.inf}",
        )
        for epoch in progress_bar_epochs:
            self.first_prediction = None
            gc.collect()
            torch.cuda.empty_cache()
            # ******************** #
            # TEST RUN
            # ******************** #
            validation_loss, validation_ccs_acc = [], []
            progress_bar_val = tqdm.tqdm(
                range(nbatches_te),
                total=nbatches_te,
                desc=f"ValLoss : {round(np.mean(validation_loss), 6)}, "
                + f"ValAcc: {round(np.mean(validation_ccs_acc), 6)}",
            )  # position=2 *
            for j in progress_bar_val:
                x0_batch = x0_t[j * batch_size_te : (j + 1) * batch_size_te]
                x1_batch = x1_t[j * batch_size_te : (j + 1) * batch_size_te]
                y1_labels = y1_test[j * batch_size_te : (j + 1) * batch_size_te]
                _, y0_batch_logits, _, y1_batch_logits = self.get_y_data(y1_labels)
                with torch.no_grad():
                    (_, p0_test_logits), (_, p1_test_logits) = self.forward_lora_model(
                        x0_batch
                    ), self.forward_lora_model(x1_batch)
                ### ********** loss ***********
                if "proba_loss" in add_name:
                    loss = self.get_loss_proba(
                        p0_test_logits, p1_test_logits, y0_batch_logits, y1_batch_logits
                    )
                elif "spin_loss" in add_name:
                    loss = self.get_loss_spin(
                        p0_test_logits, p1_test_logits, y0_batch_logits, y1_batch_logits
                    )
                ### ********** loss ***********
                ccs_acc, _ = self.get_acc(
                    p0_test_logits,
                    p1_test_logits,
                    y0_batch_logits,
                    y1_batch_logits,
                    y1_labels,
                )
                loss = loss.detach().cpu().numpy()
                validation_loss.append(loss)
                validation_ccs_acc.append(ccs_acc)
                gc.collect()
                torch.cuda.empty_cache()
                progress_bar_val.set_description(
                    desc=f"ValLoss : {round(np.mean(validation_loss), 6)}, "
                    + f"ValAcc: {round(np.mean(validation_ccs_acc), 6)}",
                    refresh=False,
                )
                # ******************** #
                # BATCH END
                # ******************** #
            validation_loss, validation_ccs_acc = np.mean(validation_loss), np.mean(
                validation_ccs_acc
            )
            if early_stopper_acc.early_stop(validation_ccs_acc):
                break
            if early_stopper_loss.early_stop(validation_loss):
                break
            if need2restartLORA.early_stop(validation_loss):
                return [], []

            del x0_batch, x1_batch, y1_labels, y0_batch_logits, y1_batch_logits
            del p0_test_logits, p1_test_logits, loss, ccs_acc
            # del validation_loss, validation_ccs_acc
            gc.collect()
            torch.cuda.empty_cache()
            # ******************** #
            # TRAIN RUN
            # ******************** #
            train_loss, train_ccs_acc = [], []
            progress_bar_train = tqdm.tqdm(
                range(nbatches_tr),
                total=nbatches_tr,
                desc=f"TrainLoss: {round(np.mean(train_loss), 6)}, "
                + f"TrainAcc: {round(np.mean(train_ccs_acc), 6)}",
            )
            for j in progress_bar_train:
                x0_batch = x0[j * batch_size_tr : (j + 1) * batch_size_tr]
                x1_batch = x1[j * batch_size_tr : (j + 1) * batch_size_tr]
                y1_labels = y1[j * batch_size_tr : (j + 1) * batch_size_tr]
                _, y0_batch_logits, _, y1_batch_logits = self.get_y_data(y1_labels)

                # probe
                with torch.enable_grad():
                    (_, p0_logits), (_, p1_logits) = self.forward_lora_model(
                        x0_batch
                    ), self.forward_lora_model(x1_batch)
                    ### ********** loss ***********
                    if "proba_loss" in add_name:
                        loss = self.get_loss_proba(
                            p0_logits, p1_logits, y0_batch_logits, y1_batch_logits
                        )
                    elif "spin_loss" in add_name:
                        loss = self.get_loss_spin(
                            p0_logits, p1_logits, y0_batch_logits, y1_batch_logits
                        )
                    ### ********** loss ***********
                    ccs_acc, _ = self.get_acc(
                        p0_logits,
                        p1_logits,
                        y0_batch_logits,
                        y1_batch_logits,
                        y1_labels,
                    )
                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.detach().cpu().numpy()
                train_loss.append(loss)
                train_ccs_acc.append(ccs_acc)
                gc.collect()
                torch.cuda.empty_cache()
                progress_bar_train.set_description(
                    desc=f"TrainLoss: {round(np.mean(train_loss), 6)}, "
                    + f"TrainAcc: {round(np.mean(train_ccs_acc), 6)}",
                    refresh=False,
                )
                # ******************** #
                # BATCH END
                # ******************** #
            # ******************** #
            # LOGGING TO WANDB
            # ******************** #
            train_loss, train_ccs_acc = np.mean(train_loss), np.mean(train_ccs_acc)
            progress_bar_epochs.set_description(
                desc=f"Train CCS, Loss: {round(train_loss, 6)}, "
                + f"ValLoss: {round(validation_loss, 6)}, "
                + f"Acc: {round(train_ccs_acc, 6)}, "
                + f"ValAcc: {round(validation_ccs_acc, 6)}",
                refresh=False,
            )
            runbd.log(
                data={
                    "Loss": train_loss,
                    "ValLoss": validation_loss,
                    "Acc": train_ccs_acc,
                    "ValAcc": validation_ccs_acc,
                },
                step=epoch,
            )
            self.save_lora(ep=epoch)
            del x0_batch, x1_batch, y1_labels, y0_batch_logits, y1_batch_logits
            del p0_logits, p1_logits, loss, ccs_acc
            # del train_loss, train_ccs_acc
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
            # ******************** #
            # EPOCH END
            # ******************** #
        # runbd.log({"p_neg": p0, "p_pos": p1, "p_test_neg": p0_test, "p_test_pos": p1_test})
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(30)

        # ******************** #
        # EVAL RUN
        # ******************** #

        acc, y_mod = [], []
        progress_bar_val = tqdm.tqdm(
            range(nbatches_te),
            total=nbatches_te,
            desc=f"EVAL RUN" + f"ValAcc: {round(np.mean(acc), 6)}",
        )
        for j in progress_bar_val:
            x0_batch = x0_t[j * batch_size_te : (j + 1) * batch_size_te]
            x1_batch = x1_t[j * batch_size_te : (j + 1) * batch_size_te]
            y1_labels = y1_test[j * batch_size_te : (j + 1) * batch_size_te]
            _, y0_batch_logits, _, y1_batch_logits = self.get_y_data(y1_labels)
            with torch.no_grad():
                (_, p0_test_logits), (_, p1_test_logits) = self.forward_lora_model(
                    x0_batch
                ), self.forward_lora_model(x1_batch)

            ccs_acc, ccs_y = self.get_acc(
                p0_test_logits,
                p1_test_logits,
                y0_batch_logits,
                y1_batch_logits,
                y1_labels,
            )
            acc.append(ccs_acc)
            y_mod.append(ccs_y)
            gc.collect()
            torch.cuda.empty_cache()
            progress_bar_val.set_description(
                desc=f"EVAL RUN" + f"ValAcc: {round(np.mean(acc), 6)}", refresh=False
            )
        acc = np.asarray(acc)
        y_mod = np.concatenate(y_mod)
        return acc, y_mod

    def repeated_train(self, x0, x1, y1, x0_test, x1_test, y1_test, layer):
        runbd = wandb.init(
            project="UDHR_LORA",
            name=f"{model_name}{add_name}_l{layer}",
            id=f"{model_name}{add_name}_l{layer}_1",
            group=f"UDHR_LORA{model_name}{add_name}",
            config={
                "post_name": model_name,
                "add_name": add_name,
                "group_name": f"{model_name}{add_name}",
            },
            reinit=True,
        )
        time.sleep(5)
        acc, y_mod = [], []
        restart_nums = 3
        while len(acc) < 1:
            self.register_lora()
            acc, y_mod = self.train(
                x0, x1, y1, x0_test, x1_test, y1_test, runbd, restart_nums
            )
            restart_nums -= 1
        runbd.finish()
        return acc, y_mod

    def eval(self, x0_t, x1_t, y1_test):
        batch_size_te = self.batch_size
        nbatches_te = (
            (len(x0_t) // batch_size_te)
            if (len(x0_t) / batch_size_te) == (len(x0_t) // batch_size_te)
            else (len(x0_t) // batch_size_te + 1)
        )

        gc.collect()
        torch.cuda.empty_cache()
        # ******************** #
        # EVAL RUN
        # ******************** #
        acc, y_mod = [], []
        progress_bar_val = tqdm.tqdm(
            range(nbatches_te),
            total=nbatches_te,
            desc=f"EVAL RUN, " + f"ValAcc: {round(np.mean(acc), 6)}",
        )
        for j in progress_bar_val:
            x0_batch = x0_t[j * batch_size_te : (j + 1) * batch_size_te]
            x1_batch = x1_t[j * batch_size_te : (j + 1) * batch_size_te]
            y1_labels = y1_test[j * batch_size_te : (j + 1) * batch_size_te]
            _, y0_batch_logits, _, y1_batch_logits = self.get_y_data(y1_labels)
            with torch.no_grad():
                (_, p0_test_logits), (_, p1_test_logits) = self.forward_lora_model(
                    x0_batch
                ), self.forward_lora_model(x1_batch)

            ccs_acc, ccs_y = self.get_acc(
                p0_test_logits,
                p1_test_logits,
                y0_batch_logits,
                y1_batch_logits,
                y1_labels,
            )
            acc.append(ccs_acc)
            y_mod.append(ccs_y)
            gc.collect()
            torch.cuda.empty_cache()
            progress_bar_val.set_description(
                desc=f"EVAL RUN, " + f"ValAcc: {round(np.mean(acc), 6)}", refresh=False
            )
        acc = np.asarray(acc)
        y_mod = np.concatenate(y_mod)
        self.last_prediction = self.first_prediction
        return acc, y_mod


def check_lora(
    model, tokenizer, data_train, data_test, layer, batch_size, PATH_TO_DICT
):
    neg_hs_train, pos_hs_train, y_train = get_tokeniser_batch(
        model, tokenizer, data_train
    )
    neg_hs_test, pos_hs_test, y_test = get_tokeniser_batch(model, tokenizer, data_test)

    # Train CCS without any labels
    ccs = LoRa_model(
        model=model,
        lora_rank=2,
        lora_layer=layer,
        batch_size=batch_size,
        PATH_TO_DICT=PATH_TO_DICT,
        load_ep=None,
    )
    ccs_acc, y_mod = ccs.repeated_train(
        neg_hs_train, pos_hs_train, y_train, neg_hs_test, pos_hs_test, y_test, layer
    )

    print("CCS Layer {} accuracy: {}".format(layer, ccs_acc.mean()))
    return y_mod


def eval_lora(model, tokenizer, data_test, layer, batch_size, PATH_TO_DICT):
    neg_hs_test, pos_hs_test, y_test = get_tokeniser_batch(model, tokenizer, data_test)

    # Train CCS without any labels
    ccs = LoRa_model(
        model=model,
        lora_rank=2,
        lora_layer=layer,
        batch_size=batch_size,
        PATH_TO_DICT=PATH_TO_DICT,
        load_ep="sd_last",
    )
    ccs_acc, y_mod = ccs.eval(neg_hs_test, pos_hs_test, y_test)

    print("CCS Layer {} accuracy: {}".format(layer, ccs_acc.mean()))
    # y_mod = [1 - int(y_m) if y_t == 0 else y_m for y_m, y_t in zip(y_mod, y_test)]
    return y_mod


def make_lora(model, tokenizer, data_test, layer, batch_size, PATH_TO_DICT):
    # neg_hs_test, pos_hs_test, y_test = get_tokeniser_batch(model, tokenizer, data_test)

    # Train CCS without any labels
    ccs = LoRa_model(
        model=model,
        lora_rank=2,
        lora_layer=layer,
        batch_size=batch_size,
        PATH_TO_DICT=PATH_TO_DICT,
        load_ep="sd_last",
    )
    # ccs_acc, y_mod = ccs.eval(neg_hs_test, pos_hs_test, y_test)

    # print("CCS Layer {} accuracy: {}".format(layer, ccs_acc.mean()))
    # y_mod = [1 - int(y_m) if y_t == 0 else y_m for y_m, y_t in zip(y_mod, y_test)]
    # return y_mod
