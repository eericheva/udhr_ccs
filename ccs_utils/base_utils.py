import numpy as np
import torch
import tqdm


class EarlyStopper:
    def __init__(self, patience=3):
        self.patience = patience
        self.min_delta = 0.00001
        self.min_delta_exp = 0.000001
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        print("************************ EarlyStopper ************************")
        print(
            f"validation_loss: {validation_loss}, self.min_validation_loss: {self.min_validation_loss}"
        )
        print(
            f"validation_loss < self.min_validation_loss: {validation_loss < self.min_validation_loss}"
        )
        print(
            f"abs(self.min_validation_loss - validation_loss): {abs(self.min_validation_loss - validation_loss)} < "
            f"self.min_delta : {self.min_delta}"
        )
        if abs(self.min_validation_loss - validation_loss) < self.min_delta_exp:
            if self.counter >= 1:
                print("************************")
                print(f"EarlyStopper {validation_loss}")
                print("************************")
                return True
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 1
        elif abs(self.min_validation_loss - validation_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("************************")
                print(f"EarlyStopper {validation_loss}")
                print("************************")
                return True
        return False


class EarlyStopperAcc:
    def __init__(self, patience=3):
        self.patience = patience
        self.min_delta = 0.00001
        self.counter = 0
        self.min_validation_loss = -float("inf")

    def early_stop(self, validation_loss):
        print("************************ EarlyStopperAcc ************************")
        print(
            f"validation_loss: {validation_loss}, self.min_validation_loss: {self.min_validation_loss}"
        )
        print(
            f"validation_loss > self.min_validation_loss: {validation_loss > self.min_validation_loss}"
        )
        print(
            f"abs(self.min_validation_loss - validation_loss): {abs(self.min_validation_loss - validation_loss)} < "
            f"self.min_delta : {self.min_delta}"
        )
        if validation_loss >= 1.0:
            if self.counter >= 1:
                print("************************")
                print(f"EarlyStopperAcc {validation_loss}")
                print("************************")
                return True
        if validation_loss > self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 1
        elif abs(self.min_validation_loss - validation_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("************************")
                print(f"EarlyStopperAcc {validation_loss}")
                print("************************")
                return True
        return False


class NeedToRestartLORA:
    def __init__(self, restart_nums=3):
        self.restart_nums = restart_nums
        self.target_validation_loss = 0.5
        self.prev = 0

    def early_stop(self, validation_loss):
        print("************************ NeedToRestartLORA ************************")
        if validation_loss == self.prev and self.restart_nums > 0:
            print("************************")
            print(
                f"NeedToRestartLORA validation_loss : {validation_loss} == self.prev : {self.prev}"
            )
            print("************************")
            return True
        if validation_loss != self.target_validation_loss:
            self.prev = 0
        else:
            self.prev = validation_loss
        return False


def get_encoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given an encoder model and some text, gets the encoder hidden states (in a given layer, by default the last)
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(
        input_text, return_tensors="pt", padding=True
    ).input_ids.to(model.device)

    # forward_probe pass
    with torch.no_grad():
        output = model(encoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["hidden_states"]

    hs = hs_tuple[layer][:, -1].detach().cpu().numpy()

    return hs


def get_encoder_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given an encoder-decoder model and some text, gets the encoder hidden states (in a given layer, by default the
    last)
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
        model.device
    )
    decoder_text_ids = tokenizer("", return_tensors="pt").input_ids.to(model.device)

    # forward_probe pass
    with torch.no_grad():
        output = model(
            encoder_text_ids,
            decoder_input_ids=decoder_text_ids,
            output_hidden_states=True,
        )

    # get the appropriate hidden states
    hs_tuple = output["encoder_hidden_states"]
    hs = hs_tuple[layer][:, -1].detach().cpu().numpy()

    return hs


def get_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given a decoder model and some text, gets the hidden states (in a given layer, by default the last) on that input
    text

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize (adding the EOS token this time)
    if isinstance(input_text, str):
        input_text = input_text + tokenizer.eos_token
    else:
        input_text = [t + tokenizer.eos_token for t in input_text]
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(
        model.device
    )

    # forward_probe pass
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)

    # get the last layer, last token hidden states
    hs_tuple = output["hidden_states"]
    hs = hs_tuple[layer][:, -1].detach().cpu().numpy()
    return hs


def get_hidden_states(model, tokenizer, input_text, layer=-1, model_type="encoder"):
    fn = {
        "encoder": get_encoder_hidden_states,
        "encoder_decoder": get_encoder_decoder_hidden_states,
        "decoder": get_decoder_hidden_states,
    }[model_type]

    return fn(model, tokenizer, input_text, layer=layer)


def format_text(prompts, labels):
    return [text + f" {label}" for text, label in zip(prompts, labels)]


def get_hidden_states_batch(model, tokenizer, data_btc, layer, model_type):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape
    (n,)
    with the ground truth labels

    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    # setup
    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    # loop
    for i, prompts in tqdm.tqdm(
        enumerate(data_btc), total=len(data_btc), desc="get_hidden_states"
    ):
        g_h, true_label, i_classes, i_names, r_names, _prompts = zip(*prompts)

        # get hidden states
        # neg_hs = get_hidden_states(model, tokenizer, format_text(_prompts, 0), layer=layer, model_type=model_type)
        # pos_hs = get_hidden_states(model, tokenizer, format_text(_prompts, 1), layer=layer, model_type=model_type)

        neg_hs = get_hidden_states(
            model,
            tokenizer,
            format_text(_prompts, 1 - np.asarray(true_label)),
            layer=layer,
            model_type=model_type,
        )
        pos_hs = get_hidden_states(
            model,
            tokenizer,
            format_text(_prompts, true_label),
            layer=layer,
            model_type=model_type,
        )

        # collect
        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(true_label)

    all_neg_hs = np.concatenate(all_neg_hs)
    all_pos_hs = np.concatenate(all_pos_hs)
    all_gt_labels = np.concatenate(all_gt_labels)

    return all_neg_hs, all_pos_hs, all_gt_labels


def get_tokeniser_batch(model, tokenizer, data_btc):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape
    (n,)
    with the ground truth labels

    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    # setup
    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    g_h, all_gt_labels, i_classes, i_names, r_names, _prompts = np.concatenate(
        data_btc
    ).T

    neg_text = format_text(_prompts, 1 - np.asarray(all_gt_labels).astype(int))
    pos_text = format_text(_prompts, all_gt_labels)

    neg_text = [t + tokenizer.eos_token for t in neg_text]
    pos_text = [t + tokenizer.eos_token for t in pos_text]
    all_neg_hs = tokenizer(
        neg_text, return_tensors="pt", padding=True
    ).input_ids  # 72 _3
    all_pos_hs = tokenizer(
        pos_text, return_tensors="pt", padding=True
    ).input_ids  # 72 _3

    return all_neg_hs, all_pos_hs, all_gt_labels
