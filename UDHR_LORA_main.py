import gc
import os

import numpy as np
import torch
import tqdm

from ccs_utils.base_LoRa import check_lora, eval_lora
from ccs_utils.model_generate_utils import model_reply
from setups import add_name, make_df, make_model
from setups import cuda_name, model_name, path2result, wandb
from UDHR.get_UDHR_inputs import make_UDHR_prompts

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
run_type = "eval_"
post_name = "_1"


# post_name = "_base_short_rather"


# run_type = ""
# post_name = model_name


# sudo /home/user/eericheva/venv_eth/bin/python3 UDHR_LORA_main.py
def do_lora():
    print(model_name)
    BTC_SIZE = 16
    # prompts_btc_train, prompts_btc_test = make_UDHR_rather_prompts(
    #         post_name=post_name, BTC_SIZE=24, TRTE_split=0.5
    #         )  # 52
    prompts_btc_train, prompts_btc_test = make_UDHR_prompts(
        post_name=post_name, BTC_SIZE=52, TRTE_split=0.5
    )  # 52, 120
    if "eval" in run_type:
        prompts_btc_test = prompts_btc_train + prompts_btc_test
    prompts_btc_train = prompts_btc_train
    prompts_btc_test = prompts_btc_test

    PATH_TO_DICT = os.path.join("/HDD/weights", f"LORA{model_name}/")
    if not os.path.exists(PATH_TO_DICT):
        os.makedirs(PATH_TO_DICT)

    # try:
    #     num_hidden_layers = int(model.config.num_hidden_layers)
    # except:
    #     num_hidden_layers = int(model.metadata.get('gemma.block_count'))
    num_hidden_layers = 18
    for layer in range(num_hidden_layers, 0, -1):
        # for layer in [6]:  # 2, 5, 6
        ###################################################################
        print("Layer: " + str(layer))
        p2s = os.path.join(
            path2result,
            f"responses{model_name}_LORA/{run_type}{post_name}UDHR_LORA{add_name}_results_l_{layer}.csv",
        )
        print(p2s)
        model, sampling_kwargs = make_model()
        gc.collect()
        torch.cuda.empty_cache()
        ###################################################################
        # Fetch the comparison data check_ccs check_lr
        if "eval" in run_type:
            mod_df = eval_lora(
                model,
                tokenizer=model.tokenizer,
                data_test=prompts_btc_test,
                layer=layer,
                batch_size=BTC_SIZE,
                PATH_TO_DICT=PATH_TO_DICT,
            )
        else:
            mod_df = check_lora(
                model,
                tokenizer=model.tokenizer,
                data_train=prompts_btc_train,
                data_test=prompts_btc_test,
                layer=layer,
                batch_size=BTC_SIZE,
                PATH_TO_DICT=PATH_TO_DICT,
            )
        gc.collect()
        torch.cuda.empty_cache()
        nom_df, acc = [], []
        progress_bar_val = tqdm.tqdm(
            range(len(prompts_btc_test)),
            total=len(prompts_btc_test),
            desc=f"MODEL REPLY RUN" + f"ModelReplyAcc: {round(np.mean(acc), 6)}",
        )
        # for i, prompts in tqdm.tqdm(enumerate(prompts_btc_test), total=len(prompts_btc_test), desc="model_reply"):
        for j in progress_bar_val:
            g_h, labels, i_classes, i_names, r_names, _prompts = zip(
                *prompts_btc_test[j]
            )
            tmp_nom_df = model_reply(
                model,
                tokenizer=model.tokenizer,
                prompts=_prompts,
                labels=labels,
                **sampling_kwargs,
            )
            nom_df.append(tmp_nom_df)
            none_inds = np.where(np.asarray(tmp_nom_df) != "None")
            ccs_acc = (
                np.asarray(tmp_nom_df)[none_inds].astype(int)
                == np.asarray(labels)[none_inds]
            ).mean()
            acc.append(ccs_acc)
            gc.collect()
            torch.cuda.empty_cache()
            progress_bar_val.set_description(
                desc=f"MODEL REPLY RUN, " + f"ModelReplyAcc: {round(np.mean(acc), 6)}",
                refresh=False,
            )
        nom_df = np.concatenate(nom_df, axis=0)
        g_h, labels, i_classes, i_names, r_names, prompts = np.concatenate(
            prompts_btc_test, axis=0
        ).T
        df = make_df(
            *[nom_df, mod_df],
            prompts=prompts,
            layer=layer,
            i_class=i_classes,
            i_name=i_names,
            r_name=r_names,
            g_h=g_h,
            gt=labels,
        )
        print("Layer " + str(layer) + " is done. ")
        df.to_csv(open(p2s, "w"), sep="\t")
        print(p2s)
        del df
        del nom_df
        del mod_df

    wandb.finish()


if __name__ == "__main__":
    do_lora()
