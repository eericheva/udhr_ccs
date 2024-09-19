import gc
import os

import numpy as np
import torch
import tqdm

from ccs_utils.base_LoRa import check_lora, eval_lora
from ccs_utils.model_generate_utils import model_reply
from HH.get_HH_inputs import make_HH_prompts
from setups import add_name, make_df, make_model
from setups import cuda_name, model_name, path2result, wandb

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
run_type = "eval_"


###??? 64


# sudo /home/user/eericheva/venv_eth/bin/python3 UDHR_LORA_main.py
def do_lora():
    print(model_name)
    BTC_SIZE = 2  # 2!!!
    prompts_btc_train, prompts_btc_test = make_HH_prompts(
        post_name=model_name, BTC_SIZE=6
    )
    prompts_btc_test = prompts_btc_train + prompts_btc_test
    prompts_btc_test = prompts_btc_test[:200]

    PATH_TO_DICT = os.path.join("/HDD/weights", f"LORA{model_name}/")
    if not os.path.exists(PATH_TO_DICT):
        os.makedirs(PATH_TO_DICT)

    # try:
    #     num_hidden_layers = int(model.config.num_hidden_layers)
    # except:
    #     num_hidden_layers = int(model.metadata.get('gemma.block_count'))
    num_hidden_layers = 18
    for layer in range(num_hidden_layers, 0, -1):
        ################################################################
        print("Layer: " + str(layer))
        p2s = os.path.join(
            path2result,
            f"responses{model_name}_LORA/{run_type}HH_LORA{add_name}_results_l_{layer}.csv",
        )
        print(p2s)
        model, sampling_kwargs = make_model()
        gc.collect()
        torch.cuda.empty_cache()
        ################################################################
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
            try:
                ccs_acc = (
                    np.asarray(tmp_nom_df).astype(int) == np.asarray(labels)
                ).mean()
                acc.append(ccs_acc)
            except:
                print(
                    "Error : ccs_acc = (np.asarray(tmp_nom_df).astype(int) == np.asarray(labels)).mean()"
                )
                print(tmp_nom_df)
                print(labels)
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
        print(
            "Layer "
            + str(layer)
            + " is done. "
            + str(model.config.num_hidden_layers - 1 - layer)
            + " remaining"
        )
        df.to_csv(open(p2s, "w"), sep="\t")
        print(p2s)
        print(model_name + add_name)
        del df
        del nom_df
        del mod_df

    wandb.finish()


if __name__ == "__main__":
    do_lora()
