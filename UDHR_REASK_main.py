import gc
import os

import numpy as np
import torch
import tqdm

from ccs_utils.base_LoRa import make_lora
from ccs_utils.model_generate_utils import model_reply
from setups import add_name, make_df, make_model
from setups import cuda_name, model_name, path2result, wandb
from UDHR.get_UDHR_inputs import make_REASK_prompts, make_UDHR_prompts

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
run_type = "eval_"
post_name = "_4"


# sudo /home/user/eericheva/venv_eth/bin/python3 UDHR_REASK_main.py
def do_lora(model, model1, sampling_kwargs):
    print(model_name)
    # prompts_btc_train, prompts_btc_test = make_UDHR_rather_prompts(post_name=post_name, BTC_SIZE=24,
    # TRTE_split=0.5)  # 52
    BTC_SIZE = 1
    prompts_btc_train, prompts_btc_test = make_UDHR_prompts(
        post_name=model_name, BTC_SIZE=BTC_SIZE, TRTE_split=0.5
    )  # 52
    prompts_btc_test = prompts_btc_train + prompts_btc_test
    prompts_btc_test = prompts_btc_test[:10]

    PATH_TO_DICT = os.path.join("/HDD/weights", f"LORA{model_name}/")
    if not os.path.exists(PATH_TO_DICT):
        os.makedirs(PATH_TO_DICT)

    try:
        num_hidden_layers = int(model.config.num_hidden_layers)
    except:
        num_hidden_layers = int(model.metadata.get("gemma.block_count"))
    for layer in range(num_hidden_layers, num_hidden_layers - 18, -1):
        #########################################################
        print("Layer: " + str(layer))
        p2s = os.path.join(
            path2result,
            f"responses{model_name}_REASK/{run_type}UDHR_REASK{add_name}_results_l_{layer}.csv",
        )
        print(p2s)
        ##########################################################
        make_lora(
            model,
            tokenizer=model.tokenizer,
            data_test=prompts_btc_test,
            layer=layer,
            batch_size=BTC_SIZE,
            PATH_TO_DICT=PATH_TO_DICT,
        )
        gc.collect()
        torch.cuda.empty_cache()
        nom_df, mod_df = [], []
        for i, prompts in tqdm.tqdm(
            enumerate(prompts_btc_test), total=len(prompts_btc_test), desc="model_reply"
        ):
            g_h, labels, i_classes, i_names, r_names, _prompts = zip(*prompts)
            from typing import Dict, Union

            sampling_kwargs: Dict[str, Union[float, int]] = {
                # "temperature":    0.8,
                # "top_p":          0.3,
                "max_new_tokens": 2,
                # "do_sample":      True,
            }
            tmp_nom_df, input, trimmed, probs01 = model_reply(
                model,
                tokenizer=model.tokenizer,
                prompts=_prompts,
                labels=labels,
                return_full=True,
                **sampling_kwargs,
            )
            nom_df.append(tmp_nom_df)
            gc.collect()
            torch.cuda.empty_cache()
            reask_prompts = make_REASK_prompts(input, trimmed)
            tmp_mod_df, input1, trimmed1, prob011 = model_reply(
                model1,
                tokenizer=model.tokenizer,
                prompts=reask_prompts,
                labels=labels,
                return_full=True,
                **sampling_kwargs,
            )
            mod_df.append(tmp_mod_df)
        nom_df = np.concatenate(nom_df, axis=0)
        mod_df = np.concatenate(mod_df, axis=0)
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
        del df
        del nom_df
        del mod_df

    wandb.finish()


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
    # model1, sampling_kwargs = make_model()
    do_lora(model, None, sampling_kwargs)
