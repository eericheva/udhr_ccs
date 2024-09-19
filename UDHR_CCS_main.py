import gc
import os

import numpy as np
import torch
import tqdm

from ccs_utils.base_CCS import check_ccs
from ccs_utils.model_generate_utils import model_reply
from setups import cuda_name, model_name, path2result, wandb
from setups import make_df, make_model
from UDHR.get_UDHR_inputs import make_UDHR_prompts

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
run_type = "eval_"
# run_type = ""
post_name = "_2"


# post_name = '_base_short_rather'
# post_name = '_4'


def do_ccs(model, sampling_kwargs):
    print(model_name)
    BTC_SIZE = 52  # 52
    prompts_btc_train, prompts_btc_test = make_UDHR_prompts(
        post_name=model_name, BTC_SIZE=52, TRTE_split=0.5
    )  # 52, 120
    prompts_btc_train1, prompts_btc_test1 = make_UDHR_prompts(
        post_name=post_name, BTC_SIZE=52, TRTE_split=0.5
    )  # 52, 120
    # prompts_btc_train1, prompts_btc_test1 = make_UDHR_rather_prompts(
    #         post_name='_base_short_rather', BTC_SIZE=52, TRTE_split=0.5
    #         )  # 52, 120
    # prompts_btc_train1, prompts_btc_test1 = make_HH_prompts(post_name="_4", BTC_SIZE=6)
    if "eval" in run_type:
        prompts_btc_test = prompts_btc_train1 + prompts_btc_test1
    prompts_btc_train = prompts_btc_train
    prompts_btc_test = prompts_btc_test

    # try:
    #     num_hidden_layers = int(model.config.num_hidden_layers)
    # except:
    #     num_hidden_layers = int(model.metadata.get("gemma.block_count"))
    num_hidden_layers = 18
    for layer in range(num_hidden_layers, 0, -1):
        ###################################################################
        print("Layer: " + str(layer))
        p2s = os.path.join(
            path2result,
            f"responses{model_name}_CCS/{run_type}{post_name}UDHR_CCS_results_l_{layer}.csv",
        )
        print(p2s)
        model, sampling_kwargs = make_model()
        gc.collect()
        torch.cuda.empty_cache()
        ###################################################################
        # Fetch the comparison data check_ccs check_lr
        mod_df = check_ccs(
            model,
            tokenizer=model.tokenizer,
            data_train=prompts_btc_train,
            data_test=prompts_btc_test,
            layer=layer,
            model_type="decoder",
        )
        gc.collect()
        torch.cuda.empty_cache()
        nom_df, acc = [], []
        progress_bar_val = tqdm.tqdm(
            range(len(prompts_btc_test)),
            total=len(prompts_btc_test),
            desc=f"MODEL REPLY RUN" + f"ModelReplyAcc: {round(np.mean(acc), 6)}",
        )
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
        g_h, labels, i_classes, i_names, r_names, prompts = np.concatenate(
            prompts_btc_test, axis=0
        ).T
        nom_df = np.concatenate(nom_df, axis=0)
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
    model, sampling_kwargs = make_model()
    do_ccs(model, sampling_kwargs)
