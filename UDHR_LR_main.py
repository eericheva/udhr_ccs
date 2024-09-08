import os

import numpy as np
import pandas as pd
import tqdm

from ccs_utils.base_CCS import check_lr
from ccs_utils.model_generate_utils import model_reply
from setups import make_df, make_model, make_UDHR_prompts

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path2result = "../ethical_llms_data/UDHR_CCS/"
post_name = "_4"


def do_ccs(model, sampling_kwargs):
    print(post_name)
    BTC_SIZE = 52
    prompts_btc = make_UDHR_prompts(post_name, BTC_SIZE=BTC_SIZE)
    TRTE_split = int(len(prompts_btc) * 0.5)
    prompts_btc_train = prompts_btc[:TRTE_split]
    prompts_btc_test = prompts_btc[-TRTE_split:]

    # We'll store 5 repetitions per layer, for all layers
    df = pd.DataFrame(
        columns=[
            "prompts",
            "completions",
            "is_harm",
            "layer",
            "id_class",
            "identifier",
            "right",
        ]
    )
    try:
        num_hidden_layers = int(model.config.num_hidden_layers)
    except:
        num_hidden_layers = int(model.metadata.get("gemma.block_count"))
    for layer in range(num_hidden_layers, 0, -1):
        print("Layer: " + str(layer))
        # Fetch the comparison data check_ccs check_lr
        mod_df = check_lr(
            model,
            tokenizer=model.tokenizer,
            data_train=prompts_btc_train,
            data_test=prompts_btc_test,
            layer=layer,
            model_type="decoder",
        )
        nom_df = []
        for i, prompts in tqdm.tqdm(
            enumerate(prompts_btc_test), total=len(prompts_btc_test)
        ):
            g_h, labels, i_classes, i_names, r_names, _prompts = zip(*prompts)
            nom_df.append(
                model_reply(
                    model,
                    tokenizer=model.tokenizer,
                    prompts=_prompts,
                    labels=labels,
                    **sampling_kwargs,
                )
            )
        nom_df = np.concatenate(nom_df, axis=0)
        g_h, labels, i_classes, i_names, r_names, prompts = np.concatenate(
            prompts_btc_test, axis=0
        ).T
        comparison_df = make_df(
            *[nom_df, mod_df],
            prompts=prompts,
            layer=layer,
            i_class=i_classes,
            i_name=i_names,
            r_name=r_names,
            g_h=g_h,
        )
        df = pd.concat([df, comparison_df], ignore_index=True)
        print(
            "Layer "
            + str(layer)
            + " is done. "
            + str(model.config.num_hidden_layers - 1 - layer)
            + " remaining"
        )
        df.to_csv(
            open(
                os.path.join(
                    path2result, f"responses{post_name}/UDHR_LR_results_l_{layer}.csv"
                ),
                "w",
            ),
            sep="\t",
        )
    df.to_csv(
        open(
            os.path.join(path2result, f"responses{post_name}_LR/UDHR_LR_results.csv"),
            "w",
        ),
        sep="\t",
    )
    print(post_name)


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
    do_ccs(model, sampling_kwargs)
