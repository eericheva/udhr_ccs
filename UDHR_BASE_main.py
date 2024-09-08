import gc
import os

import numpy as np
import pandas as pd
import torch
import tqdm

from ccs_utils.model_generate_utils import model_reply
from setups import cuda_name, model_name, path2result
from setups import make_df, make_model
from UDHR.get_UDHR_inputs import make_UDHR_rather_prompts

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
post_name = "_base_short_rather"


def do_base(model, sampling_kwargs):
    # prompts_btc_train, prompts_btc_test = make_UDHR_prompts(model_name, BTC_SIZE=40)
    prompts_btc_train, prompts_btc_test = make_UDHR_rather_prompts(
        post_name=post_name, BTC_SIZE=80, TRTE_split=0.5
    )  # 52
    # 100
    prompts_btc = prompts_btc_train + prompts_btc_test
    print(model_name)
    # We'll store 5 repetitions per layer, for all layers
    df = pd.DataFrame(
        columns=[
            "prompts",
            "completions",
            "is_harm",
            "is_modified",
            "layer",
            "id_class",
            "identifier",
            "right",
            "gt",
        ]
    )
    try:
        num_hidden_layers = int(model.config.num_hidden_layers)
    except:
        num_hidden_layers = int(model.metadata.get("gemma.block_count"))
    for layer in range(0, 1):
        print("Layer: " + str(layer))
        # Fetch the comparison data
        for i, prompts_b in tqdm.tqdm(enumerate(prompts_btc), total=len(prompts_btc)):
            g_h, labels, i_classes, i_names, r_names, prompts = zip(*prompts_b)
            neg_r = np.where(np.asarray(labels) == 0)[0]
            g_h, labels, i_classes, i_names, r_names, prompts = (
                np.asarray(g_h)[neg_r],
                np.asarray(labels)[neg_r],
                np.asarray(i_classes)[neg_r],
                np.asarray(i_names)[neg_r],
                np.asarray(r_names)[neg_r],
                np.asarray(prompts)[neg_r],
            )
            mod_df = model_reply(
                model,
                tokenizer=model.tokenizer,
                prompts=prompts.tolist(),
                labels=labels,
                **sampling_kwargs,
            )
            gc.collect()
            torch.cuda.empty_cache()

            g_h, labels, i_classes, i_names, r_names, prompts = zip(*prompts_b)
            pos_r = np.where(np.asarray(labels) == 1)[0]
            g_h, labels, i_classes, i_names, r_names, prompts = (
                np.asarray(g_h)[pos_r],
                np.asarray(labels)[pos_r],
                np.asarray(i_classes)[pos_r],
                np.asarray(i_names)[pos_r],
                np.asarray(r_names)[pos_r],
                np.asarray(prompts)[pos_r],
            )
            nom_df = model_reply(
                model,
                tokenizer=model.tokenizer,
                prompts=prompts.tolist(),
                labels=labels,
                **sampling_kwargs,
            )
            gc.collect()
            torch.cuda.empty_cache()

            comparison_df = make_df(
                *[nom_df, mod_df],
                prompts=prompts,
                layer=layer,
                i_class=i_classes,
                i_name=i_names,
                r_name=r_names,
                g_h=g_h,
                gt=labels,
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
                    path2result, f"responses{post_name}/UDHR_BASE_results_l_{layer}.csv"
                ),
                "w",
            ),
            sep="\t",
        )
        print(post_name)


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
    do_base(model, sampling_kwargs)
    print(model_name)
