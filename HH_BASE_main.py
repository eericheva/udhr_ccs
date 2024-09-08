import gc
import os

import numpy as np
import pandas as pd
import torch
import tqdm

from ccs_utils.model_generate_utils import model_reply
from HH.get_HH_inputs import make_HH_prompts
from setups import cuda_name, path2result, model_name
from setups import make_df, make_model

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name


def do_base(model, sampling_kwargs):
    prompts_btc_tr, prompts_btc_te = make_HH_prompts(model_name, BTC_SIZE=12)  # 100
    prompts_btc = prompts_btc_tr + prompts_btc_te
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
    counter = 0
    layer = 0
    prompts_btc = prompts_btc[counter:]
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
        counter += 1
        if counter % 200 == 0:
            df.to_csv(
                open(
                    os.path.join(
                        path2result,
                        f"tmp_responses{model_name}/HH_BASE_results_l_{layer}_{counter}.csv",
                    ),
                    "w",
                ),
                sep="\t",
            )
    df.to_csv(
        open(
            os.path.join(
                path2result, f"responses{model_name}/HH_BASE_results_l_{layer}.csv"
            ),
            "w",
        ),
        sep="\t",
    )


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
    do_base(model, sampling_kwargs)
    print(model_name)
