import os

import numpy as np
import tqdm

from ccs_utils.base_CCS import check_ccs
from ccs_utils.model_generate_utils import model_reply
from setups import cuda_name, model_name, path2result, wandb
from setups import make_df, make_model, make_UDHR_prompts

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name


def do_ccs(model, sampling_kwargs):
    print(model_name)
    BTC_SIZE = 28  # 52
    prompts_btc = make_UDHR_prompts(model_name, BTC_SIZE=BTC_SIZE)
    TRTE_split = int(len(prompts_btc) * 0.5)
    prompts_btc_train = prompts_btc[:TRTE_split]
    prompts_btc_test = prompts_btc[-TRTE_split:]

    try:
        num_hidden_layers = int(model.config.num_hidden_layers)
    except:
        num_hidden_layers = int(model.metadata.get("gemma.block_count"))
    for layer in range(num_hidden_layers, 0, -1):
        print("Layer: " + str(layer))
        # Fetch the comparison data check_ccs check_lr
        mod_df = check_ccs(
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
        df.to_csv(
            open(
                os.path.join(
                    path2result,
                    f"responses{model_name}_CCS/UDHR_CCS_results_l_{layer}.csv",
                ),
                "w",
            ),
            sep="\t",
        )
        print(model_name)
        del df
        del nom_df
        del mod_df
    # df.to_csv(open(os.path.join(path2result, f"responses{post_name}/UDHR_CCS_results_{rep_n}.csv"), "w"), sep="\t")
    print(model_name)

    wandb.finish()


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
    do_ccs(model, sampling_kwargs)
