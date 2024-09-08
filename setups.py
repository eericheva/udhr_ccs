from typing import Dict, Union

import pandas as pd
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from keys import WANBD_KEY

wandb.login(key=WANBD_KEY)
model_name = "_4"  # "_base_short"
add_name = "_proba_loss"  # "_spin_loss", "_proba_loss"
cuda_name = "0"
path2result = "../ethical_llms_data/UDHR_CCS/"  # "../ethical_llms_data/HH_CCS/"
# path2result = "../ethical_llms_data/HH_CCS/" # "../ethical_llms_data/UDHR_CCS/"


# 'EleutherAI/gpt-j-6B'
# 'google/gemma-2b-it'


def make_model():
    # download model from huggingface
    # from huggingface_hub import snapshot_download
    # snapshot_download(
    #         repo_id='google/gemma-2b-it',
    #         local_dir=os.path.join('/HDD/models/', 'google/gemma-2b-it'),
    #         token="",
    #         force_download=True,
    #         )

    _ = torch.set_grad_enabled(False)

    # *******************************************************************
    # if GPU run
    # bitsandbytes-foundation/bitsandbytes#40, quantization isn't really supported on cpu.
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    # model_path: str = "/HDD/models/google/Gemma-2B/"
    model_path: str = "/HDD/models/google/gemma-2b-it/"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    model.tokenizer = AutoTokenizer.from_pretrained(model_path)

    # *******************************************************************
    # if CPU gguf transformer lib run - for LLaMa, Mistral, Qwen2 models
    # pip install gguf sentencepiece
    # model_path: str = "/HDD/models/google/gemma-2b-it/"
    # model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         gguf_file="gemma-2b-it.gguf",
    #         low_cpu_mem_usage=True,
    #         device_map='cpu'
    #         )
    # model.tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file="gemma-2b-it.gguf")

    # *******************************************************************
    # for CPU gguf llama_cpp run - for Gemma
    # pip install llama-cpp-python
    # model_path: str = "/HDD/models/google/gemma-2b-it/gemma-2b-it.gguf"
    # model = Llama(
    #         model_path=model_path,
    #         torch_dtype=torch.bfloat16,
    #         n_ctx=0,
    #         n_threads=16,
    #         use_mlock=True,
    #         flash_attn=True,
    #         n_batch = 50,
    #         # logits_all=True,
    #         # device_map="auto",
    #         # token=HUGGINGFACE_TOKEN,
    #         verbose=False,
    #         )
    # model.tokenizer = LlamaTokenizer(model)

    # *******************************************************************
    # # if CPU transformer lib run
    # model_path: str = "/HDD/models/google/gemma-2b-it/"
    # model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         low_cpu_mem_usage=True,
    #         device_map='cpu'
    #         )
    # model.tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_kwargs: Dict[str, Union[float, int]] = {
        "temperature": 0.8,
        "top_p": 0.3,
        "max_new_tokens": 2,
        "do_sample": True,
    }

    return model, sampling_kwargs


def make_df(nom_df, mod_df, prompts, layer, i_class, i_name, r_name, g_h, gt):
    # i_class, i_name, r_name = zip(*[(c, n, r) for c, n, r, _g_h in zip(i_class, i_name, r_name, g_h) if _g_h ==
    # "good"])
    # g_prompts = [p for p, _g_h in zip(prompts, g_h) if _g_h == "good"]
    # nom_df = [p for p, _g_h in zip(nom_df, g_h) if _g_h == "good"]
    nom_df = pd.DataFrame(
        {
            "prompts": prompts,
            "completions": nom_df,
            "is_harm": [False if _g_h == "good" else True for _g_h in g_h],
            "is_modified": False,
            "layer": layer,
            "id_class": i_class,
            "identifier": i_name,
            "right": r_name,
            "gt": gt,
        }
    )
    # i_class, i_name, r_name = zip(*[(c, n, r) for c, n, r, _g_h in zip(i_class, i_name, r_name, g_h) if _g_h ==
    # "harm"])
    # h_prompts = [p for p, _g_h in zip(prompts, g_h) if _g_h == "harm"]
    # mod_df = [p for p, _g_h in zip(mod_df, g_h) if _g_h == "good"]
    mod_df = pd.DataFrame(
        {
            "prompts": prompts,
            "completions": mod_df,
            "is_harm": [False if _g_h == "good" else True for _g_h in g_h],
            "is_modified": True,
            "layer": layer,
            "id_class": i_class,
            "identifier": i_name,
            "right": r_name,
            "gt": gt,
        }
    )
    return pd.concat([nom_df, mod_df], ignore_index=True)


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
