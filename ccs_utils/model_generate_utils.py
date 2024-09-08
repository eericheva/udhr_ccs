import gc
import re

import torch


# {"0": 235276,
# "1": 235274}
def model_reply(
    model, tokenizer, prompts, labels, return_full=False, **sampling_kwargs
):
    gc.collect()
    torch.cuda.empty_cache()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(
        **inputs, **sampling_kwargs, output_logits=True, return_dict_in_generate=True
    )
    output_tokens = output.sequences.detach().cpu().numpy()
    probs01 = None
    # import numpy as np
    # from torch import nn
    # output_text = [tokenizer.decode(t.tolist()) for t in output_tokens[:, -2:]]
    # txt_ids = [np.where(["0" in oo or "1" in oo for oo in o])[0][0] for o in output_text]
    # output_logits = [nn.Softmax()(output.logits[i]).detach().cpu().numpy()
    #                  for i in range(len(output.logits))]
    # probs01 = [(output_logits[txt_ids[i]][i][235276], output_logits[txt_ids[i]][i][235274])
    #            for i in range(len(txt_ids))]
    output, input, trimmed = _get_trimmed(output_tokens, tokenizer, prompts, labels)
    gc.collect()
    torch.cuda.empty_cache()
    if return_full:
        return output, input, trimmed, probs01
    return output


def _get_trimmed(output_tokens: torch.Tensor, tokenizer, prompts, labels):
    completions = [
        tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in output_tokens
    ]
    input = [c[: len(p)] for p, c in zip(prompts, completions)]
    trimmed = [c[len(p) :] for p, c in zip(prompts, completions)]
    output = _get_number_from_trimmed(trimmed, labels)
    return output, input, trimmed


def _get_number_from_trimmed(trimmed, labels):
    trn = []
    for t, l in zip(trimmed, labels):
        try:
            t = re.findall(r"\d+", t)[0]
            # if l==0:
            #     t = 1 - int(t)
            trn.append(int(t))
        except:
            trn.append("None")
    return trn
