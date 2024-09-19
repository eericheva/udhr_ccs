# UDHR_CSS

# Discovering Latent Knowledge in LLMs Without Supervision on LORA adapters. LORA for CCS.

_For AI Safety Fundamentals AI Alignment Course project (summer 2024)._

Main idea: Train LORA for Contrast-Consistent Search. Find if trained LORA can handle harm prompt, that was included into inference prompt, and steer model to safe behaviour according to the safe clime, added to the trained prompt.

Existing techniques for training language models can be misaligned with the truth. Burns et al. 2022 proposed a method for direct search of latent knowledge inside the internal activations of a language model in a purely unsupervised way. Specifically, they introduce a method for accurately answering yes-no questions given only unlabeled model activations. It works by finding a direction in activation space that satisfies logical consistency properties, such as that a statement and its negation have opposite truth values. This directions is searching by linear probas, applied to different layers of the model, and trained on a contrastive loss between true and false labels for input statement.

We proceed this idea further for circumventing following issue. Due to linear probas linear nature, trained on some specific trained data distribution, they can found linear separation for trained data only. So this method could be used for latent knowledge discovering on trained dataset only. Linear probes do not generalize and do not work for another distribution of data (for other type of tasks or for QA on different topics). This issue, which arises when applying linear probes for CCS to our data, is discussed  and empirically showed further in this blog.

We propose to train nonlinear LORA for residual stream for some model layer and we hope that this LORA, trained on one type of prompts will generalize to other dataset and rephrasing prompts. So we can elicit latent model knowledge (about safe or harm behaviour) just by adding this LORA to model layer.

We also adapted the purpose of CCS method from discovering model latent factual knowledge to discovering model latent knowledge about safe and harm behaviour ot intentions.

Main report - [Discovering Latent Knowledge in LLMs Without Supervision on LORA adapters. LORA for CCS.](https://erichevaelena.substack.com/publish/post/148652551)

Supplementary materials can be found here - [Discovering Latent Knowledge in LLMs Without Supervision on LORA adapters. Supplementary.](https://erichevaelena.substack.com/publish/post/148674505)

Future proceeding can be found here - [Discovering Latent Knowledge in LLMs Without Supervision on LORA adapters. LORA as Constitution and UNIVERSAL LORA](https://erichevaelena.substack.com/publish/post/148674578)
