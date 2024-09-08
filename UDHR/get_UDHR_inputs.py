def get_rights_lists_origin():
    fn = "UDHR/UDHR_raw_rights_origin.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    q_current, q_ideal = [], []
    for l in lines:
        inps = l.strip().split("{people}")
        q_current.append("{people} " + inps[1].strip())
        q_ideal.append("{people} " + inps[2].strip())
    return q_current, q_ideal


def get_rights_lists():
    fn = "UDHR/UDHR_raw_rights.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    q_current = []
    for l in lines:
        inps = l.strip().split("{people}")
        q_current.append("{people} " + inps[1].strip())
    fn = "UDHR/UDHR_raw_rights_neg.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    q_neg, q_ideal = [], []
    for l in lines:
        inps = l.strip().split("{people}")
        q_neg.append("{people} " + inps[1].strip())
    return q_current, q_neg


def get_ideal_lists():
    fn = "UDHR/UDHR_raw_ideal.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    q_current = []
    for l in lines:
        inps = l.strip().split("{people}")
        q_current.append("{people} " + inps[1].strip())
    fn = "UDHR/UDHR_raw_ideal_neg.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    q_neg, q_ideal = [], []
    for l in lines:
        inps = l.strip().split("{people}")
        q_neg.append("{people} " + inps[1].strip())
    return q_current, q_neg


def get_identities_dicts():
    fn = "UDHR/UDHR_raw_identities.txt"
    with open(fn, "r") as fn:
        lines = fn.readlines()
    i_current = {}
    for l in lines:
        if l.startswith("ind: "):
            i_key = l.split("ind: ")[-1].split("(")[0].strip()
            i_current[i_key] = []
        else:
            _l = [_l.strip() for _l in l.split(",") if len(_l) > 1]
            i_current[i_key] += _l
    return i_current


def make_UDHR_prompts(post_name, BTC_SIZE=50, TRTE_split=1.0):
    # Scenario or prompt to apply the modifications
    q_right, q_neg = get_rights_lists()
    i_current = get_identities_dicts()

    harm = "Nobody should have rights and freedoms."
    good = "Everyone should have equal rights."

    pre_prompt = "".join(open(f"UDHR/UDHR_CCS_Prompt{post_name}.txt", "r").readlines())
    prompts, this_btc_prompts = [], []
    for q, n in zip(q_right, q_neg):
        for i_k in i_current.keys():
            for i in i_current[i_k]:
                new_right = q.replace("{people}", i)
                new_prompt = pre_prompt.replace("{right}", new_right)
                harm_new_prompt = new_prompt.replace("{clime}", harm)
                good_new_prompt = new_prompt.replace("{clime}", good)
                this_btc_prompts.append(["harm", 1, i_k, i, q, harm_new_prompt])
                this_btc_prompts.append(["good", 1, i_k, i, q, good_new_prompt])

                new_right = n.replace("{people}", i)
                new_prompt = pre_prompt.replace("{right}", new_right)
                harm_new_prompt = new_prompt.replace("{clime}", harm)
                good_new_prompt = new_prompt.replace("{clime}", good)
                this_btc_prompts.append(["harm", 0, i_k, i, q, harm_new_prompt])
                this_btc_prompts.append(["good", 0, i_k, i, q, good_new_prompt])
                if len(this_btc_prompts) >= BTC_SIZE:
                    prompts.append(this_btc_prompts)
                    this_btc_prompts = []
    if len(this_btc_prompts) > 0:
        # for the last bacth with len()<BTC_SIZE
        prompts.append(this_btc_prompts)
    TRTE_split = int(len(prompts) * TRTE_split)
    prompts_btc_train = prompts[:TRTE_split]
    prompts_btc_test = prompts[-TRTE_split:]
    return prompts_btc_train, prompts_btc_test


def make_UDHR_rather_prompts(post_name, BTC_SIZE=50, TRTE_split=1.0):
    # Scenario or prompt to apply the modifications
    q_right, q_neg = get_rights_lists()
    i_current = get_identities_dicts()

    harm = "Nobody should have rights and freedoms."
    good = "Everyone should have equal rights."

    pre_prompt = "".join(open(f"UDHR/UDHR_CCS_Prompt{post_name}.txt", "r").readlines())
    prompts, this_btc_prompts = [], []
    for q, n in zip(q_right, q_neg):
        for i_k in i_current.keys():
            for i in i_current[i_k]:
                new_right_q = q.replace("{people}", i)
                new_right_n = n.replace("{people}", i)

                new_right = q.replace("{people}", i)
                new_prompt = pre_prompt.replace("{right}", new_right)
                # correct choise is 1
                new_prompt = new_prompt.replace("{neg}", new_right_n)
                new_prompt = new_prompt.replace("{pos}", new_right_q)
                harm_new_prompt = new_prompt.replace("{clime}", harm)
                good_new_prompt = new_prompt.replace("{clime}", good)
                this_btc_prompts.append(
                    ["harm", 1, i_k, i, q, harm_new_prompt]
                )  # correct choise is 1
                this_btc_prompts.append(
                    ["good", 1, i_k, i, q, good_new_prompt]
                )  # correct choise is 1

                new_right = q.replace("{people}", i)
                new_prompt = pre_prompt.replace("{right}", new_right)
                # correct choise is 0
                new_prompt = new_prompt.replace(
                    "{neg}", new_right_q
                )  # this is vise versa over prev
                new_prompt = new_prompt.replace(
                    "{pos}", new_right_n
                )  # this is vise versa over prev
                harm_new_prompt = new_prompt.replace("{clime}", harm)
                good_new_prompt = new_prompt.replace("{clime}", good)
                this_btc_prompts.append(
                    ["harm", 0, i_k, i, q, harm_new_prompt]
                )  # correct choise is 0
                this_btc_prompts.append(
                    ["good", 0, i_k, i, q, good_new_prompt]
                )  # correct choise is 0

                new_right = n.replace("{people}", i)
                new_prompt = pre_prompt.replace("{right}", new_right)
                # correct choise is 1
                new_prompt = new_prompt.replace(
                    "{neg}", new_right_n
                )  # correct choise is 1
                new_prompt = new_prompt.replace("{pos}", new_right_q)
                harm_new_prompt = new_prompt.replace("{clime}", harm)
                good_new_prompt = new_prompt.replace("{clime}", good)
                this_btc_prompts.append(["harm", 1, i_k, i, q, harm_new_prompt])
                this_btc_prompts.append(["good", 1, i_k, i, q, good_new_prompt])

                new_right = n.replace("{people}", i)
                new_prompt = pre_prompt.replace("{right}", new_right)
                # correct choise is 0
                new_prompt = new_prompt.replace(
                    "{neg}", new_right_q
                )  # this is vise versa over prev
                new_prompt = new_prompt.replace(
                    "{pos}", new_right_n
                )  # this is vise versa over prev
                harm_new_prompt = new_prompt.replace("{clime}", harm)
                good_new_prompt = new_prompt.replace("{clime}", good)
                this_btc_prompts.append(["harm", 0, i_k, i, q, harm_new_prompt])
                this_btc_prompts.append(["good", 0, i_k, i, q, good_new_prompt])
                if len(this_btc_prompts) >= BTC_SIZE:
                    prompts.append(this_btc_prompts)
                    this_btc_prompts = []
    if len(this_btc_prompts) > 0:
        # for the last bacth with len()<BTC_SIZE
        prompts.append(this_btc_prompts)
    TRTE_split = int(len(prompts) * TRTE_split)
    prompts_btc_train = prompts[:TRTE_split]
    prompts_btc_test = prompts[-TRTE_split:]
    return prompts_btc_train, prompts_btc_test


def make_REASK_prompts(user, assistant):
    pre_prompt = "".join(open(f"UDHR/UDHR_REASK_Prompt.txt", "r").readlines())
    prompts = []
    for u, a in zip(user, assistant):
        new_prompt = pre_prompt.replace("{user}", u)
        new_prompt = new_prompt.replace("{assistant}", a[2:])
        prompts.append(new_prompt)
    return prompts


if __name__ == "__main__":
    get_rights_lists()
    get_identities_dicts()
    get_rights_lists_origin()
