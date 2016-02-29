from utils import get_hypotheses

HYP_FILE = "./data/train-test.hyp1-hyp2-ref"
GOLD_FILE = "./data/train.gold"
OUTPUT_FILE = "./output.pred"
TRADEOFF_PARAM = 0.35# between 0 and 1

[labeled_instances, unlabeled_instances] = get_hypotheses(HYP_FILE, GOLD_FILE)

def get_word_count_dict(sentence):
    word_list = sentence.split(" ")
    tot_len = len(word_list)
    wcount_dict = {}
    for word in word_list:
        wcount_dict[word] = wcount_dict.get(word, 0.0) + 1.0
    return(wcount_dict, tot_len)

def get_precision(hyp, ref):
    """
    #Currently ignoring multiple occurrences of words
    ref_words = set(ref.split(" "))
    hyp_words = hyp.split(" ")
    wcount_hyp = len(hyp_words)
    match_count = 0.0
    for hyp_word in hyp_words:
        if hyp_word in ref_words:
            match_count += 1.0
    return(match_count/wcount_hyp)
    """
    # Accounting for number of occurrences of words
    [ref_wc, ref_len] = get_word_count_dict(ref)
    [hyp_wc, hyp_len] = get_word_count_dict(hyp)
    match_count = 0.0
    for hyp_word in hyp_wc.keys():
        if ref_wc.get(hyp_word, None) is not None:
            match_count += min(ref_wc[hyp_word], hyp_wc[hyp_word])
    return(match_count/hyp_len)


def get_recall(hyp, ref):
    """
    hyp_words = set(hyp.split(" "))
    ref_words = ref.split(" ")
    wcount_ref = len(ref_words)
    match_count = 0.0
    for ref_word in ref_words:
        if ref_word in hyp_words:
            match_count += 1.0
    return(match_count/wcount_ref)
    """
    # Accounting for number of occurrences of words
    [ref_wc, ref_len] = get_word_count_dict(ref)
    [hyp_wc, hyp_len] = get_word_count_dict(hyp)
    match_count = 0.0
    for ref_word in ref_wc.keys():
        if hyp_wc.get(ref_word, None) is not None:
            match_count += min(ref_wc[ref_word], hyp_wc[ref_word])
    return (match_count / ref_len)

def get_whm(prec, rec, tradeoff_param):
    if prec == 0 or rec == 0:
        return(-1)
    return(1.0 / ((tradeoff_param/prec) + ((1 - tradeoff_param)/rec)))



def get_prediction(parallel_instance, tradeoff_param):
    [hyp1, hyp2, ref] = parallel_instance
    h1_prec = get_precision(hyp1, ref)
    h1_rec = get_recall(hyp1, ref)
    h1_met = get_whm(h1_prec, h1_rec, tradeoff_param)
    h2_prec = get_precision(hyp2, ref)
    h2_rec = get_recall(hyp2, ref)
    h2_met = get_whm(h2_prec, h2_rec, tradeoff_param)

    if h1_met > h2_met:
        return(-1)
    elif h2_met > h1_met:
        return(1)
    else:
        return(0)


with open(OUTPUT_FILE, 'w') as op_file:
    for instance in labeled_instances:
        parallel_instance = instance[0]
        op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')
    for instance in unlabeled_instances:
        parallel_instance = instance
        op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')