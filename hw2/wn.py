from utils import get_hypotheses
from collections import Counter
import nltk
from nltk.tag import pos_tag, map_tag
from nltk.corpus import wordnet as wn
import numpy, pickle

HYP_FILE = "./data/train-test.hyp1-hyp2-ref"
GOLD_FILE = "./data/train.gold"
OUTPUT_FILE = "./output-exp.pred"
TRADEOFF_PARAM = 0.35# between 0 and 1

[labeled_instances, unlabeled_instances] = get_hypotheses(HYP_FILE, GOLD_FILE)

def memoize(func):
    memodict = {}
    def inner(stuff):
        if stuff not in memodict:
            memodict[stuff] = func(stuff)
        return memodict[stuff]
    return inner

# nltk can't pos tag unicode
def preprocess(text):
    return ''.join(character for character in text if ord(character)<128)

wntags = {'NOUN':'n','VERB':'v','ADJ':'a','ADV':'r'}

@memoize
def get_word_count_dict(sentence):
    wcount_dict = Counter()

    for (word,tag) in \
        [(w, map_tag('en-ptb','universal',t))
         for (w,t) in nltk.pos_tag(nltk.word_tokenize(preprocess(sentence)))]:
        if tag in wntags.keys():
            wcount_dict.update(wn.synsets(word,wntags[tag]))
            wcount_dict.update([word])
    tot_len = sum(wcount_dict.values())
    if tot_len==0:
        print sentence
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
    for hyp_word in hyp_wc.items():
        if ref_wc.get(hyp_word, None) is not None:
            match_count += min(ref_wc[hyp_word], hyp_wc[hyp_word])
            
    if hyp_len == 0:
        return 0
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

    if ref_len == 0:
        return 0
    return (match_count / ref_len)

def get_whm(prec, rec, tradeoff_param):
    if prec == 0 or rec == 0:
        return(-1)
    return(1.0 / ((tradeoff_param/prec) + ((1 - tradeoff_param)/rec)))



def get_prediction(parallel_instance):
    [hyp1, hyp2, ref] = parallel_instance
    h1_prec = get_precision(hyp1, ref)
    h1_rec = get_recall(hyp1, ref)
#    h1_met = get_whm(h1_prec, h1_rec, tradeoff_param)
    h2_prec = get_precision(hyp2, ref)
    h2_rec = get_recall(hyp2, ref)
#    h2_met = get_whm(h2_prec, h2_rec, tradeoff_param)

    return (h1_prec,h1_rec,h2_prec,h2_rec)

#    if h1_met > h2_met:
#        return(-1)
#    elif h2_met > h1_met:
#        return(1)
#    else:
#        return(0)

features = []
i = 0

for instance in labeled_instances:
    parallel_instance = instance[0]
    features.append(numpy.array(get_prediction(parallel_instance)))
    #op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')
    i+=1
    if i%1000 == 0:
        print "{0} documents processed".format(i)
"""
for instance in unlabeled_instances:
    parallel_instance = instance
    features.append(numpy.array(get_prediction(parallel_instance)))
    #op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')
    i+=1
    if i%1000 == 0:
        print "{0} documents processed".format(i)
"""
pickle.dump(features,open("WNfeatures-lab.pickle",'wb'))
