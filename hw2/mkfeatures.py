import re, string, pdb
from utils import get_hypotheses
from gensim import corpora, models, similarities
import pickle, numpy

### BEGIN USER PARAMETERS ###
fname = "europarl-v7.de-en.en"

FEATFILE = "ldafeatures.pickle"

LDA_NTOPICS = 50
LDA_NITERS = 100

MIN_WORDFREQ = 3 #words with fewer tokens than this are ignored

dictfile = "europarl-en.dict"
corpusfile = "europarl-en.mm"
modelname = "europarl-en-{0}.lda".format(LDA_NTOPICS)

HYP_FILE = "./data/train-test.hyp1-hyp2-ref"
GOLD_FILE = "./data/train.gold"
ALPHA = .001 # minimum margin
#### END USER PARAMETERS ####

features = ([],[])

regex = re.compile("(?:{1})|[{0}]".format(re.escape(string.punctuation),
                                          "|".join(["\xc2\xa0","\xc2\xb0","\xc2\xad",
                                                    "\xc2\xa1",
                                                    "\xe2\x80\xa6","\xe2\x80\x93",
                                                    "\xe2\x80\x98","\xe2\x80\x99",
                                                    "\xe2\x80\x9c","\xe2\x80\x9d"])))
nums = re.compile("(?:^|\s+)\d+|\d+(?:\s+|$)")

def preprocess(text):
    return nums.sub(' NUM ',regex.sub(' ',text)).lower().strip()

dictionary = corpora.Dictionary.load(dictfile)
lda = models.LdaModel.load(modelname)

def get_lda_cosine(hyp, ref):
    return sum([hyp.get(i,0)*ref.get(i,0) for i in range(LDA_NTOPICS)])

def get_lda_weights(text):
    return dict(lda[dictionary.doc2bow(preprocess(text).split())])

def get_prediction(parallel_instance):
    hyp1 = get_lda_weights(parallel_instance[0])
    hyp2 = get_lda_weights(parallel_instance[1])
    ref = get_lda_weights(parallel_instance[2])
    h1_sim = get_lda_cosine(hyp1,ref)
    h2_sim = get_lda_cosine(hyp2,ref)

    if h1_sim > 0 and h1_sim > h2_sim + ALPHA:
        return(-1)
    elif h2_sim > 0 and h2_sim > h1_sim + ALPHA:
        return(1)
    else:
        return(0)

[labeled_instances, unlabeled_instances] = get_hypotheses(HYP_FILE, GOLD_FILE)

for (i,instance) in enumerate(labeled_instances):
    features[0].append(numpy.array([]))
for (i,instance) in enumerate(unlabeled_instances):
    features[1].append(numpy.array([]))

pickle.dump(features,open(FEATFILE,'wb'))
