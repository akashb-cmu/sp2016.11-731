import re, string
from collections import Counter
from gensim import corpora, models, similarities

### BEGIN USER PARAMETERS ###
fname = "europarl-v7.de-en.en"

dictfile = "europarl-en.dict"
corpusfile = "europarl-en.mm"
modelname = "europarl-en.lda"

MIN_WORDFREQ = 3 #words with fewer tokens than this are ignored

LDA_NTOPICS = 10
LDA_NITERS = 100
#### END USER PARAMETERS ####

regex = re.compile("(?:{1})|[{0}]".format(re.escape(string.punctuation),
                                          "|".join(["\xc2\xa0","\xc2\xb0","\xc2\xad",
                                                    "\xc2\xa1",
                                                    "\xe2\x80\xa6","\xe2\x80\x93",
                                                    "\xe2\x80\x98","\xe2\x80\x99",
                                                    "\xe2\x80\x9c","\xe2\x80\x9d"])))
nums = re.compile("(?:^|\s+)\d+|\d+(?:\s+|$)")
def preprocess(text):
  return nums.sub(' NUM ',regex.sub(' ',text)).lower().strip()

with open(fname, 'r') as f:
  texts = [[word for word in preprocess(document).split()] for document in f]

  # ignore infrequent words
  counts = Counter([word for text in texts for word in text])
  tokens_ignore = set(word for word in counts if counts[word]<MIN_WORDFREQ)

  dictionary = corpora.Dictionary(texts)
  dictionary.save(dictfile)
  corpus = [dictionary.doc2bow(text) for text in texts]
  corpora.MmCorpus.serialize(corpusfile, corpus)

  lda = models.LdaModel(corpus, num_topics=LDA_NTOPICS, iterations=LDA_NITERS)
  lda.save(modelname)
