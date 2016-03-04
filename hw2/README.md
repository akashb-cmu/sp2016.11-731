11-731: Machine Translation
Akash Bharadwaj and Keith Maki
Homework 2: Automatic MT Evaluation
March 4th, 2016

We have implemented several techniques for this assignment, which are combined using SVM.

The main SVM code, which also implements basic METEOR score computation, is in baseline.py
To include different feature sets, the variable FEATS can be changed to include features indexing the following approaches:

Precision and Recall features ('prec','rec') - These features encode the precision and recall of each hypothesis when compared to the reference translation.  These features do not improve on METEOR

METEOR ('whm') - This approach computes the weighted harmonic mean of precision and recall.  We find this to be a good foundation on which to build other approaches.  The optimal weighting parameter was determined experimentally, we use 0.45

Text length features ('len_char','len_word') - These features describe the length of each text in characters and words, respectively, including raw lengths and relative lengths as a proportion of the reference length.  Surprisingly, these features are very informative, and while they may seem liable to overfit, they also perform well in cross-validation.

Language model featuers ('lm_score') - These features include the probabilities of the hypotheses and reference texts under trigram and five-gram backoff language models.  We find that these features do not improve our results.

WordNet synset features ('wn_syns') - These features expand the basic precision and recall features to include WordNet synset matches between each hypothesis and the reference.  Unfortunately, these features also do not improve performance over METEOR+Character based features.

LDA cosine similarity features ('lda') - These features encode the cosine similarity between LDA topic vectors for the hypotheses and reference translations.  LDA was trained on the Europarl corpus using 5, 10, 20, and 50 latent topics.  Once again, we find these features give conflicting information with the METEOR+Character based features, and do not appear to correlate well with human judgements for this task.

##### Submitted Code #####

In addition to baseline.py, we submit code in the form of several python scripts, as follows:

wn.py, which computes WordNet synset precision and recall features.

trainLDA.py, which trains LDA topics using gensim.

LDAcosine.py, which computes LDA cosine similarity features.

###### Provided Code ######

There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.
