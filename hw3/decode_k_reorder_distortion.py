#!/usr/bin/env python
import argparse
import sys
import models
import heapq
from collections import namedtuple
import scipy.stats
import math


parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-s', '--stack-size', dest='s', default=100, type=int, help='Maximum stack size (default=1)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=True,  help='Verbose mode (default=off)')
parser.add_argument('-w', '--window', dest='window', default=7,  help='window size for skipping')
parser.add_argument('-var', '--variance', dest='variance', default=1.0,  help='window size for skipping')

opts = parser.parse_args()

window = int(opts.window)
variance = float(opts.variance)

distortion_model = scipy.stats.norm(0, variance)

tm = models.TM(opts.tm, sys.maxint)
lm = models.LM(opts.lm)
sys.stderr.write('Decoding %s...\n' % (opts.input,))
input_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

hypothesis = namedtuple('hypothesis', 'logprob, lm_state, predecessor, phrase, src_align, trans_len')

def get_distort_val(src_phr_pos, trg_phr_pos):
    return(distortion_model.pdf(abs(src_phr_pos - trg_phr_pos)))


def generate_hypothesis(h, phrase, k, is_end, src_phrase_start):
    # get kth predecessor of h. If you hit None, construct a new hypothesis with phrase followed by relevant previous phrases
    phraseStack = list()
    hpred = h
    for _ in xrange(k):
        if hpred.predecessor == None:
            break
        phraseStack.append((hpred.phrase, h.src_align))
        hpred = hpred.predecessor
    phraseStack.append((phrase, src_phrase_start))

    logprob = hpred.logprob
    lm_state = hpred.lm_state
    last_target_pos = hpred.trans_len

    while(phraseStack):
        (phr, phr_src_align) = phraseStack.pop()
        logprob += phr.logprob + get_distort_val(phr_src_align, last_target_pos + 1)
        last_target_pos += len(phr)
        for word in phr.english.split():
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
            if is_end and not phraseStack:
                logprob += lm.end(lm_state)
        hpred = hypothesis(logprob, lm_state, hpred, phr, phr_src_align, last_target_pos)
    return hpred

for f in input_sents:
    # The following code implements a DP monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of
    # the first i words of the input sentence.
    # HINT: Generalize this so that stacks[i] contains translations
    # of any i words (remember to keep track of which words those
    # are, and to estimate future costs)
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, None, 0)

    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        # extend the top s hypotheses in the current stack
        for h in heapq.nlargest(opts.s, stack.itervalues(), key=lambda h: h.logprob): # prune
            for j in xrange(i+1,len(f)+1):
                if f[i:j] in tm:
                    for phrase in tm[f[i:j]]:
                        is_end = (j == len(f))
                        for k in list(xrange(0,window+1)):
                            hyp = generate_hypothesis(h, phrase, k, is_end, i)
                            lm_state = hyp.lm_state
                            logprob = hyp.logprob
                            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
                                stacks[j][lm_state] = hyp



                                # find best translation by looking at the best scoring hypothesis
    # on the last stack
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    def extract_english_recursive(h):
        return '' if h.predecessor is None else '%s%s ' % (extract_english_recursive(h.predecessor), h.phrase.english)
    print extract_english_recursive(winner)

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write('LM = %f, TM = %f, Total = %f\n' %
                         (winner.logprob - tm_logprob, tm_logprob, winner.logprob))