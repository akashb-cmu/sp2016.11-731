# Run
The main Python program to run the alignment is `viterbi.py`. A basic example:
```
python viterbi.py -m TRANSLATION_MODEL_FILE > output.txt
```
`TRANSLATION_MODEL_FILE` is a pickled dictionary matching German vocabulary to English vocabulary with probability weights.

Run `python viterbi.py -h` for more options.

# Algorithms
## Priors
### POS prior
We multiplied our translation table probabilities by a prior distribution based on POS tag matches. This prior probability was a delta functionreturning 1 if the POS tag matched, 0 otherwise. We then weighted this result and added 1.

### Diagonal prior
We calculated the distance in position of the English and German words proportional to the lengths of both sentences and used this as another prior distribution. We weighted this "diagonal prior" and multiplied it by the POS prior and translation probability.

## German compound words
We found a German compound word tokenizer from http://www.danielnaber.de/jwordsplitter/index_en.html and split compound words. This greatly helped us match compound words with their English counterparts.

## Rare tokens
We saw that word types that occurred infrequently commanded high probabilities with words in their aligned sentences and became garbage collectors for any words with moderate translation probabilities to other tokens in the sentence. To counter this, we changed any type that occurred less than a certain number of times in the corpus to a special RARE token. We re-estimated our translation model with these RARE tokens. 

## Jump prior with beam search
To encourage words to align in groups, we added a prior distribution based on the index of the previous German word's alignment. Aligning to that index + 1 was rewarded highly, while distancing from the previous alignment's English index was less and less rewarded. If the previous German word's aligned to null, this term was ignored.

A beam search keeping a certain number of alignment options for each German word allowed us to maximize a word's alignment based on different options for the previous word.

# Original documentation
There are three Python programs here (`-h` for usage):

 - `./align` aligns words using Dice's coefficient.
 - `./check` checks for out-of-bounds alignment points.
 - `./grade` computes alignment error rate.

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./align -t 0.9 -n 1000 | ./check | ./grade -n 5


The `data/` directory contains a fragment of the German/English Europarl corpus.

 - `data/dev-test-train.de-en` is the German/English parallel data to be aligned. The first 150 sentences are for development; the next 150 is a blind set you will be evaluated on; and the remainder of the file is unannotated parallel data.

 - `data/dev.align` contains 150 manual alignments corresponding to the first 150 sentences of the parallel corpus. When you run `./check` these are used to compute the alignment error rate. You may use these in any way you choose. The notation `i-j` means the word at position *i* (0-indexed) in the German sentence is aligned to the word at position *j* in the English sentence; the notation `i?j` means they are "probably" aligned.
