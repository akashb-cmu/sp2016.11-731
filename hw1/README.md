# Run
The main Python program to run the alignment is `viterbi.py`. A basic example:
```
python viterbi.py -m TRANSLATION_MODEL_FILE > output.txt
```
`TRANSLATION_MODEL_FILE` is a pickled dictionary matching German vocabulary to English vocabulary with probability weights.

Run `python viterbi.py -h` for more options.

How to replicate best results:
Generate model 1 parameters:

In run_em.py:

Set the name of your model file using the variable OUTPUT_FILE. eg. OUTPUT_FILE = "./models/english_to_german.model"
Set the direction of parameters.
Eg.

Decomp_model = EM_DE_Compound(INPUT_FILE, OUTPUT_FILE, MAX_ITERS)
Decomp_model.estimate_params(EM_model1.ENGLISH_TO_GERMAN, 2)

Now execute:
python run_em.py

To get alignments:

python viterbi.py -n 150 -m english_to_german.model > output.txt;


# Algorithms
### EM
This forms the base of the model and gets us down to an AER score ~0.42. Words are stemmed using the nltk snowball stemmer to reduce the vocabulary size and ensure we don't estimate different parameters for different inflectional/equivalent forms of the same word. To go beyond this on the strength of parameters alone, we replace rare words (which occur < 50 times in the dataset) with a rare token. This helps us avoid a version of the label bias problem, where rare words co-occur with very few other words and the overall conditional probability mass is concentrated much more with these, as compared to more prevalent words where the probability mass is more spread out. This also helps with garbage collection of certain such rare tokens. Rare tokens account for < 0.1% of tokens and 10-20% of types. We have also successfully tried de-compounding words in german. In our experience, these restrictions on the vocabularies greatly effect the number and quality of parameters estimated by EM. While we didn't individually study the effect of this in isolation, it does behave as a force multiplier behind our other priors. 

Note: We also tried to use bi-directional lexical translation parameters. While this gets us a 0.015 AER improvement, it was largely redundant with our POS prior (which improved results more).

### Beam search
This is an improvement to the decoding procedure. At each step of decoding, we retain the the top n most probable alignments so far. Given the diagonal prior we incorporate, this allows us to avoid overly conservative alignments centered around the diagonal and alignments such as piece-wise diagonal alignments in consideration. This decoding procedure gives us ~0.03 improvement in AER.


## Priors
### POS prior
We multiplied our translation table probabilities by a prior distribution based on POS tag matches. This prior probability was a delta functionreturning 1 if the POS tag matched, 0 otherwise. We then weighted this result and added 1. This gives us a 0.02 AER improvement.

### Diagonal prior
We calculated the distance in position of the English and German words proportional to the lengths of both sentences and used this as another prior distribution. We weighted this "diagonal prior" and multiplied it by the POS prior and translation probability. This gives use a nearly 0.08 AER improvement.

## German compound words
We found a German compound word tokenizer from http://www.danielnaber.de/jwordsplitter/index_en.html and split compound words. This greatly helped us match compound words with their English counterparts.

## Rare tokens
We saw that word types that occurred infrequently commanded high probabilities with words in their aligned sentences and became garbage collectors for any words with moderate translation probabilities to other tokens in the sentence. To counter this, we changed any type that occurred less than a certain number of times in the corpus to a special RARE token. We re-estimated our translation model with these RARE tokens.

## Jump prior with beam search
To encourage words to align in groups, we added a prior distribution based on the index of the previous German word's alignment. Aligning to that index + 1 was rewarded highly, while distancing from the previous alignment's English index was less and less rewarded. If the previous German word's aligned to null, this term was ignored.

A beam search keeping a certain number of alignment options for each German word allowed us to maximize a word's alignment based on different options for the previous word.

### Orthographic similarity
This looks for an exact match between the source token and target token (or the first k characters of both tokens). This can help us identify transliterations, named entities etc. that are spelled the same across both languages. This surprisingly gives us a 0.02 AER improvement even with beam search and the other informative priors

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
