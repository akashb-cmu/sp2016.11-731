There are 4 Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder (Baseline/provided decoder)
 - './decode_k_reorder_distortion.py' (Best model)
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade or python decode_k_reorder_distortion.py | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model
 
 
 Info about the best model:
 
 Our best model achieves a model score of ~ -4964. It does so by generating new candidate translation phrases and generating up to k hypothesis involving placing this hypothesis at any position up to k positions from the last position in the target translation. In addition to the language model score and translation model score, we include a distortion model that captures the intuition that phrases at similar relative positions in source and target sentences should be aligned. The distortion distribution is a 0-mean, unit variance normal distribution over the absolute difference of the start positions of the source and translated phrases.

