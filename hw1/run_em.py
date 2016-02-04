from em import EM_model1

INPUT_FILE = "./data/em_test.txt"
OUTPUT_FILE = "./models/translation.model"
MAX_ITERS = 5
IBM_model1 = EM_model1(INPUT_FILE, OUTPUT_FILE, MAX_ITERS)

IBM_model1.estimate_params()

IBM_model1.sanity_check(10) #argument is the number of samples on which to run the check. No arg => full vocabulary

translation_probs = IBM_model1.get_params()