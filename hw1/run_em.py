from em import *
import pdb
INPUT_FILE = "./data/em_test.txt"
OUTPUT_FILE = "./models/german_to_english.model"
MAX_ITERS = 2
IBM_model1 = EM_model1(INPUT_FILE, OUTPUT_FILE, MAX_ITERS)

#IBM_model1.estimate_params(EM_model1.GERMAN_TO_ENGLISH, 2)
#IBM_model1.estimate_params(EM_model1.ENGLISH_TO_GERMAN, 2)

#IBM_model1.sanity_check(EM_model1.GERMAN_TO_ENGLISH, 22) #argument is the number of samples on which to run the check. No arg => full vocabulary
#IBM_model1.sanity_check(EM_model1.ENGLISH_TO_GERMAN, 27) #argument is the number of samples on which to run the check. No arg => full vocabulary

#IBM_model1.sanity_check(IBM_model1.GERMAN_TO_ENGLISH, 22)
#IBM_model1.sanity_check(IBM_model1.ENGLISH_TO_GERMAN, 27)

#Model that decomposes

Decomp_model = EM_DE_Compound(INPUT_FILE, OUTPUT_FILE, MAX_ITERS)
Decomp_model.estimate_params(EM_model1.GERMAN_TO_ENGLISH, 2)
#print(Decomp_model.rare_tokens)
print("No. of rare tokens combined:")
print("German:" + str(len(Decomp_model.rare_tokens[0])))
print("English:" + str(len(Decomp_model.rare_tokens[1])))

#Decomp_model.estimate_params(EM_model1.ENGLISH_TO_GERMAN, 2)

#Decomp_model.sanity_check(EM_model1.GERMAN_TO_ENGLISH, 22)
#Decomp_model.sanity_check(EM_model1.ENGLISH_TO_GERMAN, 22)
