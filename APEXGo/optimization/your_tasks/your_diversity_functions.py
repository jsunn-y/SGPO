''' Diversity functions useed for running ROBOT to discover
    a diverse set of optimal solutions with some minimum diversity between eachother
    Functions must take in two input space items (x1, x2)
    and return a float value indicating the diversity between them
'''
import sys 
sys.path.append("../")

from Levenshtein import distance

def dummy_diversity_func(x1, x2):
    return 10000

# Diversity functions with unique string identifiers 
# identifiers can be passed in when running ROBOT to specify which diversity function to use 
DIVERSITY_FUNCTIONS_DICT = {
    'edit_dist':distance,
    'dummy':dummy_diversity_func
}
