from collections import defaultdict, Counter
from itertools import permutations, tee
import numpy as np
import random


# To generate the first set of 6 permutations of conditions
# perms_set_1 = None

# To generate the second set of 6 permutations of conditions, ensure it is different than the first set already obtained by the below code
perms_set_1 = np.array([['b' 'a' 'c' 'b' 'a' 'c' 'a' 'b' 'c' 'b'],  # User study session #6
                        ['c' 'b' 'c' 'b' 'a' 'c' 'a' 'b' 'a' 'c'],  # User study session #5
                        ['c' 'a' 'b' 'c' 'a' 'b' 'a' 'c' 'b' 'c'],  # User study session #4
                        ['a' 'b' 'a' 'b' 'c' 'b' 'c' 'a' 'c' 'a'],  # User study session #3
                        ['b' 'a' 'b' 'a' 'c' 'a' 'c' 'b' 'c' 'b'],  # User study session #2
                        ['a' 'b' 'c' 'b' 'a' 'c' 'a' 'b' 'c' 'a']]) # User study session #1


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def all_perms(lst):
    transition_types = ['ab', 'ac', 'bc', 'ba', 'ca', 'cb']
    transition_tie_breakers = ['ab', 'ac', 'bc']

    # Filter out all permutations that don't have exactly one of each transition type and exactly one tie breaker
    valid_perms = []
    for perm in set(permutations(lst)):

        # Count all types of transitions, e.g. ('a', 'b'), in the permutation
        transition_counts = defaultdict(int)
        for transition in pairwise(''.join(perm)):
            transition_counts[''.join(transition)] += 1

        # Check if the permutation has exactly one of each transition type and exactly one tie breaker
        is_valid_perm = True
        for ttb in transition_tie_breakers:
            t_revt_pair = transition_counts[ttb], transition_counts[ttb[::-1]] 
            if t_revt_pair != (2, 1) and t_revt_pair != (1, 2):
                is_valid_perm = False
                break

        if is_valid_perm:
            valid_perms.append(perm)

    return valid_perms


if __name__ == '__main__':
    perms_a = all_perms(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'])
    perms_b = all_perms(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'b'])
    perms_c = all_perms(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c'])
    
    while True:
        # Sample two permutations from each type (=> 6 sessions)
        a_sample = random.sample(perms_a, 2)
        b_sample = random.sample(perms_b, 2)
        c_sample = random.sample(perms_c, 2)

        samples = np.array(a_sample + b_sample + c_sample)
        np.random.shuffle(samples)

        # Count all types of transitions, e.g. ('a', 'b'), in the permutation
        all_transitions = [ x for sample in samples for x in list(pairwise(''.join(sample)))]
        transition_counts = defaultdict(int)
        for transition in all_transitions:
            transition_counts[''.join(transition)] += 1

        # Check if number of ties is balanced overall
        for t_count in transition_counts.values():
            if t_count != 9:
                continue

        # Ensure the second set of 6 permutations is different from the first one
        if (samples == perms_set_1).all():
            continue

        # Otherwise, valid permutation was found
        break

    print(samples,'\n')
    for sample in samples:
        print(sorted(Counter(''.join(transition) for transition in list(pairwise(''.join(sample)))).items(), key=lambda pair: pair[0]))
    print()

    # Pretty print for the study design table in Google Sheet
    # In Google Sheet, paste the csv strings and click Data -> Split text to columns
    for sample in samples:
        print(','.join(sample))
    

# First set of 6 permutations
# [['b' 'a' 'c' 'b' 'a' 'c' 'a' 'b' 'c' 'b']  # User study session #6
#  ['c' 'b' 'c' 'b' 'a' 'c' 'a' 'b' 'a' 'c']  # User study session #5
#  ['c' 'a' 'b' 'c' 'a' 'b' 'a' 'c' 'b' 'c']  # User study session #4
#  ['a' 'b' 'a' 'b' 'c' 'b' 'c' 'a' 'c' 'a']  # User study session #3
#  ['b' 'a' 'b' 'a' 'c' 'a' 'c' 'b' 'c' 'b']  # User study session #2
#  ['a' 'b' 'c' 'b' 'a' 'c' 'a' 'b' 'c' 'a']] # User study session #1


# Second set of 6 permutations
# [['a' 'b' 'c' 'b' 'a' 'c' 'a' 'c' 'b' 'a']  # User study session #7
#  ['a' 'c' 'a' 'c' 'b' 'c' 'b' 'a' 'b' 'a']  # User study session #8
#  ['c' 'b' 'a' 'b' 'c' 'a' 'c' 'a' 'b' 'c']  # User study session #9
#  ['b' 'c' 'a' 'b' 'c' 'b' 'a' 'c' 'a' 'b']  # User study session #10
#  ['c' 'a' 'b' 'a' 'c' 'b' 'c' 'a' 'b' 'c']  # User study session #11
#  ['b' 'c' 'a' 'b' 'a' 'c' 'b' 'a' 'c' 'b']] # User study session #12

# [('ab', 1), ('ac', 2), ('ba', 2), ('bc', 1), ('ca', 1), ('cb', 2)]
# [('ab', 1), ('ac', 2), ('ba', 2), ('bc', 1), ('ca', 1), ('cb', 2)]
# [('ab', 2), ('ac', 1), ('ba', 1), ('bc', 2), ('ca', 2), ('cb', 1)]
# [('ab', 2), ('ac', 1), ('ba', 1), ('bc', 2), ('ca', 2), ('cb', 1)]
# [('ab', 2), ('ac', 1), ('ba', 1), ('bc', 2), ('ca', 2), ('cb', 1)]
# [('ab', 1), ('ac', 2), ('ba', 2), ('bc', 1), ('ca', 1), ('cb', 2)]

# a,b,c,b,a,c,a,c,b,a
# a,c,a,c,b,c,b,a,b,a
# c,b,a,b,c,a,c,a,b,c
# b,c,a,b,c,b,a,c,a,b
# c,a,b,a,c,b,c,a,b,c
# b,c,a,b,a,c,b,a,c,b
