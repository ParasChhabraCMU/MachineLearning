import math
from logsum import log_sum
import sys

priors = {}
trans = {}
emits = {}
output = 0

# reading hmm-prior.txt
prior_handle = open(sys.argv[4])
for line in prior_handle:
    splits = line.strip().split(" ")
    priors[splits[0]] = math.log(float(splits[1]))
#print priors


# reading hmm-trans.txt
trans_handle = open(sys.argv[2])
for line_trans in trans_handle:
    inner_dict = {}
    splits = line_trans.strip().split(" ")
    for i in splits[1:]:
        i = i.strip().split(":")
        inner_dict[i[0]] = math.log(float(i[1]))
    trans[splits[0]] = inner_dict
#print trans


# reading hmm-emit.txt
states = trans.keys()

emit_handle = open(sys.argv[3])
for line in emit_handle:
    line = line.strip().split(" ")
    #print line[0]
    inner_temp_dict = {}
    for i in line[1:]:
        j = i.strip().split(":")
        inner_temp_dict[j[0]] = math.log(float(j[1]))

    emits[line[0]] = inner_temp_dict
    inner_temp_dict = {}


# checking the dev file for forward algorithm
dev_handle = open(sys.argv[1])
first_column = True
first_column_dict = {}              # to save first column or initial column
second_column_dict = {}             # to form other columns depending on first or initial column

for line in dev_handle:
    line = line.strip().split(" ")
    for word in line:

        # checking if first column and forming first column
        if first_column:
            for state in states:
                first_column_dict[state] = priors[state] + emits[state][word]
            first_column = False

        # forming other columns over the other states
        else:
            for state in states:        # applying recursive formula
                for previous_state in states:
                    if states.index(previous_state) == 0:
                        second_column_dict[state] = first_column_dict[previous_state] + trans[previous_state][state]
                    else:
                        second_column_dict[state] = log_sum(first_column_dict[previous_state]+ trans[previous_state][state], second_column_dict[state])

                # adding the emission probability of a word given a state
                second_column_dict[state] += emits[state][word]

            first_column_dict = second_column_dict
            second_column_dict = {}

    first_column = True

    for state in states:
        if states.index(state) == 0:
            output = first_column_dict[state]
        else:
            output = log_sum(first_column_dict[state], output)
    print output








