import math
import sys

priors = {}
trans = {}
emits = {}


# reading hmm-prior.txt
prior_handle = open(sys.argv[4])
for line in prior_handle:
    splits = line.strip().split(" ")
    priors[splits[0]] = math.log(float(splits[1]))
# print priors

# reading hmm-trans.txt
trans_handle = open(sys.argv[2])
for line_trans in trans_handle:
    inner_dict = {}
    splits = line_trans.strip().split(" ")
    for i in splits[1:]:
        i = i.strip().split(":")
        inner_dict[i[0]] = math.log(float(i[1]))
    trans[splits[0]] = inner_dict
# print trans


# reading hmm-emit.txt
states = trans.keys()

emit_handle = open(sys.argv[3])
for line in emit_handle:
    line = line.strip().split(" ")
    # print line[0]
    inner_temp_dict = {}
    for i in line[1:]:
        j = i.strip().split(":")
        inner_temp_dict[j[0]] = math.log(float(j[1]))

    emits[line[0]] = inner_temp_dict
    inner_temp_dict = {}

# checking the dev file for forward algorithm
dev_handle = open(sys.argv[1])
first_column = True
first_column_dict = {}  # to save first column or initial column
second_column_dict = {}  # to form other columns depending on first or initial column
previous_states = {}
maximum_prob_state_path = {}

for line in dev_handle:
    line = line.strip().split(" ")
    for word in line:

        # checking if first column and forming first column
        if first_column:
            for state in states:
                first_column_dict[state] = priors[state] + emits[state][word]
                previous_states[state] = [state]
            first_column = False
            #print previous_states
        # forming other columns over the other states
        else:
            state_prob = {}
            
            for state in states:  # applying recursive formula
                for previous_state in states:
                    current_value = first_column_dict[previous_state] + trans[previous_state][state] + emits[state][
                        word]
                    state_prob[previous_state] = current_value

                max_key = max(state_prob, key=state_prob.get)
                temp_list = previous_states[max_key]
                maximum_prob_state_path[state] = temp_list + [state]
                second_column_dict[state] = state_prob[max_key]

            first_column_dict = second_column_dict
            second_column_dict = {}
            previous_states = maximum_prob_state_path
            maximum_prob_state_path = {}

    first_column = True

    max_key = max(first_column_dict, key=first_column_dict.get)
    max_path = previous_states[max_key]
    output = ""
    counter = 0
    for word in line:
        output += word + "_" + max_path[counter] + " "
        counter+=1
    print output
