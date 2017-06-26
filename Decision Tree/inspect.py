import sys
import math
import csv
from itertools import islice


def entropy(prob):
    return prob * math.log(float(1)/prob, 2)

#file_input = open("education_test.csv")
file_input = open(sys.argv[1])
#read = csv.reader(file_input)

entropy_value = 0
count_values = {}
total = 0

for l in islice(file_input, 1, None):
    split_line = l.strip().split(',')
    if not count_values.has_key(split_line[len(split_line)-1]):
        count_values[split_line[len(split_line)-1]] = 1
    else:
        count_values[split_line[len(split_line)-1]] += 1

total = sum(count_values.values())

for key in count_values.keys():
    entropy_value += entropy(count_values[key]/float(total))

print "entropy: " + str(entropy_value)
print "error: " + str(min(count_values.values())/float(total))


