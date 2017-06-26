import sys
import math
import csv
from itertools import islice


# to store all attributes values
attribute_values = {}

# to store current attributes under observation
current_attributes = {}

# to store attribute names
attribute_names = []

positives = ['yes', 'democrat', '+', 'A', 'y', 1, 'before1950', 'morethan3min', 'fast', 'expensive', 'Two', 'large', 'high']
negatives = ['no', 'republican', '-', 'notA', 'n', 0 ,'after1950','lessthan3min','slow', 'cheap', 'MoreThanTwo', 'small','low']
attribute_list = {'Anti_satellite_test_ban': ['y', 'n'], 'Export_south_africa': ['y', 'n'], 'Party': ['democrat', 'republican'], 'year': ['before1950','after1950']
                  ,'solo': ['yes', 'no'],'vocal':['yes', 'no'], 'original': ['yes', 'no'], 'length': ['morethan3min', 'lessthan3min'], 'folk':['yes', 'no'],
                  'tempo': ['fast','slow'],'classical': ['yes', 'no'], 'rhythm': ['yes', 'no'],'jazz': ['yes', 'no'], 'rock': ['yes', 'no'],
                  'buying': ['expensive', 'cheap'], 'maint': ['high','low'], 'doors': ['Two', 'MoreThanTwo'], 'person': ['Two', 'MoreThanTwo'], 'boot': ['large', 'small'],
                  'safety': ['high', 'low'], 'Aid_to_nicaraguan_contras': ['y', 'n'], 'Mx_missile': ['y', 'n'], 'Immigration': ['y', 'n'], 'Superfund_right_to_sue': ['y','n'], 'Duty_free_exports': ['y', 'n'],
                  'M1':['notA', 'A'], 'M2':['notA', 'A'], 'M3':['notA', 'A'], 'M4':['notA', 'A'], 'M5':['notA', 'A'], 'P1':['notA', 'A'], 'P2':['notA', 'A'], 'P3':['notA', 'A'],
                  'P4': ['notA', 'A'], 'F': ['notA', 'A'], 'grade': ['notA', 'A'], 'hit': ['yes', 'no'], 'class':['yes', 'no']}

data_lables = ['Party', 'grade', 'hit', 'class']

# probability function takes a count and total for any attribute values and gives probability
def probability(count, total):
    prob = float(count)/float(total)
    return prob


# entropy takes a list which is part of attribute values like [+,-] values
def entropy(attribute_value):
    total = float(sum(attribute_value))
    entropy_value = 0

    for i in attribute_value:
        try:
            prob = float(probability(i,total))
            if prob != 0:
                entropy_value += prob * math.log(1.0/prob, 2)
        except ZeroDivisionError:
            break
    return entropy_value

# information gain gives us the gain in respect to main entropyS [which is for attribute that is a node]
# for current_attribute which is a dictionary
def information_gain(entropyS, current_attributes):
    current_sum = 0
    for value in current_attributes.values():
        # print value
        current_sum += sum(value)

    for value in current_attributes.values():
        count = sum(value)
        entropyS -= entropy(value) * probability(count, current_sum)

    return entropyS

train_file = open(sys.argv[1])
#test_file = open(sys.argv[2])
train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
#read = csv.reader(file_input)




#train_file = open("politicians_train.csv")


#reading each line by line and storing in a list
split_line = []
for line in train_file:
    temp = line.strip().split(",")
    split_line.append(temp)

# finding attribute names
attribute_names_temp = (split_line.pop(0))
for i in attribute_names_temp:
    attribute_names.append(i.strip())

temp_data = []
track = 0
for i in attribute_names:
    for j in split_line:
        temp_data.append(j[track])

    track += 1
    attribute_values[i.strip()] = temp_data
    temp_data = []

data_labels_data = []
for key in attribute_values.keys():
    if key in data_lables:
        data_labels_data = attribute_values[key]


# now we need to define a class for storing node features and then store the node in a trained tree
# also we need to store a list of current parent to find its child nodes

class Node:
    leaf = 0
    leaf_value = ''
    left_child_label = ''
    right_child_label = ''
    def __init__(self, name, positive, negative, right_child_value, left_child_value, left_child_name, right_child_name):
        self.name = name
        self.positive = positive
        self.negative = negative
        self.right_child_value = right_child_value
        self.left_child_value = left_child_value
        self.left_child_name = left_child_name
        self.right_child_name = right_child_name

    def calculateLeaf(self):
        if self.positive > self.negative:
            self.leaf_value = 'yes'
        else:
            self.leaf_value = 'no'
    def calculateLeftChildLabel(self):
        if self.left_child_value[0] > self.left_child_value[1]:
            self.left_child_label = 'yes'
        else:
            self.left_child_label = 'no'
    def calculateRightChildLabel(self):
        if self.right_child_value[0] > self.right_child_value[1]:
            self.right_child_label = 'yes'
        else:
            self.right_child_label = 'no'

    def isLeaf(self):
        if self.leaf == 1:
            return True
        else:
            return False

    def giveLeafValue(self):
        return self.leaf_value

    def setLeaf(self):
        self.leaf = 1


def findUniqueElementsGivenParent(child_value, child_column_name, parent_value_to_compare, parent_column_name):

    parent_column_values = attribute_values[(parent_column_name).strip()]
    child_column_values = attribute_values[child_column_name.strip()]

    indices = []
    pos = 0
    neg = 0
    result = {}
    index = -1
    while True:
        try:
            index = parent_column_values.index(parent_value_to_compare, index + 1)
            indices.append(index)
        except ValueError:
            break
    for index in indices:

        if child_column_values[index] == child_value:
            if data_labels_data[index] in positives:
                pos += 1
            else:
                neg += 1
    result[child_value] = [pos, neg]
    return result



# defining functions to add left and right branches to our tree

def insertLNode(trained_tree,newLeft):
    left = trained_tree.pop(1)
    if len(left) > 1:
        trained_tree.insert(1,[newLeft,left,[]])
    else:
        trained_tree.insert(1,[newLeft, [], []])
    return trained_tree

def insertRNode(trained_tree,newRight):
    right = trained_tree.pop(2)
    if len(right) > 1:
        trained_tree.insert(2,[newRight,[],right])
    else:
        trained_tree.insert(2,[newRight,[],[]])
    return trained_tree


# finding the parent node using inital total entropy
def findRootNodeValues(attribute_name):
    type_values = {}
    pos = 0
    neg = 0
    indices = []
    index = -1

    attribute_value = attribute_list[(attribute_name).strip()]
    values = attribute_values[(attribute_name).strip()]

    for j in attribute_value:
        while True:
            try:
                index = values.index(j, index+1)
                indices.append(index)
            except ValueError:
                break

        for k in indices:
            if data_labels_data[k] in positives:
                pos += 1
            else:
                neg += 1
        indices = []
        index = -1
        type_values[j] = [pos, neg]
        pos = 0
        neg = 0


    return type_values


temp_total_yes = 0
temp_total_no = 0
for i in (attribute_list[(attribute_names[len(attribute_names)-1]).strip()]):
    #print i
    a = ((attribute_values[(attribute_names[len(attribute_names) - 1]).strip()]).count(i))
    if i in positives:
        temp_total_yes = a
    else:
        temp_total_no = a

total_yes_no = [temp_total_yes, temp_total_no]
initial_total_entropy = entropy(total_yes_no)

attribute_names.pop(len(attribute_names) - 1)

max_entropy = 0
a = 0
name = ''
for i in attribute_names:
    if max_entropy == 0:
        max_entropy = information_gain(initial_total_entropy, findRootNodeValues(i))
    else:
        max_entropy = max(max_entropy, information_gain(initial_total_entropy, findRootNodeValues(i)))

for i in attribute_names:
    if max_entropy == information_gain(initial_total_entropy, findRootNodeValues(i)):
        a = [i, findRootNodeValues(i)]
        attribute_names.pop(attribute_names.index(i))
        name = i

positive = 0
negative = 0
right_child_value = (a[1].values())[1]
left_child_value = (a[1].values())[0]
right_child_name = (a[1].keys())[0]
left_child_name = (a[1].keys())[1]


parent_node = Node(name, positive, negative, right_child_value, left_child_value, right_child_name, left_child_name)


trained_tree = [parent_node, [], []]
current_parent_node = parent_node


def findLeftChild(parent_node_left_child_value):
    main_entropy = entropy(parent_node_left_child_value)
    max_gain = 0.000000000000

    for i in attribute_names:
        all_attribute_values = 0
        given_attribute_unique_values = {}

        for k in attribute_list[i.strip()]:
            l = findUniqueElementsGivenParent(k, i, parent_node.left_child_name, parent_node.name)
            given_attribute_unique_values.update(l)

        info_gain = information_gain(main_entropy, given_attribute_unique_values)


        if info_gain > 0.100000000000 and info_gain > max_gain:
            max_gain = info_gain
            max_gain_attributes = [i, given_attribute_unique_values, max_gain]



    if max_gain == 0.000000000000:
        name = i.strip()
        positive = 0
        negative = 0
        right_child_value = 0
        left_child_value = 0
        right_child_name = 0
        left_child_name = 0
        left_child_Node1 = Node(name, positive, negative, right_child_value, left_child_value, right_child_name, left_child_name)
        insertLNode(trained_tree, left_child_Node1)
        left_child_Node1.setLeaf()

    else:
        name = max_gain_attributes[0]
        positive = 0
        negative = 0
        right_child_value = (max_gain_attributes[1].values())[1]
        left_child_value = (max_gain_attributes[1].values())[0]
        right_child_name = (max_gain_attributes[1].keys())[0]
        left_child_name = (max_gain_attributes[1].keys())[1]
        left_child_Node1 = Node(name, positive, negative, right_child_value, left_child_value, right_child_name, left_child_name)
        insertLNode(trained_tree, left_child_Node1)
        left_child_Node1.calculateRightChildLabel()
        left_child_Node1.calculateLeftChildLabel()



def findRightChild(parent_node_right_child_value):
    main_entropy = entropy(parent_node_right_child_value)
    max_gain = 0.000000000000

    for i in attribute_names:

        given_attribute_unique_values = {}
        for k in attribute_list[i.strip()]:
            l = findUniqueElementsGivenParent(k, i.strip(), parent_node.right_child_name, parent_node.name)

            given_attribute_unique_values.update(l)
            info_gain = information_gain(main_entropy, given_attribute_unique_values)

        if info_gain > 0.100000000000 and info_gain > max_gain:
            max_gain = info_gain
            max_gain_attributes = [i.strip(), given_attribute_unique_values, max_gain]


    if max_gain == 0.000000000000:
        name = i.strip()
        positive = 0
        negative = 0
        right_child_value = 0
        left_child_value = 0
        right_child_name = 0
        left_child_name = 0
        right_child_Node1 = Node(name, positive, negative, right_child_value, left_child_value, right_child_name,
                           left_child_name)
        insertRNode(trained_tree, right_child_Node1)
        right_child_Node1.setLeaf()

    else:
        name = max_gain_attributes[0]
        positive = 0
        negative = 0
        right_child_value = (max_gain_attributes[1].values())[1]
        left_child_value = (max_gain_attributes[1].values())[0]
        right_child_name = (max_gain_attributes[1].keys())[0]
        left_child_name = (max_gain_attributes[1].keys())[1]
        right_child_Node1 = Node(name, positive, negative, right_child_value, left_child_value, right_child_name, left_child_name)
        insertRNode(trained_tree, right_child_Node1)
        right_child_Node1.calculateRightChildLabel()
        right_child_Node1.calculateLeftChildLabel()

def getLChild():
    return trained_tree[1]


def getRChild():
    return trained_tree[2]

def rootNode():
    return trained_tree[0]


findLeftChild(current_parent_node.left_child_value)
findRightChild(current_parent_node.right_child_value)


if (getLChild()[0].isLeaf()):
    getLChild()[0].positive = current_parent_node.left_child_value[0]
    getLChild()[0].negative = current_parent_node.left_child_value[1]
    getLChild()[0].calculateLeaf()



if getRChild()[0].isLeaf():
    getRChild()[0].positive = current_parent_node.right_child_value[0]
    getRChild()[0].negative = current_parent_node.right_child_value[1]
    getRChild()[0].calculateLeaf()

if getLChild()[0].isLeaf() and getRChild()[0].isLeaf():
    current_parent_node.calculateLeftChildLabel()
    current_parent_node.calculateRightChildLabel()


print "[" + str(total_yes_no[0]) + "+/" + str(total_yes_no[1]) + "-"+ "]"
print current_parent_node.name + " = " + current_parent_node.left_child_name+":"  + " [" + str(current_parent_node.left_child_value[0]) + "+/" + str(current_parent_node.left_child_value[1]) + "-" + "]"

if not (getLChild()[0].isLeaf()):
    print "| " + (getLChild()[0]).name + " = " + (getLChild()[0]).left_child_name+":" + " [" + str((getLChild()[0]).left_child_value[0]) + "+/" + str((getLChild()[0]).left_child_value[1]) + "-" + "]"
    print "| " + (getLChild()[0]).name + " = " + (getLChild()[0]).right_child_name+":" + " [" + str((getLChild()[0]).right_child_value[0]) + "+/" + str((getLChild()[0]).right_child_value[1]) + "-" + "]"
print current_parent_node.name + " = " + current_parent_node.right_child_name+":"  + " [" + str(current_parent_node.right_child_value[0]) + "+/" + str(current_parent_node.right_child_value[1]) + "-" + "]"
if not getRChild()[0].isLeaf():
    print "| " + (getRChild()[0]).name + " = " + (getRChild()[0]).left_child_name+":" + " [" + str((getRChild()[0]).left_child_value[0]) + "+/" + str((getRChild()[0]).left_child_value[1]) + "-" + "]"
    print "| " + (getRChild()[0]).name + " = " + (getRChild()[0]).right_child_name+":" + " [" + str((getRChild()[0]).right_child_value[0]) + "+/" + str((getRChild()[0]).right_child_value[1]) + "-" + "]"
if len(current_parent_node.left_child_label) > 0  and len(current_parent_node.right_child_label) > 0:
    print "[" + str(total_yes_no[0]) + "+/" + str(total_yes_no[1]) + "-"+ "]"



def train_tree_error(file_name):
    #file_input = open(file_name)
    file_input = open(file_name,"r")
    first = 0
    attributes_value_dict = {}
    correct = 0
    wrong = 0

    for line in file_input:
        l = line.strip().split(',')
        if first == 0:
            attributes = l
            first += 1
        else:
            c = 0
            label = l[len(l) - 1]
            for a in attributes:
                attributes_value_dict[a.strip()] = l[c]
                c += 1
            y = attributes_value_dict[current_parent_node.name]
            if y in positives:
                current_child_node = getLChild()[0]
                if current_child_node.isLeaf():
                    if current_child_node.giveLeafValue() in positives and label in positives:
                        correct += 1
                    elif current_child_node.giveLeafValue() in negatives and label in negatives:
                        correct += 1
                    else:
                        wrong += 1

                else:
                    current_child_node = getLChild()[0]
                    current_child_node_name = current_child_node.name
                    next_value = attributes_value_dict[current_child_node_name]

                    for key in attributes_value_dict:
                        if key == current_child_node_name:
                            val = attributes_value_dict[current_child_node_name]

                    if next_value in positives:
                        if (current_child_node.left_child_label in positives and label in positives) or (
                                        current_child_node.left_child_label in negatives and label in negatives):
                            correct += 1
                            # print 'write'

                        else:
                            wrong += 1
                            # print 'wrong'
                    else:
                        if (current_child_node.right_child_label in positives and label in positives) or (
                                        current_child_node.right_child_label in negatives and label in negatives):
                            correct += 1
                            # print 'write'
                        else:
                            wrong += 1
                            # print 'wrong'

            else:
                current_child_node1 = getRChild()[0]
                if current_child_node1.isLeaf():
                    if current_child_node1.giveLeafValue() in negatives and label in negatives:
                        correct += 1
                    elif current_child_node1.giveLeafValue() in positives and label in positives:
                        correct += 1
                    else:
                        wrong += 1

                else:
                    current_child_node1 = getRChild()[0]
                    current_child1_node_name = current_child_node1.name
                    next_value = attributes_value_dict[current_child1_node_name]
                    for key in attributes_value_dict:
                        if key == current_child1_node_name:
                            val = attributes_value_dict[current_child1_node_name]

                    if next_value in positives:
                        if (current_child_node1.left_child_label in positives and label in positives) or (
                                        current_child_node1.left_child_label in negatives and label in negatives):
                            correct += 1
                        else:
                            wrong += 1
                    else:
                        if (current_child_node1.right_child_label in positives and label in positives) or (
                                        current_child_node1.right_child_label in negatives and label in negatives):
                            correct += 1
                        else:
                            wrong += 1

    sum = wrong+correct
    error = round(float(wrong)/sum, 12)
    return error

print "error(train): " + str((train_tree_error(train_file_name)))
print "wrong(test): " + str((train_tree_error(test_file_name)))

