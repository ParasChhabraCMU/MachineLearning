import numpy as np
import math
import random
import sys

# sigmoid function
def sigmoid(x):
    return 1/(1+float(math.exp(-x)))

# number of iterations to stop the code
number_of_iterations = 0

# step for learning
learning_constant = 1.8


# opening and reading a csv
#file_open_handle = open("education_train.csv")
file_open_handle = open(sys.argv[1])
file_second_handle = open(sys.argv[2])

# storing attributes names and values in list structure
first_line = 0
attributes_list = []
attributes_values = []

for line in file_open_handle:
    if first_line == 0:
        attributes_list = line.strip().split(',')
        first_line += 1
    else:
        attributes_values.append(line.strip().split(','))


# to store labels of training data
labels = []

# pre-processing the attributes values
for i in attributes_values:

    a1 = (float(i.pop(0)))/100
    i.insert(0, a1)

    a2 = (float(i.pop(1))/100)
    i.insert(1, a2)


    a3 = (float(i.pop(2)) / 100)
    i.insert(2, a3)

    a4 = (float(i.pop(3)) / 100)
    i.insert(3, a4)


    a5 = (float(i.pop(4)) / 100)
    i.insert(4, a5)

    labels.append(float(i.pop(len(i)-1))/100)

    a0 = float(1)
    i.insert(0, a0)

# assigning random weights w0, w1, w2, w3
input_hiddenlayer_weights = np.array(np.random.random((6,6)) - 0.5)

hiddenlayer_output_weights = []
for ran in range(0,6):
    hiddenlayer_output_weights.append((random.uniform(-0.5, 0.5)))

previous_error = 0.0
while number_of_iterations < 900:
    for dataindex in range(0, len(attributes_values)):

        a = attributes_values[dataindex]
        label = labels[dataindex]

        #finding hidden node layer weights
        hidden_node1 = (np.dot(input_hiddenlayer_weights[:, 0], a))
        hidden_node2 = (np.dot(input_hiddenlayer_weights[:, 1], a))
        hidden_node3 = (np.dot(input_hiddenlayer_weights[:, 2], a))
        hidden_node4 = (np.dot(input_hiddenlayer_weights[:, 3], a))
        hidden_node5 = (np.dot(input_hiddenlayer_weights[:, 4], a))
        hidden_node6 = (np.dot(input_hiddenlayer_weights[:, 5], a))

        intermediate_hiddennode_array = np.array([hidden_node1, hidden_node2, hidden_node3, hidden_node4, hidden_node5, hidden_node6])

        sigmoid_hidden_node = []

        for i in intermediate_hiddennode_array:
            sigmoid_hidden_node.append(sigmoid(i))

        sigmoid_hidden_node = (np.array(sigmoid_hidden_node)).reshape(6, 1)

        output = sigmoid((np.dot(hiddenlayer_output_weights, sigmoid_hidden_node)))

        delta_output_sum = output * (1 - output) * (label - output)

        k = 0
        hidden_delta = []
        for i in sigmoid_hidden_node:
            hidden_delta.append((float(i) * (1 - float(i)) * hiddenlayer_output_weights[k] * delta_output_sum))
            k += 1


        new_hidden_output_weigths = []
        for l in range(0, len(hiddenlayer_output_weights)):
            v = hiddenlayer_output_weights[l] + learning_constant * delta_output_sum * sigmoid_hidden_node[l][0]
            new_hidden_output_weigths.append(v)

        temp1 = []
        for l in range(0, len(a)):
            v = learning_constant * hidden_delta[0] * a[l] + (input_hiddenlayer_weights[l][0])
            temp1.append(v)

        temp2 = []
        for l in range(0, len(a)):
            v = learning_constant * hidden_delta[1] * a[l] + (input_hiddenlayer_weights[l][1])
            temp2.append(v)

        temp3 = []
        for l in range(0, len(a)):
            # input_hiddenlayer_weights[l] += (learning_constant * hidden_delta[l]) * a
            v = learning_constant * hidden_delta[2] * a[l] + (input_hiddenlayer_weights[l][2])
            temp3.append(v)

        temp4 = []
        for l in range(0, len(a)):
            # input_hiddenlayer_weights[l] += (learning_constant * hidden_delta[l]) * a
            v = learning_constant * hidden_delta[3] * a[l] + (input_hiddenlayer_weights[l][3])
            temp4.append(v)

        temp5 = []
        for l in range(0, len(a)):
            v = learning_constant * hidden_delta[4] * a[l] + (input_hiddenlayer_weights[l][4])
            temp5.append(v)

        temp6 = []
        for l in range(0, len(a)):
            v = learning_constant * hidden_delta[5] * a[l] + (input_hiddenlayer_weights[l][5])
            temp6.append(v)

        new_input_hidden_weigths = np.column_stack((temp1, temp2, temp3, temp4, temp5, temp6))

        input_hiddenlayer_weights = new_input_hidden_weigths
        hiddenlayer_output_weights = new_hidden_output_weigths

    error = 0.0

    for dataindex in range(0, len(attributes_values)):

        a = attributes_values[dataindex]

        label = labels[dataindex]

        #finding hidden node layer weights
        hidden_node1 = (np.dot(input_hiddenlayer_weights[:, 0], a))
        hidden_node2 = (np.dot(input_hiddenlayer_weights[:, 1], a))
        hidden_node3 = (np.dot(input_hiddenlayer_weights[:, 2], a))
        hidden_node4 = (np.dot(input_hiddenlayer_weights[:, 3], a))
        hidden_node5 = (np.dot(input_hiddenlayer_weights[:, 4], a))
        hidden_node6 = (np.dot(input_hiddenlayer_weights[:, 5], a))

        intermediate_hiddennode_array = np.array([hidden_node1, hidden_node2, hidden_node3, hidden_node4, hidden_node5, hidden_node6])
        sigmoid_hidden_node = []

        for i in intermediate_hiddennode_array:
            sigmoid_hidden_node.append(sigmoid(i))

        sigmoid_hidden_node = (np.array(sigmoid_hidden_node)).reshape(6, 1)

        output = sigmoid((np.dot(hiddenlayer_output_weights, sigmoid_hidden_node)))

        error += math.pow(((label - output)*100), 2)

    print error

    number_of_iterations += 1

    if error > previous_error and number_of_iterations > 1:
        print "error"
        break

    previous_error = error

print "TRAINING COMPLETED! NOW PREDICTING."


#file_second_handle = open("education_dev.csv")

# storing attributes names and values in list structure
first_line = 0
predict_attributes_list = []
predict_attributes_values = []

for line1 in file_second_handle:
    if first_line == 0:
        predict_attributes_list = line1.strip().split(',')
        first_line += 1
    else:
        predict_attributes_values.append(line1.strip().split(','))

# to store labels of training data
predict_labels = []

# pre-processing the attributes values
for i in predict_attributes_values:

    a1 = (float(i.pop(0)))/100
    i.insert(0, a1)

    a2 = (float(i.pop(1))/100)
    i.insert(1, a2)


    a3 = (float(i.pop(2)) / 100)
    i.insert(2, a3)

    a4 = (float(i.pop(3)) / 100)
    i.insert(3, a4)


    a5 = (float(i.pop(4)) / 100)
    i.insert(4, a5)

    labels.append(float(i.pop(len(i)-1))/100)

    a0 = float(1)
    i.insert(0, a0)


for dataindex in range(0, len(predict_attributes_values)):

    a = predict_attributes_values[dataindex]
    # print a

    label = labels[dataindex]
    # print label

    # finding hidden node layer weights
    hidden_node1 = (np.dot(input_hiddenlayer_weights[:, 0], a))
    hidden_node2 = (np.dot(input_hiddenlayer_weights[:, 1], a))
    hidden_node3 = (np.dot(input_hiddenlayer_weights[:, 2], a))
    hidden_node4 = (np.dot(input_hiddenlayer_weights[:, 3], a))
    hidden_node5 = (np.dot(input_hiddenlayer_weights[:, 4], a))
    hidden_node6 = (np.dot(input_hiddenlayer_weights[:, 5], a))

    intermediate_hiddennode_array = np.array(
        [hidden_node1, hidden_node2, hidden_node3, hidden_node4, hidden_node5, hidden_node6])


    sigmoid_hidden_node = []

    for i in intermediate_hiddennode_array:
        sigmoid_hidden_node.append(sigmoid(i))

    sigmoid_hidden_node = (np.array(sigmoid_hidden_node)).reshape(6, 1)

    preidcted_output = sigmoid((np.dot(hiddenlayer_output_weights, sigmoid_hidden_node)))

    print preidcted_output*100

