from itertools import izip

__author__ = 'vlad'
#!/usr/bin/python
import os
import numpy as np

from threading import Thread, Lock
from random import randrange
from copy import deepcopy

from numpy import array, arange
from numpy import random, dot


# user imports
from read_mnist import read_images

# comment this to disable debug printing
DEBUG = False
INFO = True

errors_net = {}

print_lock = Lock()
errors_lock = Lock()


'''
    Feed forward neural network with one output that's using backpropagation of error
    to adjust weights. Weights are adjusted using stochastic gradient descent.
'''

class BackPropagationNeuralNetwork(Thread):

    OUTPUT_NODE_COUNT = 10
    #BATCH_SIZE = 60000

    def __init__(self, t_id, no_iterations, eta, train_set, valid_set, hidden_layer_count,
                 hidden_layers_nodes_count, stop_at_convergence=False):
        Thread.__init__(self)
        self.eta, self.id = eta, t_id
        self.train_set, self.valid_set = deepcopy(train_set), deepcopy(valid_set)
        self.train_features, self.train_targets = self.train_set[0], self.train_set[1]
        self.valid_features, self.valid_targets = self.valid_set[0], self.valid_set[1]
        self.batch_size = self.train_features.shape[0]
        print_info("Batch size ", self.batch_size)

        self.hidden_layer_count = hidden_layer_count
        self.total_layers = hidden_layer_count + 1
        self.stop_at_convergence = stop_at_convergence

        self.node_count = [self.get_input_node_count()]
        self.node_count.extend(hidden_layers_nodes_count)
        self.node_count.append(self.OUTPUT_NODE_COUNT)
        self.total_iterations = no_iterations

        self.training_errors, self.validation_errors = [], []
        self.iterations_convergence = 0
        self.weights = self.init_weights()
        print_debug(self.node_count)

    def get_input_node_count(self):
        row = self.train_features[0]
        return len(row)

    def run(self):
        self.display_info()
        # self.normalized_values = self.normalize_inputs()

        print "Thread id = {0}\n{1}".format(self.id, self.train_features[0])
        self.iterations_convergence = self.total_iterations

        for i in xrange(self.total_iterations):
            print "Iteration " + str(i)
            training_errors = self.do_epoch(i)
            print_info(self.id, " Computing mispredictions")
            predictions = 0
            idx = 0
            for example, target in izip(self.train_features, self.train_targets):
              prediction = self.predict(example)
              #print_info("Prediction, Actual ", prediction, target)
              if prediction == target:
                predictions += 1
              idx += 1
              if idx >= self.batch_size:
                break

            print_info(self.id, " Prediction count ", predictions, float(predictions)/self.batch_size)

        self.test_errors = self.compute_testing_error()
        print_info("Thread id {0} Normalized Test Set error (absolute, relative): {1}".format(self.id,
                                                                                              self.test_errors))

        self.write_errors()

    def display_info(self):
        print_lock.acquire()
        print "Starting Thread for Neural Network with id ", self.id
        print "Hidden Layers, Neuron counts/Layer ", self.hidden_layer_count, self.node_count
        print "Total itartions count ", self.total_iterations
        print_lock.release()

    def do_epoch(self, iteration_no):
      print_info(self.id, " Epoch ", iteration_no)
      ridx_list = xrange(self.batch_size) #rnd.sample(range(len(self.train_features)), self.batch_size)
      count = 0
      for ridx in ridx_list:
        print_debug("Optimizing for example ", ridx)
        if count % 5000 == 0:
          print_info(self.id, " Example ", count)

      # for feature_v, target in izip(self.train_features, self.train_targets):
        outputs = self.feed_forward(self.train_features[ridx])
        print_debug("Outputs ", outputs[-1])
        print_debug("Targets ", self.train_targets[ridx])

        errors = self.propagate_errors_back(outputs, self.train_targets[ridx])
        print_debug("Weights ", len(self.weights))
        print_debug("Output ", outputs)
        print_debug("Errors sdasdas", errors)
        self.update_weights(outputs, errors)
        count += 1

    def compute_validation_error(self, iteration_no):
        return self.compute_error(self.valid_features, self.valid_targets, iteration_no, "Iteration, validation error (abs, rel): ")

    def compute_testing_error(self):
        return self.compute_error(self.test_set, self.total_iterations, "Test set error (abs, rel): ")

    def feed_forward(self, attribute):
      # print_debug(inputs)
      output = attribute
      outputs = []
      for layer_weights in self.weights:
       # output = np.insert(output, 0, 1)
        outputs.append(output)
        print_debug("Layer weights", layer_weights.shape)
        layer_inputs = self.compute_inputs(output, layer_weights)
        output = [sigmoid(neuron_input) for neuron_input in layer_inputs]
        print_debug("output sss ", len(output))

      outputs.append(output)

      print_debug("All outputs ", outputs)
      print_debug("Final output ", output)
      #outputs[-1] = self.normalize_vector(outputs[-1])
      print_debug("Outputs ", outputs[-1])
      return outputs

    def predict(self, attribute):
      outputs = self.feed_forward(attribute)
      print_debug("Ooutputs are ", outputs[-1])
      return np.argmax(outputs[-1])

    def compute_inputs(self, output, layer_weights):
      print_debug("Output eraeraeS: ", output)
      print_debug("first node incident weights: ", layer_weights[0, :])
      print_debug("First node dot product", dot(output, layer_weights[0, :]))
      linear_combinations = [dot(output, w_ij) for w_ij in layer_weights]
      print_debug("All dot products ", linear_combinations)
      return linear_combinations

    def propagate_errors_back(self, outputs, target):
      deltas = []

      # compute the delta for the output units
      output = outputs[-1]
      target_v = np.zeros(10)
      target_v[target] = 1
      print_debug("Target ", target, target_v)
      print_debug("Computing delta for output units")
      print_debug("Output layer output ", output)
      delta = []
      for o_i in xrange(self.node_count[-1]):
        derivative = output[o_i] * (1 - output[o_i])
        delta.append(derivative * (target_v[o_i] - output[o_i]))
        print_debug("Output layer delta ", delta)
        deltas.append(delta)

      # compute the error term for the hidden units
      print_debug("Computing error terms for the hidden units")

      # for all hidden outputs in reverse order, but not for first output
      layers = range(self.hidden_layer_count, -1, -1)
      print_debug("Layers list ", layers)
      print_debug("Outputs ", outputs[-2:0:-1])
      for l_idx, l_output in zip(layers, outputs[-2:0:-1]):
          delta = []
          for node_idx, node_output in enumerate(l_output):
              derivative = node_output * (1 - node_output)
              delta.append(derivative * self.compute_weighted_delta(l_idx, node_idx, deltas[-1]))
          deltas.append(delta)

      deltas.reverse()
      return deltas

    def normalize_vector(self, vector):
      #print_debug("Vector ", vector)
      vector = np.array(vector)
      #print_debug("Vector ", vector)
      return (vector / sum(vector)).tolist()

    def update_weights(self, outputs, errors):
        for layer in xrange(len(self.weights)):
            l_weights = self.weights[layer]
            print_debug("Layer weights ", l_weights.shape, len(l_weights), len(l_weights[0]))
            for i_w in xrange(len(l_weights)):
                for j_w in xrange(len(l_weights[i_w])):
                    delta_w = errors[layer][i_w] * outputs[layer][j_w]
                    l_weights[i_w][j_w] += self.eta * delta_w

    def compute_weighted_delta(self, layer_index, node_j, delta_i):
        print_debug("Computing weighted delta for layer, node, delta_i ", layer_index, node_j, delta_i)
        l_weights = self.weights[layer_index]
        print_debug("Next layer weights ", l_weights)
        w_delta = dot(l_weights[:, node_j], delta_i)
        print_debug("Weighted delta ", w_delta)
        return w_delta

    """
    This method initializes weights between nodes.
    weights is a list of 3 dimensions W[L][i][j]:
    L - the current layer
    i -  the node from the current layer (incident neuron)
    j - the node of  the previous layer (emergent neuron)

    extra: added bias nodes with no incident weights


    W[L][i][j] - the weight from jth node on layer L-1 to ith node on layer L.
    """
    def init_weights(self):
        weights = []
        for level in xrange(self.total_layers):
            weights_l = random.random((self.node_count[level + 1], self.node_count[level])) - 0.5
            print_debug("layer ", level, self.node_count[level + 1], self.node_count[level])
            weights.append(weights_l)

        for w in weights:
            print_info(w.shape)

        return array(weights)

    def write_errors(self):
        errors_lock.acquire()
        errors_net[self.id] = [self.hidden_layer_count, self.node_count,
                           self.training_errors, self.validation_errors,
                           self.test_errors, self.iterations_convergence]
        errors_lock.release()


def print_debug(*args):
    if DEBUG:
        print_lock.acquire()
        for arg in args:
            print arg,
        print '\n',
        print_lock.release()


def print_info(*args):
    if INFO:
        print_lock.acquire()
        for arg in args:
            print arg,
        print '\n',
        print_lock.release()


def sigmoid(x):
  if x > 100: return 1.0
  if x< -100: return 0.0
  return 1 / (1 + np.exp(-x))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def replace_string_attrbutes(attribute_values):
    string_attrs =[]
    for idx, attr in enumerate(attribute_values[0]):
        if not is_number(attr):
            string_attrs.append(idx)

    print_debug("String attr idxs: ", string_attrs)
    for idx in string_attrs:
        col = [el[idx] for el in attribute_values]
        print col
        col_set = set(col)
        counts = len(col_set)
        interval = 1.0 / (counts - 1)
        intervals = arange(0.0, 1.0 + interval, interval)

        print col_set, len(col_set)
        print interval, intervals
        str_to_no = {}
        for i, el in zip(intervals,col_set):
            str_to_no[el] = i
        print str_to_no
        col_no = [str_to_no[el] for el in col]
        print col_no
        for i, row in enumerate(attribute_values):
            row[idx] = col_no[i]
    return attribute_values


# def plot_errors(directory, thread_id, training_errors, validation_errors):
#     plt.title("Evolutia erorii absolute pt reteaua {0}".format(thread_id))
#     plt.plot(range(len(training_errors)), [er[0] for er in training_errors], label="Training error (absolute)")
#     plt.plot(range(len(validation_errors)), [er[0] for er in validation_errors],label="Validation error (absolute)")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
#     plt.autoscale()
#     plt.savefig("{0}/plot_{1}_absolute".format(directory, thread_id))
#     plt.clf()
#
#     plt.title("Evolutia erorii relative pt reteaua {0}".format(thread_id))
#     plt.plot(range(len(training_errors)), [er[1] for er in training_errors], label="Training error (relative)")
#     plt.plot(range(len(validation_errors)), [er[1] for er in validation_errors], label="Validation error (relative)")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
#     plt.autoscale()
#     plt.savefig("{0}/plot_{1}_relative".format(directory, thread_id))
#     plt.clf()



# def read_data(filename):
#
#     #csv_reader = CsvDataReader(filename)
#     # attributes = csv_reader.read()
#
#     # attribute_values = attributes[1:]
#     # print_info(attribute_values)
#     # attribute_values = replace_string_attrbutes(attribute_values)
#
#     # print_info(attribute_values)
#
#     attribute_values = array([[float(el) for el in row] for row in attribute_values])
#     print_debug(attribute_values)
#
#     return attribute_values


def test_for_multiple_layers_variable_neurons():

    # attribute_values = read_data("BodyFat.csv")

    nets = []
    max_hidden_layers = 3
    test_multiple_layers_multiple_neurons(0, max_hidden_layers, [], 0, nets)

    print_info("Nets are: length, list: ", len(nets), nets)

    #return

    threads = []
    for id, net in enumerate(nets):
        net_depth = len(net)
        # neural_net = BackPropagationNeuralNetwork(id, 200, 0.3, attribute_values, hidden_layer_count=net_depth,
        #                                           hidden_layers_nodes_count=net, stop_at_convergence=True)
        # threads.append(neural_net)

    print_info("Total threads to start ", len(threads))

    offset = 5
    for index in xrange(0, len(threads), offset):
        print_info("Starting threads from {0} to {1}".format(index, index + offset - 1))
        for thread in threads[index: min(index + offset, len(threads))]:
            thread.start()
        for t in threads[index: min(index + offset, len(threads))]:
            t.join()

    print_info("All nets finished. Doing plots")
    data = []
    for thread_id in errors_net:

        # errors[self.id] = [self.hidden_layer_count, self.node_count,
        #                    self.training_errors, self.validation_errors,
        #                    self.test_errors, self.iterations_convergence]

        print_info("Plotting for thread id ", thread_id)
        net_errors = errors_net[thread_id]
        hidden_layer_count, node_count = net_errors[0], net_errors[1]
        training_errors, validation_errors = net_errors[2], net_errors[3]
        test_errors, iterations_convergence = net_errors[4], net_errors[5]
        data.append([hidden_layer_count, sum(node_count[1:]), iterations_convergence])
        print_info("Net {0} converged after {1} iters.".format(thread_id, iterations_convergence))

    # csv_writer = CsvDataWriter("data_{0}_hidden_layers_{1}_neurons.csv".format(max_hidden_layers, "20"))
    # csv_writer.write(data)

    print_info(data)


def test_multiple_layers_multiple_neurons(current_layer, total_layers, nodes_list, neurons_count, nets):
    if current_layer == total_layers + 1:
        return
    if len(nodes_list) <= current_layer:
        nodes_list.append(0)

    for node_count in xrange(5, 12):
        if node_count + neurons_count < 20:
            nodes_list[current_layer] = node_count
            print_debug(nodes_list)
            nets.append(nodes_list[:])
            test_multiple_layers_multiple_neurons(current_layer + 1, total_layers, nodes_list,
                                                  neurons_count + node_count, nets)

    nodes_list.pop()


def test_few_nets():

    filename = "BodyFat.csv"
    directory = "/".join([filename.split(".")[0], "plots"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # attribute_values = read_data(filename)

    # net1 = BackPropagationNeuralNetwork(1, 200, 0.3, attribute_values, hidden_layer_count=1,
    #                                     hidden_layers_nodes_count=[6])
    # net2 = BackPropagationNeuralNetwork(2, 200, 0.3, attribute_values, hidden_layer_count=2,
    #                                     hidden_layers_nodes_count=[6, 7])
    #
    # net3 = BackPropagationNeuralNetwork(3, 200, 0.6, attribute_values, hidden_layer_count=1,
    #                                     hidden_layers_nodes_count=[10])

    # threads = [net1 ] #, net2, net3]

    # net1.start()
    #net2.start()
    #net3.start()

    # for t in threads:
    #     t.join()

    print_info("All nets finished. Doing plots")

    for thread_id in errors_net:

        # errors[self.id] = [self.hidden_layer_count, self.node_count,
        #                    self.training_errors, self.validation_errors,
        #                    self.test_errors, self.iterations_convergence]

        print_info("Plotting for thread id ", thread_id)
        net_errors = errors_net[thread_id]
        training_errors, validation_errors = net_errors[2], net_errors[3]
        test_errors, iterations_convergence = net_errors[4], net_errors[5]

        print_info("Net {0} converged after {1} iters.".format(thread_id, iterations_convergence))
        # plot_errors(directory, thread_id, training_errors, validation_errors)


if __name__ == "__main__":
  [train_set, valid_set, test_set] = read_images('/home/vlad/Documents/datasets/mnist.pkl.gz')
  net1 = BackPropagationNeuralNetwork(1, 200, 0.3, train_set, valid_set, hidden_layer_count=1,
                                      hidden_layers_nodes_count=[6])
  net2 = BackPropagationNeuralNetwork(2, 200, 0.3, train_set, valid_set, hidden_layer_count=1,
                                      hidden_layers_nodes_count=[60])

  net1.start()
 # net2.start()
  net1.join()
  net2.join()
    # net3.start()

    # for t in threads:
    #     t.join()

  print train_set[0].shape

   #test_few_nets()

    # test_for_multiple_layers_variable_neurons()

    # filename = "MPG.csv"
    # directory = "/".join([filename.split(".")[0], "plots"])
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # csv_reader = CsvDataReader(filename)
    # attributes = csv_reader.read()
    #
    # attribute_values = attributes[1:]
    # print_info(attribute_values)
    # attribute_values = replace_string_attrbutes(attribute_values)
    #
    # print_info(attribute_values)
    #
    # attribute_values = array([[float(el) for el in row] for row in attribute_values])
    # print_debug(attribute_values)
    #
    # net1 = BackPropagationNeuralNetwork(1, 200, 0.3, attribute_values, hidden_layer_count=1,
    #                                     hidden_layers_nodes_count=[6])
    # net2 = BackPropagationNeuralNetwork(2, 200, 0.3, attribute_values, hidden_layer_count=2,
    #                                     hidden_layers_nodes_count=[6, 7])
    #
    # net3 = BackPropagationNeuralNetwork(3, 200, 0.6, attribute_values, hidden_layer_count=1,
    #                                     hidden_layers_nodes_count=[10])
    #
    # threads = [net1, net2, net3]
    #
    # net1.start()
    # net2.start()
    # net3.start()

    # for t in threads:
    #     t.join()
