
# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

'''

1) Package/Library Used : sklearn, matplotlib, numpy, graphviz, matplotlib
2) To visuazlize confusion matrix by plotting it - uncomment the code in the def confuse
3) scikit_tree function is used to do part c of assignment. If graphviz gives error while running then add
    graphviz path to os.
'''
import os
from sklearn import tree
import numpy as np
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
#os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    res = {i: [] for i in set(x)}
    for index, i in enumerate(x):
        res[i].append(index)
    return res


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    rep_dict = partition(y)
    H_z = 0
    for k, v in rep_dict.items():
        p = len(v) / len(y)
        H_z += -p * np.log2(p)
    return H_z


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    H_y = entropy(y)
    feature_partition = partition(x)
    H_y_x = 0
    for k, v in feature_partition.items():
        weightage = len(v) / len(y)
        H_y_x += weightage * entropy([y[k] for k in v])
    I = H_y - H_y_x
    return I


def get_majority(y):
    res = partition(y)
    max = 0
    majority = 0
    for k, v in res.items():
        if len(v) > max:
            max = len(v)
            majority = k
    return majority


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    tree = {}
    # entire set of label is pure
    e = partition(y)
    if len(e) == 1:
        return y[0]
    # attribute set is empty
    elif not attribute_value_pairs or depth == max_depth:
        return get_majority(y)
    else:
        # check best information gain among list of features
        ml_info_gain = -1.0
        best_feature, best_value = 0, 0
        pos_subset_indx, neg_subset_indx = [], []
        for i in attribute_value_pairs:
            feature, value = i[0], i[1]
            temp_x = [1 if k == value else 0 for k in x[:, feature]]
            m = mutual_information(temp_x, y)
            if m > ml_info_gain:
                ml_info_gain = m
                best_feature, best_value = feature, value
                pos_subset_indx = np.where(x[:, feature] == value)[0]
                neg_subset_indx = np.where(x[:, feature] != value)[0]

        modified_x_pos = np.take(x, pos_subset_indx, axis=0)
        modified_y_pos = np.take(y, pos_subset_indx)

        modified_x_neg = np.take(x, neg_subset_indx, axis=0)
        modified_y_neg = np.take(y, neg_subset_indx)

        temp_attr_value = (best_feature, best_value)

        av1, av2 = attribute_value_pairs.copy(), attribute_value_pairs.copy()
        av1.remove(temp_attr_value)
        av2.remove(temp_attr_value)
        tree[(best_feature, best_value, True)] = id3(modified_x_pos,
                                                     modified_y_pos,
                                                     av1,
                                                     depth + 1, max_depth)
        tree[(best_feature, best_value, False)] = id3(modified_x_neg,
                                                      modified_y_neg,
                                                      av2,
                                                      depth + 1, max_depth)
        return tree


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if tree == 0 or tree == 1 or tree==2:
        return tree
    attr_pair = list(tree.keys())[0]
    key, val = attr_pair[0], attr_pair[1]
    res = bool(x[key] == val)
    return predict_example(x, tree[key, val, res])


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = np.sum(y_true != y_pred)
    error = (1 / len(y_true)) * error
    return error


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def plot_graph(trn, tst, dataset):
    x = [i for i in range(1, len(trn) + 1)]
    plt.plot(x, trn, "r", label="Train")
    plt.plot(x, tst, "b", label="Test")
    plt.legend()
    plt.savefig(dataset + '.png')
    plt.show()


def confuse(y, y_hat):
    cm = confusion_matrix(y,y_hat)
    # plt.clf()
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    # plt.title('MONKS 1 - Depth = 1')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # s = [['', '',''], ['', '',''],['','','']]
    # for i in range(2):
    #     for j in range(2):
    #         plt.text(j, i, str(s[i][j]) + str(cm[i][j]))
    # plt.show()
    print(cm)

def scikit_tree():
    M = np.genfromtxt('./monk/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    print("Size of training data", Xtrn.shape)
    M = np.genfromtxt('./monk/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    print("Size of testing data", Xtst.shape)
    model = tree.DecisionTreeClassifier()
    model = model.fit(Xtrn, ytrn)
    dot_data = tree.export_graphviz(model, out_file=None, filled=True,
                                    rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("balanc-sci")
    y_hat = model.predict(Xtst)
    cm = confusion_matrix(ytst, y_hat)

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for Monk1, normalize in titles_options:
        disp = plot_confusion_matrix(model, Xtst, ytst,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(Monk1)

        print(Monk1)
        print(disp.confusion_matrix)
    plt.show()

def decisiontree():
    dataset = "monks-1"
    train_file = './monk/monks-1.train'
    test_file = './monk/monks-1.test'
    max_depth = 11

    # Load the train data
    M = np.genfromtxt(train_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt(test_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    trn_err_list, tst_err_list = [], []

    for k in range(1, max_depth):
        attr_value = []
        for i in range(Xtrn.shape[1]):
            attr_value.extend([(i, k) for k in set(Xtrn[:, i])])
        decision_tree = id3(Xtrn, ytrn, attribute_value_pairs=attr_value, max_depth=k)
        visualize(decision_tree)
        # Compute the train error
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred_trn)
        trn_err_list.append(trn_err)

        # Compute the test error
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]

        confuse(ytst, y_pred_tst)     ####To visualize the printed confusion matrix - uncomment the part in function confuse

        tst_err = compute_error(ytst, y_pred_tst)
        tst_err_list.append(tst_err)
    plot_graph(trn_err_list, tst_err_list, dataset)

if __name__ == '__main__':
    decisiontree()
    scikit_tree()
