from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.tree._tree import TREE_LEAF
import matplotlib.pyplot as plt
import numpy as np
import time

def mainDecisionTree(data, mainTitle):
    # This code was originally taken and modified from https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
    start = time.time()
    mainData = data

    X = mainData['data']
    y = mainData['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    prune_index(classifier.tree_, 0, 5)

    y_pred = classifier.predict(X_test)

    end = time.time()

    print("Decision Tree: " + mainTitle)
    print(classification_report(y_test, y_pred))
    print("Accuracy: ",
          accuracy_score(y_test, y_pred) * 100)
    print("Time: " + str(end-start))

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Decision Tree Learning Curve " + mainTitle
    dcplot = plot_learning_curve(classifier, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    dcplot.savefig("DecisionTreeLearningCurve" + mainTitle + ".png")

def prune_index(inner_tree, index, threshold):
    # This code was originally taken and modified from https://stackoverflow.com/questions/49428469/pruning-decision-trees
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


def mainNeuralNetwork(data, mainTitle):
    # This code was orifinally taken and modified from https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
    start = time.time()
    mainData = data

    X = mainData['data']
    y = mainData['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #mlp = MLPClassifier(hidden_layer_sizes=(X.shape[1], X.shape[1], X.shape[1]))
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        random_state=0)
    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)

    end = time.time()

    print("Neural Network: " + mainTitle)
    print(classification_report(y_test, predictions))
    print("Accuracy: ",
          accuracy_score(y_test, predictions) * 100)
    print("Time: " + str(end - start))

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Neural Network Learning Curve " + mainTitle
    nnplot = plot_learning_curve(mlp, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    nnplot.savefig("NeuralNetworkLearningCurve" + mainTitle + ".png")


def mainDecisionTreeWithBoosting(data, mainTitle):
    # this code was originally taken and modified from https://stackoverflow.com/questions/31231499/pruning-and-boosting-in-decision-trees
    start = time.time()
    mainData = data

    X = mainData['data']
    y = mainData['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # boosting: many many weak classifiers (max_depth=1) refine themselves sequentially
    # tree is the default the base classifier
    estimator = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    end = time.time()

    print("DecisionTree with Boosting: " + mainTitle)
    print(classification_report(y_test, y_pred))
    print("Accuracy: ",
          accuracy_score(y_test, y_pred) * 100)
    print("Time: " + str(end - start))

    title = "Decision Tree with Boosing Learning Curve " + mainTitle
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    dcbplot = plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    dcbplot.savefig("DecisionTreeBoostingLearningCurve" + mainTitle + ".png")

def mainSVM(data, mainTitle):
    # this code was originally taken and modified from https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
    start = time.time()
    mainData = data

    X = mainData['data']
    y = mainData['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    svclassifier = SVC(kernel='linear')
    #svclassifier = SVC(kernel='poly', degree=8, gamma='auto')
    #svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    end = time.time()

    print("SVM: " + mainTitle)
    print(classification_report(y_test, y_pred))
    print("Accuracy: ",
          accuracy_score(y_test, y_pred) * 100)
    print("Time: " + str(end - start))

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "SVM Learning Curve " + mainTitle
    svmplot = plot_learning_curve(svclassifier, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    svmplot.savefig("SVMLearningCurve" + mainTitle + ".png")

def mainKNN(data, mainTitle):
    # this code was originally taken and modified from https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
    start = time.time()
    mainData = data

    X = mainData['data']
    y = mainData['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    end = time.time()

    print("K Nearest Neighbors: " + mainTitle)
    print(classification_report(y_test, y_pred))
    print("Accuracy: ",
          accuracy_score(y_test, y_pred) * 100)
    print("Time: " + str(end - start))

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "KNN Learning Curve " + mainTitle
    knnplot = plot_learning_curve(classifier, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    knnplot.savefig("KNNLearningCurve" + mainTitle + ".png")

    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

    plt.savefig("KNN Error vs Neighbors" + mainTitle + ".png")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # this code was taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Calling main function
if __name__ == "__main__":
    dataArray = [load_breast_cancer(), load_wine()]
    titleArray = ["Breast Cancer", "Wine"]
    for x in range(len(dataArray)):
        mainDecisionTree(dataArray[x], titleArray[x])
        mainNeuralNetwork(dataArray[x], titleArray[x])
        mainDecisionTreeWithBoosting(dataArray[x], titleArray[x])
        mainSVM(dataArray[x], titleArray[x])
        mainKNN(dataArray[x], titleArray[x])