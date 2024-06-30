from abc import ABC, abstractmethod
from random import Random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore

TRAIN_SIZE = 130
TEXT_WIDTH = 6.718
LINE_WIDTH = 3.220


def plot_border(ax) -> None:
    """Setup the plot border with a linewidth of 0.5."""
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)


def all_columns(X: NDArray, _: Random) -> list[int]:
    return list(range(X.shape[1]))


def random_sqrt_columns(X: NDArray, rand: Random) -> list[int]:
    n = np.sqrt(X.shape[1])
    return rand.sample(range(X.shape[1]), int(n))


class TreeNode(ABC):
    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        raise NotImplementedError


class TreeSplit(TreeNode):
    def __init__(
        self,
        feature: int,
        threshold: float,
        left: TreeNode,
        right: TreeNode,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def predict(self, X: NDArray) -> NDArray:
        mask_left = X[:, self.feature] < self.threshold

        predictions = np.zeros(X.shape[0])
        predictions[mask_left] = self.left.predict(X[mask_left])
        predictions[~mask_left] = self.right.predict(X[~mask_left])

        return predictions


class TreeLeaf(TreeNode):
    def __init__(self, prediction: float):
        self.prediction = prediction

    def predict(self, X: NDArray) -> NDArray:
        return np.full(X.shape[0], self.prediction)


class Tree:

    def __init__(
        self,
        rand: Optional[Random] = None,
        get_candidate_columns=all_columns,
        min_samples: int = 2,
    ):
        if rand is None:
            rand = Random()

        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples

    def build(self, X: NDArray, y: NDArray) -> TreeNode:
        """Recursively build a decision tree using the given data. In each
        step the algorithm selects the best feature and threshold to split
        the data into two parts. The process stops when the number of
        samples is less than min_samples or the samples are pure.
        """
        # Stop splitting if there are too few samples
        if len(y) < self.min_samples or np.all(y == y[0]):
            # Get 0 if there are more zeros in y, otherwise 1
            return TreeLeaf(np.mean(y).round().astype(int))

        columns = self.get_candidate_columns(X, self.rand)
        feature, threshold = self.select_split(X, y, columns)

        # Stop building if there is no valid split (constant feature values)
        if feature is None or threshold is None:
            return TreeLeaf(np.mean(y).round().astype(int))

        mask_left = X[:, feature] < threshold
        left = self.build(X[mask_left], y[mask_left])
        right = self.build(X[~mask_left], y[~mask_left])
        return TreeSplit(feature, threshold, left, right)

    def select_split(
        self, X: NDArray, y: NDArray, columns: list[int]
    ) -> tuple[Optional[int], Optional[float]]:
        """Return the best feature and threshold for a given set of columns."""
        # Sort each column of X and compute averages of consecutive values
        column_sorts = np.argsort(X, axis=0)
        X_sorted = X[column_sorts, np.arange(X.shape[1])]
        thresholds = (X_sorted[1:] + X_sorted[:-1]) / 2

        k, t, best = None, None, np.inf
        for c in columns:
            # Feature values are not valid thresholds (would get empty splits)
            _thresholds = thresholds[~np.in1d(thresholds[:, c], X[:, c]), c]
            if len(_thresholds) == 0:
                continue

            # Compute mask for each possible split of column c
            left_mask = X[:, c, np.newaxis] < _thresholds

            # Compute probabilites of class 1 in each left split
            left_n = np.sum(left_mask, axis=0)
            left_counts = np.sum(left_mask * y[:, np.newaxis], axis=0)
            left_probs = left_counts / left_n

            # Compute probabilities of class 1 in the right splits
            right_n = len(y) - left_n
            right_counts = np.sum(y) - left_counts
            right_probs = right_counts / right_n

            # Compute Gini impurity weighted by the number of samples
            left_gini = 2 * left_probs * (1 - left_probs)
            right_gini = 2 * right_probs * (1 - right_probs)
            gini = left_n * left_gini + right_n * right_gini

            gini_argmin = np.argmin(gini)
            if gini[gini_argmin] < best:
                k, t, best = c, _thresholds[gini_argmin], gini[gini_argmin]

        return k, t


class RFModel:

    def __init__(
        self,
        trees: list[TreeNode],
        rand: Random,
        oob_samples: NDArray,
        X: NDArray,
        y: NDArray,
    ):
        self.trees = trees
        self.rand = np.random.default_rng(rand.randint(0, 2**32 - 1))
        self.oob_samples = oob_samples
        self.X = X
        self.y = y

    def predict(self, X: NDArray) -> NDArray:
        # Prediction for each tree is one row
        tree_predictions = np.array([t.predict(X) for t in self.trees])
        # Compute majority using rounding
        return tree_predictions.mean(axis=0).round().astype(int)

    def importance(self) -> NDArray:
        """Return the feature importances for the model. The importance of
        a feature is computed as the average decrease in accuracy for
        out-of-bag samples over all threes when the values of the feature
        are randomly permuted.
        """
        decreases = np.zeros((len(self.trees), self.X.shape[1]))

        for i, tree in enumerate(self.trees):
            # Decrease cannot be computed without out-of-bag samples
            if not np.any(self.oob_samples[i]):
                decreases[i] = np.nan
                continue

            y_oob = self.y[self.oob_samples[i]]
            X_oob = self.X[self.oob_samples[i]]
            acc = np.mean(tree.predict(X_oob) == y_oob)

            for feature in range(self.X.shape[1]):
                # Compute accuracy with permuted feature values
                X_permuted = X_oob.copy()
                self.rand.shuffle(X_permuted[:, feature])
                acc_permuted = np.mean(tree.predict(X_permuted) == y_oob)
                decreases[i, feature] = acc - acc_permuted

        # Compute the average decrease in accuracy for each feature over trees
        return np.nanmean(decreases, axis=0)


class RandomForest:

    def __init__(self, rand: Optional[Random] = None, n: int = 50):
        if rand is None:
            rand = Random()

        self.n = n
        self.rand = rand
        self.rftree = Tree(rand, random_sqrt_columns, 2)

    def build(self, X: NDArray, y: NDArray) -> RFModel:
        """Build a random forest model using the given data. The forest
        consists of n trees, each trained on a bootstrapped sample of the
        data. The number of features used for each split is the square
        root of the number of features in the original dataset. The
        out-of-bag samples (not used in any tree) are stored in the model.
        """
        trees = []
        # Each row is a bootstrapped sample for a tree
        boot_samples = np.array(
            self.rand.choices(range(len(y)), k=len(y) * self.n)
        ).reshape(self.n, -1)

        for tree in range(self.n):
            trees.append(
                self.rftree.build(X[boot_samples[tree]], y[boot_samples[tree]])
            )

        # Each row contains a mask for the out-of-bag samples for a tree
        oob_samples = np.ones((self.n, len(y)), dtype=bool)
        for i, boot in enumerate(boot_samples):
            oob_samples[i, boot] = False

        return RFModel(trees, self.rand, oob_samples, X, y)


def tki() -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray], list[str]]:
    data = pd.read_csv('tki-resistance.csv')
    legend = list(data.columns)

    # Replace string class labels with integers
    data['Class'] = data['Class'].map(dict(zip(np.unique(data['Class']), (0, 1))))

    learn = (
        data.iloc[:TRAIN_SIZE].drop('Class', axis=1).to_numpy(),
        data.iloc[:TRAIN_SIZE]['Class'].to_numpy(),
    )
    test = (
        data.iloc[TRAIN_SIZE:].drop('Class', axis=1).to_numpy(),
        data.iloc[TRAIN_SIZE:]['Class'].to_numpy(),
    )

    return learn, test, legend


def evaluate(
    model: TreeNode | RFModel, data: tuple[NDArray, NDArray]
) -> tuple[float, float]:
    """Helper function to compute misclassification rate and standard error."""
    errors = model.predict(data[0]) != data[1]
    misclassification = np.mean(errors)
    standard_error = np.std(errors) / np.sqrt(len(errors))

    return misclassification, standard_error


def hw_tree_full(
    learn: tuple[NDArray, NDArray], test: tuple[NDArray, NDArray]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Build a decision tree model for the given data and return
    misclassification rates and standard errors for both the training and
    test sets."""
    t = Tree(rand=Random(1))
    model = t.build(*learn)
    return evaluate(model, learn), evaluate(model, test)


def hw_randomforests(
    learn: tuple[NDArray, NDArray], test: tuple[NDArray, NDArray]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Build a random forest model for the given data and return
    missclassification rates and standard errors for both the training and
    test sets."""
    rf = RandomForest(rand=Random(1), n=100)
    model = rf.build(*learn)
    return evaluate(model, learn), evaluate(model, test)


def majority_misclassification(learn: NDArray, test: NDArray) -> float:
    majority = np.mean(learn).round().astype(int)
    return np.mean(test != majority)


def plot_rf_misclassification(
    learn: tuple[NDArray, NDArray], test: tuple[NDArray, NDArray]
) -> None:
    """Plot the misclassification rates for random forests with different
    number of trees."""
    results = pd.DataFrame(columns=['n', 'misclassification', 'se'])
    for n in range(1, 130, 9):
        rf = RandomForest(rand=Random(1), n=n)
        missclass, se = evaluate(rf.build(*learn), test)
        results.loc[n] = [n, missclass * 100, se * 100]

    plt.figure(figsize=(LINE_WIDTH, LINE_WIDTH / 2.2))
    plt.fill_between(
        results['n'],
        results['misclassification'] - results['se'],
        results['misclassification'] + results['se'],
        color='black',
        alpha=0.2,
        linewidth=0.0,
    )
    plt.plot(results['n'], results['misclassification'], color='black')
    plt.xlabel('Number of trees')
    plt.ylabel('1 - accuracy [%]')
    plt.xlim(1, 125)
    plt.ylim(0, 30)
    plot_border(plt.gca())

    plt.savefig('misclassification.pdf', bbox_inches='tight')
    plt.close()


def plot_rf_importance(learn: tuple[NDArray, NDArray]) -> None:
    """Plot variable importance for random forest with n=100 trees and
    variables from the roots of 100 non-random trees."""
    rf = RandomForest(rand=Random(1), n=100)
    model = rf.build(*learn)
    importance = model.importance()
    importance[importance < 0.0005] = 0

    root_nodes = np.zeros_like(importance)
    t = Tree(rand=rf.rand)
    for _ in range(100):
        samples = rf.rand.sample(range(len(learn[0])), TRAIN_SIZE // 5)
        tree = t.build(learn[0][samples], learn[1][samples])
        root_nodes[tree.feature] += 1  # type: ignore

    plt.figure(figsize=(TEXT_WIDTH, LINE_WIDTH / 1.8))

    # Importance
    plt.bar(
        np.arange(len(importance)) + 1,
        importance,
        width=0.35,
        color='black',
        label='Importance',
    )
    plt.xlabel('Feature')
    plt.ylabel('Avg. acc. decrease')
    plt.xlim(1, len(importance))
    plt.ylim(0, 0.044)
    plt.tick_params(axis='x', which='both', bottom=False)
    plot_border(plt.gca())

    # Root nodes
    right_axis = plt.gca().twinx()
    right_axis.bar(  # type: ignore
        np.arange(len(root_nodes)) + 1,
        root_nodes,
        color='black',
        alpha=0.2,
        width=1,
    )
    right_axis.set_ylabel('Root nodes')
    right_axis.set_ylim(bottom=0)
    plot_border(right_axis)

    labels = ['Importance', 'Root nodes']
    handles = [
        plt.Line2D([0], [0], color='black', linewidth=0.35 * 2),
        plt.Line2D([0], [0], color='black', linewidth=1 * 2, alpha=0.2),
    ]
    frame = plt.legend(handles, labels, borderpad=0.2).get_frame()
    frame.set_boxstyle('square')  # type: ignore
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')
    frame.set_alpha(None)

    plt.savefig('importance.pdf', bbox_inches='tight')
    plt.close()


def tree_toy_example() -> None:
    """Toy example to illustrate correct tree building."""

    def f(x: NDArray) -> float:
        a, b = x
        if a < 0.5:
            return 0
        if b > 0.5:
            return 1
        if a > 0.75:
            return 0
        if b < 0.25:
            return 1
        if a < 0.625:
            return 0
        return b < 0.375

    t = Tree(rand=Random(1))
    a, b = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    X = np.stack([a.flatten(), b.flatten()], axis=1)
    y = np.array([f(x) for x in X])
    model = t.build(X, y)

    _, ax = plt.subplots(figsize=(LINE_WIDTH / 2.5, LINE_WIDTH / 2.5))
    ax.set_xticks([0.5, 1])
    ax.set_yticks([0.5, 1])
    ax.text(-0.05, -0.05, '0.0', transform=ax.transAxes, ha='right', va='top')
    patch_args = {'color': 'black', 'alpha': 0.1, 'linewidth': 0.0}
    ax.add_patch(plt.Rectangle((0, 0), 0.5, 1, **patch_args))  # type: ignore
    ax.add_patch(plt.Rectangle((0.75, 0), 0.25, 0.5, **patch_args))  # type: ignore
    ax.add_patch(plt.Rectangle((0.5, 0.25), 0.125, 0.25, **patch_args))  # type: ignore
    ax.add_patch(plt.Rectangle((0.625, 0.375), 0.125, 0.125, **patch_args))  # type: ignore
    plot_tree_splits(model, (0, 1), (0, 1))
    plot_border(ax)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.savefig('toy_tree.pdf', bbox_inches='tight')
    plt.close()


def plot_tree_splits(
    node: TreeNode, x: tuple[float, float], y: tuple[float, float]
) -> None:
    """Plot the decision boundaries of a tree model."""
    if not isinstance(node, TreeSplit):
        return

    if node.feature == 0:
        plt.axvline(node.threshold, *y, color='black', linestyle=(0, (5, 10)))
        plot_tree_splits(node.left, (x[0], node.threshold), y)
        plot_tree_splits(node.right, (node.threshold, x[1]), y)
    else:
        plt.axhline(node.threshold, *x, color='black', linestyle=(0, (5, 10)))
        plot_tree_splits(node.left, x, (y[0], node.threshold))
        plot_tree_splits(node.right, x, (node.threshold, y[1]))


def rf_toy_importance() -> None:
    """Toy example to illustrate correct feature importance computation."""

    def f(x: NDArray) -> float:
        return np.round((x[4] + x[9]) / 2).astype(int)

    rand = Random(1)
    X = np.random.default_rng(rand.randint(0, 2**32 - 1)).random((200, 20))
    y = np.array([f(x) for x in X])
    rf = RandomForest(rand=rand, n=100)
    model = rf.build(X, y)
    importance = model.importance()

    plt.figure(figsize=(LINE_WIDTH, LINE_WIDTH / 2.5))
    plt.plot(np.arange(20) + 1, importance, color='black')
    plt.xlim(1, 20)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plot_border(plt.gca())

    plt.savefig('toy_importance.pdf', bbox_inches='tight')
    plt.close()


def scikit_misclassification(
    learn: tuple[NDArray, NDArray], test: tuple[NDArray, NDArray]
) -> None:
    """Compute the misclassification rates for the decision tree and random
    forest models using the scikit-learn library."""
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(learn[0], learn[1])
    print('scikit tree', evaluate(tree, test))

    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(learn[0], learn[1])
    print('scikit random forest', evaluate(rf, test))


def main() -> None:
    # Plots configuration
    plt.rcParams['lines.linewidth'] = 0.6
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.labelsize'] = 10

    learn, test, _ = tki()
    print('full', hw_tree_full(learn, test))
    print('random forests', hw_randomforests(learn, test))
    print('majority', majority_misclassification(learn[1], test[1]))

    # Plots and testing
    plot_rf_misclassification(learn, test)
    plot_rf_importance(learn)
    tree_toy_example()
    rf_toy_importance()
    scikit_misclassification(learn, test)


if __name__ == '__main__':
    main()
