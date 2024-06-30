from random import Random
from typing import Protocol

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # type: ignore

MBOG_TRAIN = 100
MBOG_TEST = 1000

TEXT_WIDTH = 6.718
LINE_WIDTH = 3.220

SAVE_FIG = False
SHOW_FIG = True


class Classifier(Protocol):
    @property
    def parameters(self) -> NDArray: ...

    def predict(self, X: NDArray) -> NDArray: ...


class Model(Protocol):
    def build(self, X: NDArray, y: NDArray) -> Classifier: ...


def softmax(X: NDArray) -> NDArray:
    """Compute softmax function for each row of a 2D array. The latent
    strengths of the reference class are assumed to be zero.

    Parameters:
        - X: a 2D array of shape (n_samples, n_classes - 1) with latent strengths

    Returns:
        - a 2D array of shape (n_samples, n_classes) with probabilities
    """
    exp = np.exp(np.hstack([np.zeros((X.shape[0], 1)), X]))
    return exp / np.sum(exp, axis=1).reshape(-1, 1)


def log_likelihood(parameters: NDArray, X: NDArray, y: NDArray) -> float:
    """Compute log-likelihood of the multinomial logistic regression model.
    It is a sum of log-probabilities of the correct class for each sample.

    Parameters:
        - parameters: model parameters of shape (n_features * (n_classes - 1),)
        - X: a 2D array of shape (n_samples, n_features) with input data
        - y: a 1D array of shape (n_samples,) with target classes

    Returns:
        - negative log-likelihood (cost) of the model
    """
    n_samples, n_features = X.shape

    # Reshape vector of parameters into a matrix, select probs of correct class
    parameters = parameters.reshape(n_features, -1)
    probabilities = softmax(X @ parameters)[np.arange(n_samples), y]

    return -np.sum(np.log(probabilities)) / n_samples


def log_likelihood_grad(parameters: NDArray, X: NDArray, y: NDArray) -> NDArray:
    """Compute gradient of the log-likelihood of the multinomial logistic
    regression model. Optimisation is more stable with analytical gradient
    than with numerical approximation.

    Parameters:
        - parameters: model parameters of shape (n_features * (n_classes - 1),)
        - X: a 2D array of shape (n_samples, n_features) with input data
        - y: a 1D array of shape (n_samples,) with target classes

    Returns:
        - a 1D array with the gradient of the log-likelihood
    """
    n_samples, n_features = X.shape
    parameters = parameters.reshape(n_features, -1)
    probabilities = softmax(X @ parameters)

    # Create matrix for 1{y^i = j} part of the gradient
    classes = np.zeros_like(probabilities)
    classes[np.arange(n_samples), y] = 1

    # Remove first column because of the reference class
    probabilities, classes = probabilities[:, 1:], classes[:, 1:]

    return -(X.T @ (classes - probabilities)).flatten() / n_samples


def logistic_cdf(x: NDArray) -> NDArray:
    """Compute logistic cumulative distribution function for each element of
    an array."""
    return 1 / (1 + np.exp(-x))


def ordinal_likelihood(parameters: NDArray, X: NDArray, y: NDArray) -> float:
    """Compute log-likelihood of the ordinal logistic regression model.
    It is a sum of log-probabilities of the correct class for each sample.

    Parameters:
        - X: a 2D array of shape (n_samples, n_features) with input data
        - y: a 1D array of shape (n_samples,) with target classes
        - parameters: model parameters of shape (n_features + (n_classes - 2),)

    Returns:
        - negative log-likelihood (cost) of the model
    """
    n_features = X.shape[1]

    # Split parameters into weights and thresholds
    weights = parameters[:n_features]
    splits = np.concatenate(
        # Splits are modeled as deltas
        ([-np.inf, 0], np.cumsum(parameters[n_features:]), [np.inf])
    )

    latent_strengths = X @ weights
    # P(y'_i = j) = F(b_j - u_i) - F(b_j-1 - u_i)
    upper_cdf = logistic_cdf(splits[y + 1] - latent_strengths)
    lower_cdf = logistic_cdf(splits[y] - latent_strengths)
    probabilities = upper_cdf - lower_cdf

    return -np.sum(np.log(probabilities))


class MultinomialClassifier:
    def __init__(self, parameters: NDArray, n_classes: int) -> None:
        """Initialize the multinomial classifier with learned parameters.

        Parameters:
            - parameters: model parameters of shape (n_features * (n_classes - 1),)
        """
        # Reshape vector of parameters into a matrix
        self.parameters = parameters.reshape(-1, n_classes - 1)

    def predict(self, X: NDArray) -> NDArray:
        """Compute probabilities of each class for each sample.

        Parameters:
            - X: a 2D array of shape (n_samples, n_features) with input data

        Returns:
            - a 2D array of shape (n_samples, n_classes) with probabilities
        """
        if len(X.shape) != 2:
            raise ValueError('Input data must be 2D array with n_samples rows')

        return softmax(X @ self.parameters)


class MultinomialLogReg:
    def build(self, X: NDArray, y: NDArray) -> MultinomialClassifier:
        """Fit a multinomial logistic regression model.

        Parameters:
            - X: a 2D array of shape (n_samples, n_features) with input data
            - y: a 1D array of shape (n_samples,) with target classes

        Returns:
            - a fitted MultinomialClassifier instance with predict() method
        """
        n_classes = len(np.unique(y))
        initial_parameters = np.random.rand(X.shape[1] * (n_classes - 1))
        results = minimize(
            log_likelihood,
            initial_parameters,
            jac=log_likelihood_grad,
            args=(X, y),
            method='L-BFGS-B',
            tol=1e-5,
        )

        return MultinomialClassifier(results.x, n_classes)


class OrdinalClassifier:
    def __init__(self, parameters: NDArray, n_features: int) -> None:
        """Initialize the ordinal classifier with learned parameters.

        Parameters:
            - parameters: model parameters of shape (n_features + (n_classes - 2),)
        """
        self.parameters = parameters[:n_features]
        self.splits = np.concatenate(
            ([-np.inf, 0], np.cumsum(parameters[n_features:]), [np.inf])
        )

    def predict(self, X: NDArray) -> NDArray:
        """Compute probabilities of each class for each sample.

        Parameters:
            - X: a 2D array of shape (n_samples, n_features) with input data

        Returns:
            - a 2D array of shape (n_samples, n_classes) with probabilities
        """
        if len(X.shape) != 2:
            raise ValueError('Input data must be 2D array with n_samples rows')

        n_samples = X.shape[0]
        latent_strengths = (X @ self.parameters).reshape(-1, 1)
        splits = np.tile(self.splits, (n_samples, 1)) - latent_strengths
        upper_cdf = logistic_cdf(splits[:, 1:])
        lower_cdf = logistic_cdf(splits[:, :-1])

        return upper_cdf - lower_cdf


class OrdinalLogReg:
    def build(self, X: NDArray, y: NDArray) -> OrdinalClassifier:
        """Fit a ordinal logistic regression model.

        Parameters:
            - X: a 2D array of shape (n_samples, n_features) with input data
            - y: a 1D array of shape (n_samples,) with target classes

        Returns:
            - a fitted OrdinalClassifier instance with predict() method
        """
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        initial_parameters = np.random.rand(n_features + n_classes - 2)
        initial_parameters[n_features:] = 2
        results = minimize(
            ordinal_likelihood,
            initial_parameters,
            args=(X, y),
            method='L-BFGS-B',
            tol=1e-5,
            bounds=Bounds([-np.inf] * n_features + [1e-2] * (n_classes - 2)),
        )

        return OrdinalClassifier(results.x, n_features)


def multinomial_bad_ordinal_good(size: int, rand: Random) -> tuple[NDArray, NDArray]:
    """Generate a synthetic dataset with ordinal target classes.

    Parameters:
        - size: number of samples in the dataset
        - rand: a random number generator

    Returns:
        - X: a 2D array of shape (size, 2) with input data
        - y: a 1D array of shape (size,) with target classes
    """
    np_rand = np.random.default_rng(rand.randint(0, 2**32 - 1))

    X = np.hstack(
        [
            np_rand.normal(0, 1, (size, 1)),
            np_rand.normal(-1, 1, (size, 1)),
            np_rand.normal(1, 1, (size, 1)),
            np_rand.uniform(-1, 1, (size, 1)),
            (np_rand.uniform(0, 1, (size, 1)) * 6).astype(int),
        ]
    )

    error = np_rand.normal(0, 0.2, size)
    weights = np.array([-0.2, 0.6, 0.1, 0.5, 1])
    y = np.clip((X @ weights + error).astype(int), 0, 5)

    return X, y


def get_data(file_name: str, intercept: bool = False):
    """Load and preprocess the dataset used in this assignment. The target
    variable contains ordinal classes, which need to be encoded as numbers.
    It also includes categorical features that need to be one-hot encoded.

    Parameters:
        - file_name: a string with the name of the CSV file
        - intercept: a boolean flag to add a column of ones to the input data

    Returns:
        - X: a 2D array of shape (n_samples, n_features) with input data
        - y: a 1D array of shape (n_samples,) with target classes as numbers
        - feature_names: a list of strings with the names of input features
        - targets: a list of strings with the names of target classes
    """
    data = pd.read_csv(file_name, sep=';')

    # Standardize continuous features
    normalize = ['Angle', 'Distance']
    min, max = data[normalize].min(), data[normalize].max()
    data[normalize] = (data[normalize] - min) / (max - min)

    targets = ['other', 'dunk', 'tip-in', 'layup', 'hook shot', 'above head']
    categorical_features = ['Competition', 'PlayerType', 'Movement']
    ordinal_enc = OrdinalEncoder(categories=[targets], dtype=np.int8)
    onehot_enc = OneHotEncoder(sparse_output=False, drop='first')

    X_onehot = onehot_enc.fit_transform(data[categorical_features])
    X_numeric = data.drop(columns=['ShotType'] + categorical_features)
    features_onehot = onehot_enc.get_feature_names_out().tolist()
    features_numeric = X_numeric.columns.tolist()

    X = np.hstack([X_onehot, X_numeric])
    y = ordinal_enc.fit_transform(data['ShotType'].to_numpy().reshape(-1, 1))

    if intercept:
        X = np.hstack([np.ones((len(X), 1)), X])
        features_onehot = ['intercept'] + features_onehot

    return (X, y.flatten(), features_onehot + features_numeric, targets)


def evaluate(
    X_train: NDArray, y_train: NDArray, X_test: NDArray, y_test: NDArray, model: Model
):
    """Evaluate the model using training and testing data."""
    c = model.build(X_train, y_train)
    probabilities_train = c.predict(X_train)
    probabilities_test = c.predict(X_test)

    correct_train = np.sum(np.argmax(probabilities_train, axis=1) == y_train)
    correct_test = np.sum(np.argmax(probabilities_test, axis=1) == y_test)
    acc_train = correct_train / len(X_train) * 100
    acc_test = correct_test / len(X_test) * 100

    print(f'Train accuracy: {correct_train}/{len(X_train)} ({acc_train:.1f}%)')
    print(f'Test accuracy: {correct_test}/{len(X_test)} ({acc_test:.1f}%)')

    log_train = np.sum(np.log(probabilities_train[np.arange(len(X_train)), y_train]))
    log_test = np.sum(np.log(probabilities_test[np.arange(len(X_test)), y_test]))

    print(f'Train log-score: {log_train:.2f}')
    print(f'Test log-score: {log_test:.2f}')


def feature_importance(
    X: NDArray, y: NDArray, features: list, targets: list, model: Model
):
    """Compute feature importance for the model. Uncertainty of the importance
    is estimated using bootstrapping.
    """
    n_samples, n_features = X.shape
    bootstrap_size = 50
    bootstraps = []

    for _ in range(bootstrap_size):
        indices = np.random.randint(0, n_samples, n_samples)
        bootstraps.append(model.build(X[indices], y[indices]).parameters)

    parameters = np.stack(bootstraps, axis=0)
    mean = np.mean(parameters, axis=0).T
    std = np.std(parameters, axis=0).T

    cmap = mpl.colormaps['tab20']
    colors = [cmap(i) for i in range(n_features)]
    legend_items = [
        Patch(color=colors[i], label=features[i]) for i in range(n_features)
    ]

    plt.figure(figsize=(TEXT_WIDTH, 0.6 * TEXT_WIDTH))
    plt.subplots_adjust(hspace=0.6)
    for i in range(len(targets) - 1):
        plt.subplot(2, 3, i + 1)
        plt.title(targets[i + 1])
        plt.barh(range(n_features), mean[i], color=colors, xerr=std[i])
        plt.yticks([])
        plt.ylim(*plt.ylim()[::-1])
        plt.xlabel('Coefficient value')

    frame = plt.legend(legend_items, features, bbox_to_anchor=(2.1, 1.25)).get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor('black')
    frame.set_boxstyle('square')  # type: ignore

    if SAVE_FIG:
        plt.savefig('feature_importance.pdf', bbox_inches='tight')
    if SHOW_FIG:
        plt.show()
    plt.close()


def main() -> None:
    # Plots configuration
    plt.rcParams['lines.linewidth'] = 0.6
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.labelsize'] = 8

    np.random.seed(1)
    X, y, features, targets = get_data('dataset.csv', intercept=False)
    y_counts = np.bincount(y)
    print(f'Samples: {X.shape[0]}, Features: {X.shape[1]}')
    print('Classes distribution:')
    print(' '.join(targets))
    for i, count in enumerate(y_counts):
        print(str(count).center(len(targets[i]), ' '), end=' ')
    print('\n')

    # Part 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('Multinomial logistic regression:')
    evaluate(X_train, y_train, X_test, y_test, MultinomialLogReg())

    # Part 2
    feature_importance(X, y, features, targets, MultinomialLogReg())

    # Part 3
    rand = Random(1)
    X_train, y_train = multinomial_bad_ordinal_good(MBOG_TRAIN, rand)
    X_test, y_test = multinomial_bad_ordinal_good(MBOG_TEST, rand)

    print('\nMultinomial logistic regression:')
    evaluate(X_train, y_train, X_test, y_test, MultinomialLogReg())

    print('Ordinal logistic regression:')
    evaluate(X_train, y_train, X_test, y_test, OrdinalLogReg())


if __name__ == '__main__':
    main()
