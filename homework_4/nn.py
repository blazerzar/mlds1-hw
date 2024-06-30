from abc import ABC, abstractmethod
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray
from scipy.optimize import fmin_l_bfgs_b  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from tqdm import tqdm  # type: ignore

LINE_WIDTH = 3.220
SHOW_PLOTS = True
SAVE_PLOTS = False
EVAL_NN = False
GEN_PREDICTIONS = True


def mse_loss(y_pred: NDArray, y_true: NDArray) -> float:
    return np.mean((y_pred - y_true) ** 2) / 2


def log_loss(y_pred: NDArray, y_true: NDArray) -> float:
    return -np.mean(np.log(y_pred[y_true == 1]))


class Layer(ABC):
    train: bool = False

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        """Forward pass of the layer.

        Parameters:
            - x : input to the layer

        Returns:
            - output of the layer
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, running_grad: NDArray) -> tuple[NDArray, NDArray]:
        """Compute the gradient of the loss with respect to the current layer
        using the formula:

            dJ/dw = dz/dw * dJ/dz,

        where w is the current layer and z the next output z^(i). Since the
        activation layers are not trained, we do not compute the gradient with
        respect to the current layer, but only with respect to the input.

        Parameters:
            - running_grad: gradient of the loss with respect to the output

        Returns:
            - gradient of the loss with respect to the current layer
            - gradient of the loss with respect to the input z^(i-1) or a^(i-1)
        """
        raise NotImplementedError

    def num_params(self) -> int:
        """Return the number of parameters in the layer."""
        return 0


class Activation(Layer):
    @abstractmethod
    def gradient(self) -> NDArray:
        """Compute the gradient of the activation function.

        Returns:
            - gradient of the activation function with same shape as input
        """
        raise NotImplementedError

    def backward(self, running_grad: NDArray) -> tuple[NDArray, NDArray]:
        # Gradient of the current layer is computed in the next weights
        # layer which updates the running_grad ahead.
        return np.array([]), running_grad * self.gradient()


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        # We add bias using 1s in the last column
        self.input_dim = input_dim + 1
        self.output_dim = output_dim
        self.train = True

        rand = np.random.default_rng(1)
        std = np.sqrt(2 / input_dim)
        self.weights = rand.normal(0, std, (input_dim + 1, output_dim))
        # Initialise biases to zero
        self.weights[-1, :] = 0

    def forward(self, x: NDArray) -> NDArray:
        self.input = np.hstack([x, np.ones((x.shape[0], 1))])
        return self.input @ self.weights

    def backward(self, running_grad: NDArray) -> tuple[NDArray, NDArray]:
        return self.input.T @ running_grad, running_grad @ self.weights[:-1, :].T

    def num_params(self) -> int:
        return self.input_dim * self.output_dim

    def load_weights(self, weights: NDArray) -> None:
        # Last row contains the biases
        self.weights = weights.reshape(self.input_dim, self.output_dim)

    def get_weights_mask(self) -> NDArray:
        num_weights = (self.input_dim - 1) * self.output_dim
        num_biases = self.output_dim
        return np.repeat([1, 0], [num_weights, num_biases])


class Sigmoid(Activation):
    def forward(self, x: NDArray) -> NDArray:
        x = np.clip(x, -50, 50)
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def gradient(self) -> NDArray:
        return self.output * (1 - self.output)


class ReLU(Activation):
    def forward(self, x: NDArray) -> NDArray:
        self.output = np.maximum(0, x)
        return self.output

    def gradient(self) -> NDArray:
        return np.where(self.output > 0, 1, 0)


class Softmax(Activation):
    def forward(self, x: NDArray) -> NDArray:
        exps = np.exp(np.clip(x, -50, 50))
        output = exps / np.sum(exps, axis=1).reshape(-1, 1)
        self.output_shape = output.shape
        return output

    def gradient(self) -> NDArray:
        # We do not use this because we only need the gradient of the last
        # output layer, which simplifies. Just a placeholder for convinience.
        return np.ones(self.output_shape)


class Identity(Activation):
    """Used as a placeholder in the list of layers because we skip the last
    activation layer (softmax) in classification."""

    def forward(self, x: NDArray) -> NDArray:
        self.output_shape = x.shape
        return x

    def gradient(self) -> NDArray:
        return np.ones(self.output_shape)


class ANN:
    def __init__(self, layers: list[Layer]):
        """Create a new neural network model with the given list of layers.
        Each layer is either a Linear or an Activation layer.

        Parameters:
            - layers: list of Layer objects
        """
        self.layers = layers

    def predict(self, x: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x)
        return np.squeeze(x)

    def weights(self) -> list[NDArray]:
        """Return a list of weight matrices with intercept in the last row."""
        return [layer.weights for layer in self.layers if isinstance(layer, Linear)]


class Fitter:
    def __init__(self, units, lambda_, *, mode):
        """Create a new fitter which trains a neural network with the given
        dimensions of hidden layers.

        Parameters:
            - units: list of integers, the sizes of hidden layers
            - lambda_: float, regularization parameter
            - mode: str, 'classification' or 'regression'

        Returns:
            - fitted ANN object
        """
        if mode not in ('classification', 'regression'):
            raise ValueError('Mode must be either classification or regression')

        self.units = units
        self.lambda_ = lambda_
        self.mode = mode
        self.loss_function = [log_loss, mse_loss][mode == 'regression']

    def initialize_model(self, X: NDArray, y: NDArray):
        input_dim = X.shape[1]
        output_dim = 1 if self.mode == 'regression' else np.unique(y).size
        layers = self._build_layers(input_dim, output_dim)

        # Create the model and some auxiliary variables
        self.model = ANN(layers)
        self.weights_mask = np.concatenate(
            [l.get_weights_mask() for l in layers if isinstance(l, Linear)]
        )
        self.weights_splits = np.cumsum([l.num_params() for l in layers])
        self.grad_splits = [
            slice(start, end)
            for start, end in zip(
                [0] + self.weights_splits.tolist(), self.weights_splits
            )
        ]

    def load_weights(self, weights: NDArray) -> None:
        for layer, w in zip(self.model.layers, np.split(weights, self.weights_splits)):
            if isinstance(layer, Linear):
                layer.load_weights(w)

    def get_weights(self) -> NDArray:
        return np.concatenate(
            [l.weights.flatten() for l in self.model.layers if isinstance(l, Linear)]
        )

    def _build_layers(self, input_dim: int, output_dim: int) -> list[Layer]:
        layers: list[Layer] = []
        for i, (in_dim, out_dim) in enumerate(
            zip([input_dim] + self.units, self.units + [output_dim])
        ):
            layers.append(Linear(in_dim, out_dim))
            if i < len(self.units):
                layers.append(ReLU())
        layers.append([Softmax(), Identity()][self.mode == 'regression'])
        return layers

    def _predict(self, X: NDArray, weights: NDArray) -> NDArray:
        """Forward pass with the given weights."""
        for layer, w in zip(self.model.layers, np.split(weights, self.weights_splits)):
            if isinstance(layer, Linear):
                layer.load_weights(w)
            X = layer.forward(X)
        return X

    def fit(self, X: NDArray, y: NDArray) -> ANN:
        """Create the network and optimise it using scipy.optimize."""
        self.initialize_model(X, y)
        output_dim = 1 if self.mode == 'regression' else np.unique(y).size

        # For classification, we need to convert the target to one-hot encoding
        if self.mode == 'classification':
            y = (np.arange(output_dim) == y[:, None]).astype(int)

        weights = self.get_weights()
        res = fmin_l_bfgs_b(self.loss_grad, weights, None, (X, y))
        self.load_weights(res[0])
        return self.model

    def loss_grad(
        self, weights: NDArray, X: NDArray, y: NDArray
    ) -> tuple[float, NDArray]:
        predictions = self._predict(X, weights)
        reg = self.lambda_ / 2 * np.sum(weights[self.weights_mask] ** 2)
        l = self.loss_function(predictions.squeeze(), y) + reg

        running_grad = (predictions - y.reshape(predictions.shape)) / len(y)
        grad = np.zeros_like(weights)
        # Skip the last activation layer - gradient is already incorporated
        for layer, split in zip(self.model.layers[-2::-1], self.grad_splits[-2::-1]):
            grad_w, running_grad = layer.backward(running_grad)
            grad[split] = grad_w.flatten()

        grad += self.lambda_ * weights * self.weights_mask
        return l, grad


class ANNClassification(Fitter):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_, mode='classification')


class ANNRegression(Fitter):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_, mode='regression')


def verify_gradient(fitter: Fitter) -> None:
    X = np.random.rand(1000, 4)
    y = 5 * X[:, 0] * X[:, 1] - 2 * X[:, 2] ** 2 + 1
    if fitter.mode == 'classification':
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        y = (4 * y).astype(int)
    fitter.initialize_model(X, y)
    if fitter.mode == 'classification':
        output_dim = len(np.unique(y))
        y = (np.arange(output_dim) == y[:, None]).astype(int)

    weights = fitter.get_weights()
    n = len(weights)

    diffs = []
    for eps in np.logspace(-2, -8, 15):
        _, grad = fitter.loss_grad(weights, X, y)
        approx_grad = np.zeros_like(grad)

        for i in range(len(weights)):
            loss_plus, _ = fitter.loss_grad(weights + eps * (np.arange(n) == i), X, y)
            loss_minus, _ = fitter.loss_grad(weights - eps * (np.arange(n) == i), X, y)
            approx_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        diffs.append(np.mean(np.abs(grad - approx_grad)))

    if fitter.mode == 'regression':
        plt.figure(figsize=(LINE_WIDTH / 1.3, LINE_WIDTH / 2))
        plt.ylabel('avg. abs. diff.')
    else:
        plt.figure(figsize=(LINE_WIDTH / 1.6, LINE_WIDTH / 2))
        plt.gca().tick_params(labelleft=False)
    plt.plot(np.logspace(-2, -8, 15), diffs, color='black')
    plt.xscale('log')
    plt.xlim(1e-2, 1e-8)
    plt.ylim(7.5e-5, 2.3e-4)
    plt.xlabel('$h$')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(f'nn_{fitter.mode}_gradient.pdf', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def housing_task() -> None:
    no_hidden: list[list[int]] = [[]]
    one_hidden = [[8], [16], [32], [64]]
    two_hidden = [[8, 8], [16, 8], [16, 16], [32, 8], [32, 16], [32, 32]]

    housing2r = pd.read_csv('housing2r.csv')
    X = housing2r.drop('y', axis=1).to_numpy()
    y = housing2r['y'].to_numpy()
    results_reg = pd.DataFrame(columns=['units', 'nn loss', 'lr loss'])
    for units in tqdm(no_hidden + one_hidden + two_hidden, desc='Housing 2r'):
        nn, _, lr, _ = housing_cv(X, y, ANNRegression(units, lambda_=0.01))
        results_reg.loc[len(results_reg)] = [units, nn, lr]
    results_reg.sort_values('nn loss').to_csv('housing2r_results.csv', index=False)

    housing3 = pd.read_csv('housing3.csv')
    X = housing3.drop('Class', axis=1).to_numpy()
    y = np.where(housing3['Class'] == 'C1', 0, 1)
    results_cls = pd.DataFrame(
        columns=['units', 'nn loss', 'nn acc', 'lr loss', 'lr acc']
    )
    for units in tqdm(no_hidden + one_hidden + two_hidden, desc='Housing 3'):
        res = housing_cv(X, y, ANNClassification(units, lambda_=0.01))
        results_cls.loc[len(results_cls)] = [units, *res]
    results_cls.sort_values('nn loss').to_csv('housing3_results.csv', index=False)


def housing_cv(
    X: NDArray, y: NDArray, fitter: Fitter
) -> tuple[float, float, float, float]:
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    results = np.zeros((10, 4))

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        results[i] = housing_eval(X_train, y_train, X_test, y_test, fitter)

    return tuple(np.mean(results, axis=0))  # type: ignore


def housing_eval(
    X_train: NDArray, y_train: NDArray, X_test: NDArray, y_test: NDArray, fitter: Fitter
) -> tuple[float, float, float, float]:
    """Evaluate the regression and classification neural network.

    Returns:
        - NN loss
        - NN accuracy for classification
        - LR loss
        - LR accuracy for classification
    """
    # Not great, needed for tests to work without hw3.py
    from hw3 import MultinomialLogReg

    m = fitter.fit(X_train, y_train)
    predictions = m.predict(X_test)

    if fitter.mode == 'classification':
        y_one_hot = (np.arange(2) == y_test[:, None]).astype(int)
        loss_nn = log_loss(predictions, y_one_hot)
        predictions = np.argmax(predictions, axis=1)
        acc_nn = np.mean(predictions == y_test)

        lr = MultinomialLogReg().build(X_train, y_train)
        predictions = lr.predict(X_test)
        loss_lr = log_loss(predictions, y_one_hot)
        predictions = np.argmax(predictions, axis=1)
        acc_lr = np.mean(predictions == y_test)

        return loss_nn, acc_nn, loss_lr, acc_lr

    loss_nn = mse_loss(predictions, y_test)
    lr = LinearRegression().fit(X_train, y_train)
    predictions = lr.predict(X_test)
    loss_lr = mse_loss(predictions, y_test)

    return loss_nn, 0, loss_lr, 0


def read_final_data() -> tuple[NDArray, NDArray, NDArray]:
    train = pd.read_csv('train.csv.gz', compression='gzip')
    test = pd.read_csv('test.csv.gz', compression='gzip')
    X_train = train.drop(['id', 'target'], axis=1).to_numpy()
    y_train = train['target'].str.split('_').str[1].astype(int).to_numpy() - 1
    X_test = test.drop('id', axis=1).to_numpy()
    return X_train, y_train, X_test


def final_eval() -> None:
    """Find the best model for the final dataset using CV on training data."""
    X_train, y_train, _ = read_final_data()
    results = pd.DataFrame(columns=['units', 'loss', 'accuracy', 'time'])

    no_hidden: list[list[int]] = [[]]
    one_hidden = [[8], [16], [32], [64]]
    two_hidden = [[8, 8], [16, 8], [16, 16], [32, 8], [32, 16], [32, 32], [64, 64]]
    three_hidden = [[8, 8, 8], [16, 16, 16], [32, 32, 32]]
    for units in tqdm(no_hidden + one_hidden + two_hidden + three_hidden, desc='Final'):
        fitter = ANNClassification(units, lambda_=1e-3)
        start = time()
        loss, acc = final_cv(X_train, y_train, fitter)
        results.loc[len(results)] = [units, loss, acc, time() - start]

    results = results.sort_values('loss')
    results.to_csv('final_models.csv', index=False)


def final_cv(X: NDArray, y: NDArray, fitter: Fitter) -> tuple[float, float]:
    """Evaluate fitter using CV and report loss and accuracy."""
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=2)
    results = np.zeros((folds, 2))

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m = fitter.fit(X_train, y_train)
        predictions = m.predict(X_test)

        loss = log_loss(predictions, (np.arange(9) == y_test[:, None]).astype(int))
        acc = np.mean(np.argmax(predictions, axis=1) == y_test)
        results[i] = loss, acc

    return tuple(np.mean(results, axis=0))  # type: ignore


def create_final_predictions() -> None:
    X_train, y_train, X_test = read_final_data()
    fitter = ANNClassification(units=[64, 64], lambda_=1e-3)
    m = fitter.fit(X_train, y_train)
    results = pd.DataFrame(
        m.predict(X_test),
        columns=[f'Class_{i + 1}' for i in range(9)],
    )
    results.index = results.index + 1
    results.index.name = 'id'
    results.to_csv('final.txt')


def main() -> None:
    # Plots configuration
    plt.rcParams['text.usetex'] = True
    plt.rcParams['lines.linewidth'] = 0.6
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.labelsize'] = 10

    verify_gradient(ANNRegression(units=[16, 8], lambda_=0.0001))
    verify_gradient(ANNClassification(units=[16, 8], lambda_=0.0001))

    housing_task()
    if EVAL_NN:
        final_eval()
    if GEN_PREDICTIONS:
        create_final_predictions()


if __name__ == '__main__':
    main()
