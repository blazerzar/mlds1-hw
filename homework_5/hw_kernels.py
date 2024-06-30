from abc import ABC, abstractmethod
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cvxopt import matrix  # type: ignore
from cvxopt import solvers  # type: ignore
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from sklearn.model_selection import KFold  # type: ignore

LINE_WIDTH = 3.220
SHOW_PLOTS = True
SAVE_PLOTS = False


class Kernel(Protocol):
    def __call__(self, a: NDArray, b: NDArray):
        """Compute the kernel values between:
            - two vectors: results is a number
            - a vector and a matrix: results is a vector
            - two matrices: results is a matrix

        Parameters:
            - a: a vector or a matrix
            - b: a vector or a matrix

        Returns:
            - a number, a vector, or a matrix
        """


class Polynomial:
    """The polynomial kernel."""

    def __init__(self, M: int):
        """Initialize the polynomial kernel.

        Parameters:
            - M: the degree of the polynomial kernel
        """
        self.M = M

    def __call__(self, a: NDArray, b: NDArray):
        """Compute the polynomial kernel according to the protocol."""
        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if len(b.shape) == 1:
            b = b.reshape(1, -1)

        values = (1 + a @ b.T) ** self.M
        return values.item() if values.shape == (1, 1) else values.squeeze()


class RBF:
    """The radial basis function kernel."""

    def __init__(self, sigma: float):
        """Initialize the RBF kernel.

        Parameters:
            - sigma: kernel bandwidth
        """
        self.sigma = sigma

    def __call__(self, a: NDArray, b: NDArray):
        """Compute the RBF kernel according to the protocol."""
        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if len(b.shape) == 1:
            b = b.reshape(1, -1)

        norms_a = np.tile(np.sum(a**2, axis=1), (b.shape[0], 1)).T
        norms_b = np.tile(np.sum(b**2, axis=1), (a.shape[0], 1))
        values = np.exp(-(norms_a + norms_b - 2 * a @ b.T) / (2 * self.sigma**2))
        return values.item() if values.shape == (1, 1) else values.squeeze()


class Predictor:
    def __init__(self, X: NDArray, alpha: NDArray, kernel: Kernel, b: float = 0):
        self._alpha = alpha
        self.kernel = kernel
        self.b = b

        self.alpha = alpha
        if len(alpha.shape) == 2:
            self.alpha = alpha[:, 0] - alpha[:, 1]
        self.support = np.abs(self.alpha) > 1e-5

        self.X = X[self.support]
        self.alpha = self.alpha[self.support]

    def predict(self, X: NDArray):
        """Make prediction using the kernel method for instances in the rows of X."""
        return ((self.kernel(X, self.X)) @ self.alpha + self.b).squeeze()

    def get_alpha(self) -> NDArray:
        return self._alpha

    def get_b(self) -> float:
        return self.b


class Fitter(ABC):
    def __init__(self, kernel: Kernel, lambda_: float):
        self.kernel = kernel
        self.lambda_ = lambda_

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> Predictor:
        """Fit the model to the data.

        Parameters:
            - X: the input data of shape (n_samples, n_features)
            - y: the target data of shape (n_samples,)

        Returns:
            - a predictor
        """


class KernelizedRidgeRegression(Fitter):
    def fit(self, X: NDArray, y: NDArray) -> Predictor:
        K = self.kernel(X, X)
        alpha = np.linalg.solve(K + self.lambda_ * np.eye(K.shape[0]), y)
        return Predictor(X, alpha, self.kernel)


class SVR(Fitter):
    def __init__(self, kernel: Kernel, lambda_: float, epsilon: float):
        super().__init__(kernel, lambda_)
        self.epsilon = epsilon

    def fit(self, X: NDArray, y: NDArray) -> Predictor:
        l = X.shape[0]
        K = self.kernel(X, X)

        # min 1/2 x^T P x + q^T x
        P = np.kron(K, [[1.0, -1.0], [-1.0, 1.0]])
        q = self.epsilon + np.repeat(y, 2) * np.tile([-1, 1], l)

        # Gx <= h
        G = np.vstack([-np.eye(2 * l), np.eye(2 * l)])
        h = np.repeat([0, 1.0 / self.lambda_], 2 * l)

        # Ax = b
        A = np.tile([1.0, -1.0], l).reshape(1, -1)
        b = np.array([0.0])

        solvers.options['show_progress'] = False
        solution = solvers.qp(*(matrix(arg) for arg in [P, q, G, h, A, b]))
        alpha = np.array(solution['x']).reshape(-1, 2)
        offset = solution['y'][0]

        return Predictor(X, alpha, self.kernel, offset)


def get_sine_data() -> pd.DataFrame:
    sine = pd.read_csv('sine.csv')
    min, max = sine['x'].min(), sine['x'].max()
    sine['x'] = (sine['x'] - min) / (max - min)
    return sine


def sine_eval(data: pd.DataFrame, fitter: Fitter, name: str):
    """Plot data points, support vectors, and the prediction for the sine
    dataset using both methods and both kernels."""
    X = data[['x']].to_numpy()
    y = data['y'].to_numpy()
    m = fitter.fit(X, y)
    sv = m.support
    print('Ratio of support vectors: ', np.mean(sv), f'({name})')

    xs = np.linspace(np.min(X), np.max(X), 500).reshape(-1, 1)
    pred = m.predict(xs)
    plt.figure(figsize=(LINE_WIDTH, LINE_WIDTH / 1.8))
    plt.scatter(X[~sv], y[~sv], 15, color='k', facecolor='#aaa', linewidth=0.8)
    plt.scatter(X[sv], y[sv], 15, color='k', facecolor='white', linewidth=0.8)
    plt.plot(xs, pred, color='crimson', linewidth=1.0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(f'sine_{name}.pdf', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def read_housing_data() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    data = pd.read_csv('housing2r.csv')
    X, y = data.drop('y', axis='columns').to_numpy(), data['y'].to_numpy()

    train_size = int(0.8 * X.shape[0])
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    min, max = X_train.min(axis=0), X_train.max(axis=0)
    X_train = (X_train - min) / (max - min)
    X_test = (X_test - min) / (max - min)

    return X_train, y_train, X_test, y_test


def housing_lambda(data: tuple[NDArray, NDArray], fitter: Fitter) -> float:
    """Find the best lambda for the housing dataset using cross-validation."""
    X, y = data
    lambdas = np.logspace(-5, 5, 11)
    best_lambda, best_mse = 0, np.inf

    for lambda_ in lambdas:
        fitter.lambda_ = lambda_
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        mse = np.mean(
            [
                housing_mse(
                    (X[train_index], y[train_index], X[test_index], y[test_index]),
                    fitter,
                )[0]
                for train_index, test_index in kf.split(X)
            ]
        )
        if mse < best_mse:
            best_lambda, best_mse = lambda_, mse  # type: ignore

    return best_lambda


def housing_mse(
    data: tuple[NDArray, NDArray, NDArray, NDArray], fitter: Fitter, *, cv=False
) -> tuple[float, int]:
    """Compute MSE and number of support vectors for the housing dataset. If cv
    is True, use cross-validation to find the best lambda."""
    X_train, y_train, X_test, y_test = data

    if cv:
        fitter.lambda_ = housing_lambda((X_train, y_train), fitter)

    m = fitter.fit(X_train, y_train)
    pred = m.predict(X_test)
    return np.mean((pred - y_test) ** 2), np.sum(m.support)


def housing_eval(
    data: tuple[NDArray, NDArray, NDArray, NDArray], fitter: Fitter, name: str
):
    """Plot MSE curves for the given fitter on the housing dataset."""
    poly = isinstance(fitter.kernel, Polynomial)
    values = np.arange(1, 11) if poly else np.linspace(0.5, 6.5, 13)

    def set_value(value):
        fitter.lambda_ = 1
        if poly:
            fitter.kernel.M = value
        else:
            fitter.kernel.sigma = value

    # For each value we compute MSE using two lambdas and number of support vectors
    mse_results = np.zeros((len(values), 4))

    for i, value in enumerate(values):
        set_value(value)
        mse_results[i, :2] = housing_mse(data, fitter)
        mse_results[i, 2:] = housing_mse(data, fitter, cv=True)

    fig_width = LINE_WIDTH / (1.5 if name.startswith('svr') else 1.8)
    _, ax = plt.subplots(figsize=(LINE_WIDTH, fig_width))
    plt.plot(values, mse_results[:, 0], label='$\lambda = 1$', color='k')
    plt.plot(values, mse_results[:, 2], label='CV $\lambda$', color='k', linestyle='--')

    plt.xlabel('M' if poly else '$\sigma$')
    plt.ylabel('MSE')
    plt.xlim(values[0], values[-1])
    plt.xticks(values if poly else np.arange(1, 7))
    plt.tight_layout()
    frame = plt.legend(loc='upper right', borderpad=0.1).get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor('black')
    frame.set_boxstyle('square')  # type: ignore

    # Annotate minimum MSE
    best_mse = np.min(mse_results[:, [0, 2]])
    best_val = values[np.argmin(np.min(mse_results[:, [0, 2]], axis=1))]
    x, y = best_val - 1, np.mean(plt.ylim())
    plt.text(x, y, f'{best_mse:.2f}', fontsize=9, ha='center', va='bottom')  # type: ignore
    plt.plot([best_val, x], [best_mse, y], color='k', linewidth=0.8)

    if name.startswith('svr'):
        plot_num_support_vectors(values, mse_results, ax)

    if SAVE_PLOTS:
        plt.savefig(f'housing_{name}.pdf', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_num_support_vectors(values: NDArray, results: NDArray, ax) -> None:
    # Number of support vectors is shown above the plot
    top = plt.ylim()[1]
    left = plt.xlim()[0]
    height = plt.ylim()[1] - plt.ylim()[0]
    width = plt.xlim()[1] - plt.xlim()[0]
    rect_size = 1.08 * width, 0.12 * height
    rect_start = left - 0.04 * width
    plt.subplots_adjust(top=0.8)

    kwargs = {
        'facecolor': 'white',
        'edgecolor': 'black',
        'linewidth': 0.8,
        'clip_on': False,
    }
    r_1 = Rectangle((rect_start, top + 0.19 * height), *rect_size, **kwargs)  # type: ignore
    r_cv = Rectangle((rect_start, top + 0.04 * height), *rect_size, linestyle='--', **kwargs)  # type: ignore
    ax.add_patch(r_1)
    ax.add_patch(r_cv)

    text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 7}

    # Plot number of support vectors for lambda = 1
    for x, y in zip(values, results[:, 1]):
        plt.text(x, top + 0.24 * height, str(int(y)), **text_kwargs)  # type: ignore
    # Plot number of support vectors for CV
    for x, y in zip(values, results[:, 3]):
        plt.text(x, top + 0.08 * height, str(int(y)), **text_kwargs)  # type: ignore


def main() -> None:
    np.random.seed(1)

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

    sine = get_sine_data()
    sine_eval(sine, KernelizedRidgeRegression(Polynomial(19), 1e-5), 'rr_poly')
    sine_eval(sine, KernelizedRidgeRegression(RBF(0.1), 1e-3), 'rr_rbf')
    sine_eval(sine, SVR(Polynomial(19), 1e-5, 0.4), 'svr_poly')
    sine_eval(sine, SVR(RBF(0.1), 1e-3, 0.4), 'svr_rbf')

    # Kernel parameters are set by the housing_eval function
    housing = read_housing_data()
    housing_eval(housing, KernelizedRidgeRegression(Polynomial(0), 0), 'rr_poly')
    housing_eval(housing, KernelizedRidgeRegression(RBF(0), 0), 'rr_rbf')
    housing_eval(housing, SVR(Polynomial(0), 0, 8), 'svr_poly')
    housing_eval(housing, SVR(RBF(0), 0, 8), 'svr_rbf')


if __name__ == '__main__':
    main()
