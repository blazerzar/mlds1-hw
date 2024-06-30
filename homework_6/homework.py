import arviz as az  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore
from scipy.stats import norm  # type: ignore

LINE_WIDTH = 3.220


def normalize(x):
    min, max = np.min(x), np.max(x)
    return (x - min) / (max - min)


def mcmc(
    data: pd.DataFrame, prior_mu: NDArray, prior_sigma: NDArray
) -> az.InferenceData:
    rand = np.random.default_rng(1)

    with pm.Model() as model:
        angle_data = pm.Data('angle', data['Angle'])
        distance_data = pm.Data('distance', data['Distance'])

        intercept = pm.Normal('intercept', mu=prior_mu[0], sigma=prior_sigma[0])
        beta_angle = pm.Normal('beta_angle', mu=prior_mu[1], sigma=prior_sigma[1])
        beta_distance = pm.Normal('beta_distance', mu=prior_mu[2], sigma=prior_sigma[2])

        lin_est = intercept + beta_angle * angle_data + beta_distance * distance_data
        mu = pm.invlogit(lin_est)
        pm.Bernoulli('posterior', p=mu, observed=data['Made'])

    with model:
        samples = pm.sample(10000, random_seed=rand)

    return samples


def plot_posterior(samples: az.InferenceData, name: str, show=False) -> None:
    az.plot_pair(
        samples,
        kind='kde',
        figsize=(LINE_WIDTH, LINE_WIDTH),
        var_names=['beta_angle', 'beta_distance'],
        textsize=11,
        kde_kwargs={
            'contourf_kwargs': {'cmap': 'Blues'},
        },
    )
    plt.xlabel(r'$\beta_\mathrm{angle}$')
    plt.ylabel(r'$\beta_\mathrm{distance}$')
    plt.savefig(name + '.pdf', bbox_inches='tight')
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()


def posterior_grad(
    parameters: NDArray, X: NDArray, y: NDArray, prior_mu: NDArray, prior_sigma: NDArray
) -> tuple[float, NDArray]:
    """Compute the logistic regression posterior for the normal prior and its
    gradient.

    Parameters:
        - parameters: the model parameters with shape (3,)
        - X: the data matrix with shape (n, 2)
        - y: the target vector with shape (n,)

    Returns:
        - negative log-posterior (minimization problem)
        - gradient of the negative log-posterior
    """
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    lin_est = X @ parameters
    mu = 1 / (1 + np.exp(-lin_est))
    sigma_inv = np.linalg.inv(prior_sigma)

    p = parameters - prior_mu
    likelihood = -np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    prior = 0.5 * p.T @ sigma_inv @ p

    grad = X.T @ (mu - y) + sigma_inv @ p

    return likelihood + prior, grad


def posterior_hessian(parameters: NDArray, X: NDArray, prior_sigma: NDArray) -> NDArray:
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    lin_est = X @ parameters
    mu = 1 / (1 + np.exp(-lin_est))
    return X.T @ np.diag(mu * (1 - mu)) @ X + np.linalg.inv(prior_sigma)


def opinion_plots(show=False) -> None:
    betas = np.linspace(-3.5, 0.5, 100)
    p = norm.pdf(betas, -1.6, 0.4)

    plt.figure(figsize=(LINE_WIDTH, 0.4 * LINE_WIDTH))
    plt.plot(betas, p, color='black', linewidth=0.8)
    plt.xlabel(r'$\beta_\mathrm{distance}$')
    plt.ylabel('probability')
    plt.xlim(-3.5, 0.5)

    plt.savefig('prior_opinion.pdf', bbox_inches='tight')
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()


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

    data = pd.read_csv('dataset.csv')
    data[['Angle', 'Distance']] = data[['Angle', 'Distance']].apply(normalize)
    X = data[['Angle', 'Distance']].to_numpy()
    y = data['Made'].to_numpy()

    prior_mu = np.zeros(3)
    prior_sigma = np.eye(3)

    samples = mcmc(data, prior_mu, np.diag(prior_sigma))
    plot_posterior(samples, 'posterior_1000')
    print(az.summary(samples))

    angles = samples['posterior']['beta_angle'].values
    distances = samples['posterior']['beta_distance'].values
    # P(distance < angle) ... probability that distance is more important
    distance_more_important = np.mean(distances < angles)
    # P(angle < 0) ... probability that angle decreases shot accuracy
    angle_decreases_accuracy = np.mean(angles < 0)
    print(f'P(β_distance < β_angle) = {distance_more_important:.3f}')
    print(f'P(β_angle < 0) = {angle_decreases_accuracy:.3f}')

    samples = mcmc(data.sample(50, random_state=1), prior_mu, np.diag(prior_sigma))
    plot_posterior(samples, 'posterior_50')
    print(az.summary(samples))

    res = minimize(
        posterior_grad,
        np.zeros(3),
        (X, y, prior_mu, prior_sigma),
        'L-BFGS-B',
        True,
    )
    hessian = posterior_hessian(res.x, X, prior_sigma)
    var = np.diag(np.linalg.inv(hessian))

    print('μ =', res.x)
    print('√Σ =', np.sqrt(var))

    opinion_plots()


if __name__ == '__main__':
    main()
