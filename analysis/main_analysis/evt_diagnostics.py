import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, kstest, pareto
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox



def fit_gpd_exceedances(series, u):
    """
    Fit GPD to excesses above u.
    Returns parameters and exceedance data.
    """
    excess = series[series > u] - u
    params = genpareto.fit(excess, floc=0)
    return params, excess


def gpd_qq_data(excess, params):
    c, loc, scale = params
    n = len(excess)
    probs = (np.arange(1, n+1) - 0.5) / n
    emp_q = np.sort(excess)
    theo_q = genpareto.ppf(probs, c, loc=loc, scale=scale)
    return emp_q, theo_q


def fit_pareto_mle(series, u):
    """
    Fit Pareto via MLE: alpha = n / sum(log(x/u)).
    Returns alpha and exceedances.
    """
    exc = series[series > u]
    n = len(exc)
    alpha = n / np.sum(np.log(exc / u)) if n > 0 else np.nan
    return alpha, exc


def pareto_qq_data(exc, u, alpha):
    n = len(exc)
    probs = (np.arange(1, n+1) - 0.5) / n
    emp_q = np.sort(exc - u)
    theo_q = u * ((1 - probs) ** (-1/alpha) - 1)
    return emp_q, theo_q



def plot_acf_only(series, lags=300, figsize=(8,4), title="ACF of full series"):
    """
    Plot the ACF with Ljung-Box p-value.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_acf(series, ax=ax, lags=lags, zero=False)
    lb = acorr_ljungbox(series, lags=[lags], return_df=True)
    pval = lb['lb_pvalue'].iloc[0]
    ax.set_title(f"ACF of full series (Ljung-Box p={pval:.3f})")
    fig.tight_layout()
    return fig


def plot_gpd_qq(series, u, figsize=(6,6), title="ACF of exceedances above 98th quantile"):
    """
    Plot QQ-plot of GPD exceedances above threshold u with KS p-value.
    """
    params, excess = fit_gpd_exceedances(series, u)
    emp_q, theo_q = gpd_qq_data(excess, params)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emp_q, theo_q, s=10)
    ax.plot(emp_q, emp_q, 'k--')
    _, p_ks = kstest(excess, 'genpareto', args=params)
    ax.set_title(title or f"GPD QQ (u={u:.2f}, KS p={p_ks:.3f})")
    ax.set_xlabel('Empirical quantiles')
    ax.set_ylabel('Theoretical quantiles')
    fig.tight_layout()
    return fig


def plot_pareto_qq(series, u, figsize=(6,6), title=None):
    """
    Plot QQ-plot of Pareto exceedances above threshold u with KS p-value.
    """
    alpha, exc = fit_pareto_mle(series, u)
    emp_q, theo_q = pareto_qq_data(exc, u, alpha)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(emp_q, theo_q, s=10)
    ax.plot(emp_q, emp_q, 'k--')
    y = exc / u
    _, p_pareto = kstest(y, 'pareto', args=(alpha, 0, 1))
    ax.set_title(title or f"Pareto QQ (u={u:.2f}, α={alpha:.2f}, KS p={p_pareto:.3f})")
    ax.set_xlabel('Empirical exceedances')
    ax.set_ylabel('Theoretical exceedances')
    fig.tight_layout()
    return fig


def select_best_threshold(series, model='gpd', quantiles=(0.90,0.9998), n=50):
    """
    Grid-search thresholds in given quantile range to maximize KS p-value.
    model: 'gpd' or 'pareto'
    Skips thresholds with no exceedances.
    """
    qs = np.linspace(quantiles[0], quantiles[1], n)
    best = {'u': None, 'p': -1}
    for q in qs:
        u = series.quantile(q)
        # compute exceedances
        if model == 'gpd':
            _, excess = fit_gpd_exceedances(series, u)
            if len(excess) < 5:
                continue
            params = genpareto.fit(excess, floc=0)
            _, p = kstest(excess, 'genpareto', args=params)
        else:
            alpha, exc = fit_pareto_mle(series, u)
            if len(exc) < 5 or np.isnan(alpha) or alpha <= 0:
                continue
            y = exc / u
            _, p = kstest(y, 'pareto', args=(alpha,0,1))
        if p > best['p']:
            best = {'u': u, 'p': p}
    return best['u'], best['p']


def plot_four_diagnostics(series, lags=40,
                           gpd_q=(0.90,0.995), pareto_q=(0.90,0.995),
                           exceed_q=0.98, n=50, figsize=(20,5), title=None):
    """
    Plot 1) ACF of full series
         2) ACF of exceedances above exceed_q quantile
         3) GPD QQ
         4) Pareto QQ
    series: pd.Series of data
    exceed_q: quantile for selecting exceedances
    title: optional overall title
    """
    # thresholds
    u_gpd, p_gpd = select_best_threshold(series, model='gpd', quantiles=gpd_q, n=n)
    u_par, p_par = select_best_threshold(series, model='pareto', quantiles=pareto_q, n=n)
    # exceedances ACF
    u_ex = series.quantile(exceed_q)
    exc_series = series[series > u_ex]

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    ax0, ax1, ax2, ax3 = axes

    # 1: full series ACF
    plot_acf(series, ax=ax0, lags=lags, zero=False)
    lb0 = acorr_ljungbox(series, lags=[lags], return_df=True)
    p0 = lb0['lb_pvalue'].iloc[-1]
    ax0.set_title(f"ACF of full series (LB p={p0:.3f})")

    # 2: exceedances ACF
    plot_acf(exc_series, ax=ax1, lags=lags, zero=False)
    lb1 = acorr_ljungbox(exc_series, lags=[lags], return_df=True)
    p1 = lb1['lb_pvalue'].iloc[-1]
    ax1.set_title(f"ACF of exceedances above 98th quantile (q={exceed_q:.2f}, LB p={p1:.3f})")

    # 3: GPD QQ
    params, excess = fit_gpd_exceedances(series, u_gpd)
    emp_q, theo_q = gpd_qq_data(excess, params)
    ax2.scatter(emp_q, theo_q, s=10)
    ax2.plot(emp_q, emp_q, 'k--')
    _, p_ks = kstest(excess, 'genpareto', args=params)
    ax2.set_title(f"GPD QQ (u={u_gpd:.2f}, p={p_ks:.3f})")
    ax2.set_xlabel('Emp quantiles')
    ax2.set_ylabel('Theo quantiles')

    # 4: Pareto QQ
    alpha, exc = fit_pareto_mle(series, u_par)
    emp_q_p, theo_q_p = pareto_qq_data(exc, u_par, alpha)
    ax3.scatter(emp_q_p, theo_q_p, s=10)
    ax3.plot(emp_q_p, emp_q_p, 'k--')
    y = exc / u_par
    _, p_pareto = kstest(y, 'pareto', args=(alpha,0,1))
    ax3.set_title(f"Pareto QQ (u={u_par:.2f}, α={alpha:.2f}, p={p_pareto:.3f})")
    ax3.set_xlabel('Emp exceedances')
    ax3.set_ylabel('Theo exceedances')

    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig
