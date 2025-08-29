import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)
import matplotlib.pyplot as plt
from scipy.stats import kstest, genpareto
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds, kpss

"""
This module provides one function `plot_evt_diagnostics` that creates a 1×4 figure:

1. KPSS on full series
2. ACF of full series with min Ljung–Box p and BDS p-values
3. ACF of exceedances (q) with min Ljung–Box p and BDS p-values
4. QQ-plot of exceedances vs fitted GPD with KS p-value and count
"""

def plot_evt_diagnostics(series,
                         exceed_q=0.98,
                         acf_lags=40,
                         bds_max_dim=4,
                         figsize=(20,5),
                         title=None):
    """
    Parameters
    ----------
    series : pd.Series
        Preprocessed time series.
    exceed_q : float
        Quantile threshold for exceedances.
    acf_lags : int
        Number of lags for ACF/Ljung–Box.
    bds_max_dim : int
        Max embedding dimension for BDS.
    figsize : tuple
        Figure size.
    title : str
        Overall figure title.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes

    # 1) KPSS
    clean = series.dropna()
    #stat, p_kpss, _, _ = kpss(series, regression='c', nlags='auto')
    ax1.plot(series.index, series.values)
    ax1.set_title("Full Time Series")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')

    # 2) ACF full + LB + BDS
    max_lag = min(acf_lags, len(series)-1)
    plot_acf(series, ax=ax2, lags=max_lag, zero=False)
    lb_df = acorr_ljungbox(series, lags=list(range(1, max_lag+1)), return_df=True)
    lb_p = lb_df['lb_pvalue'].min()
    _, bds_p = bds(series.values, max_dim=bds_max_dim, epsilon=None)
    bds_txt = ", ".join(f"m={m}:p={p:.3f}" for m,p in zip(range(2, bds_max_dim+1), bds_p))
    ax2.set_title(f"ACF full series (min LB p<= {max_lag}: {lb_p:.3f}), \n BDS [{bds_txt}]")
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('ACF')

    # 3) ACF exceedances + LB + BDS
    thresh = series.quantile(exceed_q)
    exc = series[series > thresh]
    if len(exc) > 1:
        max_elag = min(acf_lags, len(exc)-1)
        plot_acf(exc, ax=ax3, lags=max_elag, zero=False)
        lb_e = acorr_ljungbox(exc, lags=list(range(1, max_elag+1)), return_df=True)
        lb_p_e = lb_e['lb_pvalue'].min()
        _, bds_pe = bds(exc.values, max_dim=bds_max_dim, epsilon=None)
        bds_txt_e = ", ".join(f"m={m}:p={p:.3f}" for m,p in zip(range(2, bds_max_dim+1), bds_pe))
        ax3.set_title(
            f"ACF exc q={exceed_q:.2f} (min LB p<= {max_elag}: {lb_p_e:.3f}), \n BDS [{bds_txt_e}]")
    else:
        ax3.text(0.5, 0.5, 'Not enough exceedances', ha='center')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('ACF')

    # 4) QQ vs GPD
    exc_vals = exc - thresh
    n_exc = len(exc_vals)
    if n_exc > 0:
        params = genpareto.fit(exc_vals, floc=0)
        c, loc, scale = params
        probs = (np.arange(1, n_exc+1) - 0.5) / n_exc
        emp = np.sort(exc_vals)
        theo = genpareto.ppf(probs, c, loc=loc, scale=scale)
        _, p_ks = kstest(exc_vals, 'genpareto', args=params)
        ax4.scatter(emp, theo, s=10)
        ax4.plot(emp, emp, 'k--')
        ax4.set_title(f"GPD QQ q={exceed_q:.2f}, n={n_exc}, KS p={p_ks:.3f}")
    else:
        ax4.text(0.5, 0.5, 'No exceedances', ha='center')
    ax4.set_xlabel('Emp quantiles')
    ax4.set_ylabel('Theor quantiles')

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0,0,1,0.95])
    else:
        fig.tight_layout()

    return fig
