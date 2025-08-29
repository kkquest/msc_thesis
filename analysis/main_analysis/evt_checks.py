import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)
import matplotlib.pyplot as plt
from scipy.stats import kstest, genpareto
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds
import matplotlib.dates as mdates

def _fmt_bds_pvals(pvals, m_start=2):
    # Format like: "m=2:0.741, m=3:0.523"
    return ", ".join(f"m={m}: {p:.3f}" for m, p in zip(range(m_start, m_start+len(pvals)), pvals))


def plot_evt_diagnostics(series,
                         exceed_q=0.98,
                         acf_lags=40,
                         bds_max_dim=3,         # show m=2..3 in titles; raise if you need more
                         figsize=(20, 5),
                         title=None):
    """
    1) Full series (trace)
    2) ACF — full series (lags ≤ L); Ljung–Box min p = ...; BDS p: m=2:..., m=3:...
    3) ACF — excesses (q = ..., nₑ = ..., lags ≤ Lₑ); Ljung–Box min p = ...; BDS p: m=2:..., m=3:...
    4) GPD Q–Q — excesses (q = ..., nₑ = ...); KS p = ...
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes

    # 1) Full series trace
    ax1.plot(series.index, series.values, linewidth=1.0)
    ax1.set_title("Full series (trace)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    # Major ticks every 5 days; label as "DD Mon"
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    ax1.tick_params(axis="x", which="major", labelsize=8)
    ax1.grid(alpha=0.3, linestyle="--", which="major")
    ax1.tick_params(axis="x", which="major", labelrotation=45)

    # 2) ACF full + LB + BDS
    clean_full = series.dropna()
    max_lag = max(1, min(acf_lags, len(clean_full) - 1))
    plot_acf(clean_full, ax=ax2, lags=max_lag, zero=False)
    lb_df = acorr_ljungbox(clean_full, lags=list(range(1, max_lag + 1)), return_df=True)
    lb_p_min = float(lb_df["lb_pvalue"].min())
    _, bds_p = bds(clean_full.values, max_dim=bds_max_dim, epsilon=None)
    bds_txt = _fmt_bds_pvals(bds_p)
    ax2.set_title(f"ACF of full series (lags ≤ {max_lag}): \n Ljung–Box min p-value = {lb_p_min:.3f}; \n  "
                  f"BDS p-value: {bds_txt}")
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ACF")

    # Threshold & excesses
    u = series.quantile(exceed_q)
    exc = series[series > u]
    exc_vals = exc - u
    n_exc = int(exc_vals.shape[0])

    # 3) ACF of excesses + LB + BDS
    if n_exc > 1:
        max_elag = max(1, min(acf_lags, n_exc - 1))
        plot_acf(exc_vals, ax=ax3, lags=max_elag, zero=False)
        lb_e = acorr_ljungbox(exc_vals, lags=list(range(1, max_elag + 1)), return_df=True)
        lb_p_e_min = float(lb_e["lb_pvalue"].min())
        _, bds_p_e = bds(exc_vals.values, max_dim=bds_max_dim, epsilon=None)
        bds_txt_e = _fmt_bds_pvals(bds_p_e)
        ax3.set_title(
            f"ACF of excesses (lags ≤ {max_elag}): \n "
            f"Ljung–Box min p-value = {lb_p_e_min:.3f}; \n BDS p-value: {bds_txt_e}"
        )
    else:
        ax3.text(0.5, 0.5, f"Insufficient exceedances (q = {exceed_q:.2f})",
                 ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("ACF")

    # 4) GPD Q–Q
    if n_exc > 0:
        params = genpareto.fit(exc_vals, floc=0)
        c, loc, scale = params
        probs = (np.arange(1, n_exc + 1) - 0.5) / n_exc
        emp = np.sort(exc_vals.values)
        theo = genpareto.ppf(probs, c, loc=loc, scale=scale)
        _, p_ks = kstest(exc_vals, "genpareto", args=params)

        ax4.scatter(emp, theo, s=10)
        # 45-degree reference
        mn = min(emp.min(), theo.min())
        mx = max(emp.max(), theo.max())
        ax4.plot([mn, mx], [mn, mx], "k--", linewidth=1.0)

        ax4.set_title(f"GPD Q–Q of excesses (q = {exceed_q:.2f}, n_exc = {n_exc}); \n KS test p-value = {p_ks:.3f}")
    else:
        ax4.text(0.5, 0.5, "No exceedances", ha="center", va="center", transform=ax4.transAxes)

    ax4.set_xlabel("Empirical quantiles")
    ax4.set_ylabel("Theoretical quantiles")

    if title:
        fig.suptitle(title,
                 x=0.0, y=1,           # left align, a bit above the top
                 ha="left", va="bottom",  # align relative to that anchor
                 fontsize=14,             # bigger font
                 fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()
    return fig
