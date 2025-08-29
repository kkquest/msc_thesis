import numpy as np

def grimshaw_fit(peaks, epsilon=1e-8):
    """
    Estimate GPD parameters (shape xi and scale sigma) for the given excess values using Grimshaw's method.
    Grimshaw's method finds the maximum likelihood estimates by solving the score equation for xi via root-finding:contentReference[oaicite:3]{index=3}.
    
    Parameters:
        peaks (1D array-like): Exceedances (data values above a threshold, minus the threshold).
        epsilon (float): Small value to avoid numerical issues near boundaries.
    
    Returns:
        xi_est (float): Estimated GPD shape parameter (ξ).
        sigma_est (float): Estimated GPD scale parameter (σ).
    """
    peaks = np.asarray(peaks, dtype=float)
    n = len(peaks)
    if n == 0:
        raise ValueError("No data points provided to grimshaw_fit")
    # Define the functions u(t), v(t) and the equation w(t) = u(t)*v(t) - 1 to find roots.
    def w_val(t):
        s = 1 + t * peaks
        if np.any(s <= 0):
            # t is too negative (invalid, as 1 + t*x > 0 is required for all x)
            return None
        u = 1 + np.mean(np.log(s))
        v = np.mean(1.0 / s)
        return u * v - 1

    # Determine search intervals for roots of w(t)
    y_min, y_max, y_mean = peaks.min(), peaks.max(), peaks.mean()
    # Left interval for negative roots (t in (a, 0), where a = -1/y_max is the lower bound to keep 1 + t*y_max > 0).
    a = -1.0 / y_max
    if abs(a) < 3 * epsilon:
        a = -3 * epsilon  # ensure a is sufficiently negative if y_max is extremely large
    left_interval = (a + epsilon, -epsilon)  # just above -1/y_max up to 0 (exclusive)
    # Initialize list of candidate root solutions
    candidate_roots = []

    # Find roots in the left interval by scanning for sign changes and using bisection.
    num_scan = 50
    ts = np.linspace(left_interval[0], left_interval[1], num_scan)
    w_values = [w_val(t) for t in ts]
    prev_t, prev_w = ts[0], w_values[0]
    for t, w_t in zip(ts[1:], w_values[1:]):
        if prev_w is None or w_t is None:
            prev_t, prev_w = t, w_t
            continue
        if prev_w * w_t < 0:  # sign change indicates a root in (prev_t, t)
            left, right = prev_t, t
            w_left, w_right = prev_w, w_t
            # Bisection refinement
            for _ in range(50):
                mid = 0.5 * (left + right)
                w_mid = w_val(mid)
                if w_mid is None:
                    # If mid is invalid, adjust slightly inside the interval
                    mid += epsilon
                    w_mid = w_val(mid)
                if w_mid is None:
                    break
                if w_left * w_mid <= 0:
                    right, w_right = mid, w_mid
                else:
                    left, w_left = mid, w_mid
                if abs(right - left) < 1e-6 or abs(w_mid) < 1e-6:
                    break
            candidate_roots.append(0.5 * (left + right))
        if w_t is not None and abs(w_t) < 1e-8:
            # Found a root (very close to zero crossing)
            candidate_roots.append(t)
        prev_t, prev_w = t, w_t

    # Find root in the positive domain (t > 0) if it exists.
    # We start from a very small positive t and increase until w(t) becomes negative (indicating a crossing from 0).
    t_pos = 1e-6
    w_pos = w_val(t_pos)
    # If w_val is extremely close to zero at t_pos, move a bit further for a clear sign.
    if w_pos is not None and abs(w_pos) < 1e-8:
        t_pos = 1e-4
        w_pos = w_val(t_pos)
    if w_pos is not None and w_pos > 0:
        # Increase t until w(t) < 0 or until a maximum limit.
        max_t = 1e6
        prev_t, prev_w = t_pos, w_pos
        while t_pos < max_t:
            t_pos *= 10.0
            w_curr = w_val(t_pos)
            if w_curr is None:
                break  # went out of valid range (though for t>0 with finite data, w_val usually stays defined)
            if w_curr < 0:
                # Root in (prev_t, t_pos)
                left, right = prev_t, t_pos
                w_left, w_right = prev_w, w_curr
                for _ in range(50):  # bisection to refine positive root
                    mid = 0.5 * (left + right)
                    w_mid = w_val(mid)
                    if w_mid is None:
                        mid += epsilon
                        w_mid = w_val(mid)
                    if w_mid is None:
                        break
                    if w_left * w_mid <= 0:
                        right, w_right = mid, w_mid
                    else:
                        left, w_left = mid, w_mid
                    if abs(right - left) < 1e-6 or abs(w_mid) < 1e-6:
                        break
                candidate_roots.append(0.5 * (left + right))
                break  # assume only one positive root (w will stay negative afterwards)
            prev_t, prev_w = t_pos, w_curr
    # If w_pos < 0 immediately, that means no positive root (w(t) went negative right after 0).

    # Always consider t = 0 as a candidate (corresponds to ξ = 0 case)
    candidate_roots.append(0.0)

    # Evaluate log-likelihood at each candidate root and choose the best
    def gpd_log_likelihood(xi, sigma):
        # GPD log-likelihood for given parameters (threshold = 0 for peaks)
        if sigma <= 0:
            return -np.inf
        if xi == 0:
            # Exponential case: L = -n*log(sigma) - sum(peaks)/sigma
            return -n * np.log(sigma) - np.sum(peaks) / sigma
        # If any 1 + xi * x/sigma <= 0, log-likelihood is invalid
        if np.any(1 + xi * peaks / sigma <= 0):
            return -np.inf
        return -n * np.log(sigma) - (1 + 1/xi) * np.sum(np.log(1 + xi * peaks / sigma))

    best_xi, best_sigma = 0.0, y_mean  # start with ξ=0, σ = mean(excess) as a baseline
    best_ll = gpd_log_likelihood(best_xi, best_sigma)
    for t in set(np.round(r, 6) for r in candidate_roots):
        # Compute xi and sigma from root t (if t = 0, that yields xi=0 which we already have as baseline)
        if abs(t) < 1e-8:
            xi = 0.0
            sigma = y_mean
        else:
            # For a root t, Grimshaw's formulas for MLE: ξ = u(1+t*x) - 1, σ = ξ / t
            s = 1 + t * peaks
            if np.any(s <= 0):
                continue
            xi = 1 + np.mean(np.log(s)) - 1  # (which simplifies to np.mean(np.log(s)))
            sigma = xi / t
        ll = gpd_log_likelihood(xi, sigma)
        if ll > best_ll:
            best_ll = ll
            best_xi, best_sigma = xi, sigma

    return best_xi, best_sigma

def gpd_fit(excesses):
    """
    Fit a Generalized Pareto Distribution (GPD) to the excess data.
    This uses Grimshaw's method under the hood to obtain MLEs for ξ and σ.
    
    Parameters:
        excesses (1D array-like): Values above threshold (with threshold subtracted).
    Returns:
        (xi, sigma): Tuple of fitted GPD parameters.
    """
    return grimshaw_fit(excesses)

def bootstrap_resample(data, size=None):
    """
    Generate a bootstrap resample of the given data.
    
    Parameters:
        data (array-like): Original data points.
        size (int, optional): Size of the resampled dataset. If None, uses len(data).
    Returns:
        np.ndarray: Array of resampled data (with replacement).
    """
    data = np.asarray(data)
    if size is None:
        size = len(data)
    return np.random.choice(data, size=size, replace=True)

def eqd(data, thresholds, k=100, m=500, min_excess=10, fit_func=None):
    """
    Apply the EQD threshold selection method on the data for given candidate thresholds.
    
    Parameters:
        data (array-like): 1D array of data values.
        thresholds (array-like): Sequence of candidate threshold values to evaluate.
        k (int): Number of bootstrap resamples for each threshold (default 100).
        m (int): Number of quantile levels to compare (default 500).
        min_excess (int): Minimum number of exceedances required to fit GPD (default 10).
        fit_func (callable, optional): Custom function to fit GPD. It should take an array of excesses 
                                       and return (xi, sigma). If None, uses the grimshaw-based gpd_fit.
    
    Returns:
        dict: A dictionary with keys:
              - "threshold": the selected threshold (value that minimizes mean quantile distance).
              - "params": (sigma, xi) tuple of GPD parameters at the selected threshold.
              - "num_excess": number of exceedances above the selected threshold.
              - "distances": array of mean quantile distances for each candidate threshold (NaN if not computed).
              - "all_sigmas": list of fitted σ for each threshold (NaN where not applicable).
              - "all_gammas": list of fitted ξ for each threshold (NaN where not applicable).
    """
    data = np.asarray(data, dtype=float)
    distances = []
    all_sigmas = []
    all_gammas = []
    num_excesses = []

    for u in thresholds:
        excess = data[data > u] - u  # excess values above threshold u
        n_exc = len(excess)
        num_excesses.append(n_exc)
        if n_exc < min_excess:
            # Not enough exceedances to fit GPD
            distances.append(np.nan)
            all_sigmas.append(np.nan)
            all_gammas.append(np.nan)
            continue

        # Fit GPD to the excesses
        if fit_func is not None:
            xi, sigma = fit_func(excess)
        else:
            xi, sigma = gpd_fit(excess)
        all_gammas.append(xi)
        all_sigmas.append(sigma)

        # Compute mean quantile distance via bootstrapping
        dist_list = []
        probs = np.linspace(1, m, m) / (m + 1)  # m equally spaced probabilities in (0,1)
        for _ in range(k):
            sample = bootstrap_resample(excess, size=n_exc)
            try:
                # Fit GPD to bootstrap sample
                if fit_func is not None:
                    xi_b, sigma_b = fit_func(sample)
                else:
                    xi_b, sigma_b = gpd_fit(sample)
            except Exception:
                # If fitting fails, skip this bootstrap iteration
                dist_list.append(np.nan)
                continue
            # Compute theoretical quantiles of GPD for probabilities probs
            if xi_b is None or sigma_b is None or np.isnan(xi_b) or np.isnan(sigma_b):
                dist_list.append(np.nan)
                continue
            if abs(xi_b) < 1e-8:
                # If shape ~ 0, use limit formula for quantiles (exponential case)
                gpd_quantiles = sigma_b * np.log(1 / probs)
            else:
                gpd_quantiles = (sigma_b / xi_b) * ((1 / probs) ** xi_b - 1)
            # Empirical quantiles of the bootstrap sample
            emp_quantiles = np.quantile(sample, probs)
            # Compute average absolute difference
            q_distance = np.nanmean(np.abs(emp_quantiles - gpd_quantiles))
            dist_list.append(q_distance)
        # Mean distance for this threshold (ignore NaNs from any failed fits)
        mean_dist = np.nanmean(dist_list) if len(dist_list) > 0 else np.nan
        distances.append(mean_dist)

    distances = np.array(distances)
    # Identify the threshold with minimum mean distance
    if np.all(np.isnan(distances)):
        best_idx = None
    else:
        best_idx = int(np.nanargmin(distances))
    result = {
        "threshold": float(thresholds[best_idx]) if best_idx is not None else None,
        "params": (float(all_sigmas[best_idx]), float(all_gammas[best_idx])) if best_idx is not None else None,
        "num_excess": int(num_excesses[best_idx]) if best_idx is not None else None,
        "distances": distances,
        "all_sigmas": all_sigmas,
        "all_gammas": all_gammas,
    }
    return result


# Example usage for eqd
# data = np.random.rand(1000)  # replace with your dataset
# thresholds = np.quantile(data, np.arange(0, 0.95, 0.05))  # e.g., 5th to 95th percentiles as candidates
# result = eqd(data, thresholds)
# print("Selected threshold:", result["threshold"])
# print("GPD parameters (sigma, xi):", result["params"])
