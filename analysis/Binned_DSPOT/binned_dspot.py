import numpy as np

class BinnedDSPOT:
    def __init__(self, q=1e-4, init_data=None, num_bins=50, t_quantile=0.98, window_size=200):
        """
        q: desired false alarm rate (tail probability)
        init_data: initial batch of data for POT
        num_bins: number of log-spaced bins for excesses
        t_quantile: quantile for initial threshold t
        window_size: sliding window size for DSPOT
        """
        self.q = q
        self.num_bins = num_bins
        self.window_size = window_size
        self.t = None
        self.bin_edges = None
        self.counts = None
        self.N_t = 0  # number of peaks
        self.n = 0    # total samples
        self.alpha = None
        self.xmin = None
        self.z_q = None
        # sliding window for DSPOT
        self.window = []

        if init_data is not None:
            self.fit_initial(init_data, t_quantile)

    def fit_initial(self, data, t_quantile=0.98):
        # compute initial threshold t
        self.n = len(data)
        self.t = np.quantile(data, t_quantile)
        # collect excesses
        excesses = data[data > self.t] - self.t
        self.N_t = len(excesses)
        if self.N_t < 1:
            raise ValueError("Not enough peaks above threshold for initial fit.")
        # set up log-spaced bins
        self.xmin = excesses.min()
        self.bin_edges = np.logspace(
            np.log10(self.xmin), np.log10(excesses.max()), self.num_bins + 1
        )
        self.counts, _ = np.histogram(excesses, bins=self.bin_edges)
        # fit tail
        self._estimate_tail()
        # compute initial z_q
        self._update_threshold()

    def _estimate_tail(self):
    	# pseudocount smoothing
    	counts = self.counts + 1

   	 # only keep bins with reasonable support
    	mids = np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])
    	mask = counts >= 5

    	x = np.log(mids[mask])
    	y = np.log(counts[mask])

    	slope, intercept = np.linalg.lstsq(
        	np.vstack([x, np.ones_like(x)]).T, y, rcond=None
    	)[0]
    	raw_alpha = -slope

    	# regularize alpha to avoid near-zero
    	self.alpha = max(raw_alpha, 0.5)

    	# floor xmin to avoid zero
    	raw_xmin = mids[mask][0]
    	self.xmin = max(raw_xmin, 1e-3)

    def _update_threshold(self):
        # invert Pareto CDF: P(X > z) = (xmin / (z - t + xmin))^alpha => z = t + xmin * (q*n/N_t)^(-1/alpha) - xmin
        factor = (self.N_t / (self.q * self.n))**(1 / self.alpha)
        self.z_q = self.t + self.xmin * factor - self.xmin

    def update(self, x_new):
        """
        Process new sample x_new. Returns True if anomaly detected.
        """
        # DSPOT: maintain sliding window and center
        self.window.append(x_new)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        # compute local mean
        M = np.mean(self.window)
        x_resid = x_new - M
        self.n += 1
        # anomaly check
        if x_resid > self.z_q:
            return True  # anomaly
        # peak update
        if x_resid > self.t:
            y = x_resid - self.t
            # find bin index
            idx = np.searchsorted(self.bin_edges, y, side='right') - 1
            if 0 <= idx < len(self.counts):
                self.counts[idx] += 1
                self.N_t += 1
                # re-fit and threshold
                self._estimate_tail()
                self._update_threshold()
        # no anomaly
        return False

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    # generate synthetic heavy-tailed data
    data = np.random.pareto(2.5, size=10000)
    detector = BinnedDSPOT(q=1e-4, init_data=data[:2000], num_bins=40, t_quantile=0.98)
    anomalies = []
    for x in data[2000:]:
        if detector.update(x):
            anomalies.append(x)
    print(f"Detected {len(anomalies)} anomalies out of {len(data)-2000} samples.")
