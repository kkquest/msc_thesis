# 1) Definition of the fixed BinnedDSPOT  
import numpy as np  
import matplotlib.pyplot as plt  

class BinnedDSPOT:  
    def __init__(  
        self,  
        q=1e-4,  
        init_data=None,  
        num_bins=50,  
        t_quantile=0.98,  
        window_size=200,  
        min_bin_count=5,  
        xmin_floor=1e-3,  
        force_min_alpha=0.5,  
        use_equipopulated=False  
    ):  
        self.q = q  
        self.num_bins = num_bins  
        self.window_size = window_size  
        self.min_bin_count = min_bin_count  
        self.xmin_floor = xmin_floor  
        self.force_min_alpha = force_min_alpha  
        self.use_equipopulated = use_equipopulated  

        self.t = None  
        self.bin_edges = None  
        self.counts = None  
        self.N_t = 0  
        self.n = 0  
        self.alpha = None  
        self.xmin = None  
        self.z_q = None  

        self.window = []  

        if init_data is not None:  
            self.fit_initial(init_data, t_quantile)  

    def fit_initial(self, data, t_quantile=0.98):  
        self.n = len(data)  
        self.t = np.quantile(data, t_quantile)  
        excesses = data[data > self.t] - self.t  
        self.N_t = len(excesses)  
        if self.N_t < 1:  
            raise ValueError("Not enough peaks above threshold for initial fit.")  

        if self.use_equipopulated:  
            sorted_ex = np.sort(excesses)  
            qs = np.linspace(0, 1, self.num_bins + 1)  
            self.bin_edges = np.unique(np.quantile(sorted_ex, qs))  
        else:  
            raw_xmin = excesses.min()  
            raw_xmax = excesses.max()  
            self.bin_edges = np.logspace(  
                np.log10(raw_xmin), np.log10(raw_xmax), self.num_bins + 1  
            )  

        self.counts, _ = np.histogram(excesses, bins=self.bin_edges)  
        self._estimate_tail()  
        self._update_threshold()  

    def _estimate_tail(self):  
        counts = self.counts + 1  
        mids = np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])  
        mask = counts >= self.min_bin_count  
        if not np.any(mask):  
            raise RuntimeError("No bins have enough counts for tail estimation.")  

        x = np.log(mids[mask])  
        y = np.log(counts[mask])  
        A = np.vstack([x, np.ones_like(x)]).T  
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]  
        raw_alpha = -slope  

        self.alpha = max(raw_alpha, 1)  
        raw_xmin = mids[mask][0]  
        self.xmin = max(raw_xmin, self.xmin_floor)  

    def _update_threshold(self):  
        factor = (self.N_t / (self.q * self.n)) ** (1 / self.alpha)  
        self.z_q = self.t + self.xmin * factor - self.xmin  


    def update(self, x_new):  
        self.window.append(x_new)  
        if len(self.window) > self.window_size:  
            self.window.pop(0)  
        M = np.mean(self.window)  

        x_resid = x_new - M  
        self.n += 1  

        #if x_resid > self.z_q:  # compare to the residuals 
         #   return True  
        if x_new > (M + self.z_q): # compare to the raw value
            return True


        if x_resid > self.t:  
            y = x_resid - self.t  
            idx = np.searchsorted(self.bin_edges, y, side='right') - 1  
            if 0 <= idx < len(self.counts):  
                self.counts[idx] += 1  
                self.N_t += 1  
                self._estimate_tail()  
                self._update_threshold()  

        return False  
