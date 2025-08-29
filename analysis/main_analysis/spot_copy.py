"""
SPOT algorithms for anomaly detection edited from the following repository,
which is released under the GNU GPLv3 license
https://github.com/Amossys-team/SPOT
"""
# pylint: disable=invalid-name
from abc import ABC
from enum import Enum
from enum import auto
import json
import logging
from typing import Callable
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import deque



_Template = TypeVar("_Template")


def _asc_key(value: _Template) -> _Template:
    return value


def _desc_key(value: _Template) -> _Template:
    return -value


class ExtremeValue:
    """
    Extreme value with one threshold
    """

    class Status(Enum):
        """
        Detection result
        """

        NORMAL = auto()
        ABNORMAL = auto()
        ALARM = auto()

    def __init__(
        self,
        q: float = 1e-4,
        n_points: int = 10,
        key: Callable[[_Template], _Template] = _asc_key,
        logging_level: int = logging.WARNING,
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
        """
        self._proba = q
        self._n_points = n_points
        self._key = key

        self._extreme_quantile: float = None
        self._init_threshold: float = None
        self._peaks: np.ndarray = None

        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._logger.setLevel(level=logging_level)

    @property
    def extreme_quantile(self) -> float:
        """
        current threshold (bound between normal and abnormal events)
        """
        return self._extreme_quantile

    @property
    def num_peaks(self) -> int:
        """
        number of observed peaks
        """
        return self._peaks.size

    def summary(self) -> dict:
        """
        Summary running status
        """
        return {
            "Detection level q": self._proba,
            "initial threshold": self._init_threshold,
            "#(peaks)": self.num_peaks,
            "extreme quantile": self._extreme_quantile,
        }

    @staticmethod
    def _roots_finder(
        fun: Callable[[float], float],
        jac: Callable[[float], float],
        bounds: Tuple[float, float],
        npoints: int,
        method: str,
    ) -> np.ndarray:
        """
        Find possible roots of a scalar function

        Parameters:
            fun: scalar function
            jac: first order derivative of the function
            bounds: (min,max) interval for the roots search
            npoints: maximum number of roots to output
            method:
                'regular' : regular sample of the search interval,
                'random' : uniform (distribution) sample of the search interval

        Returns: possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            initial_guess = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            initial_guess = np.random.uniform(bounds[0], bounds[1], npoints)

        def _object(variable: np.ndarray) -> Tuple[float, np.ndarray]:
            value = np.array([fun(item) for item in variable])
            gradient = np.array([jac(item) for item in variable])
            return (value**2).sum(), 2 * value * gradient

        opt = minimize(
            _object,
            initial_guess,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(initial_guess),
        )

        X: np.ndarray = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    @staticmethod
    def _log_likelihood(Y: np.ndarray, gamma: float, sigma: float) -> float:
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters:
            Y: observations
            gamma: GPD index parameter
            sigma: GPD scale parameter (>0)

        Returns: log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * np.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + np.log(Y.mean()))
        return L

    def _grimshaw(
        self, peaks: np.ndarray, epsilon: float = 1e-8
    ) -> Tuple[float, float, float]:
        # pylint: disable=too-many-locals
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters:
            epsilon: numerical parameter to perform (default : 1e-8)

        Returns: gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def _u(s: np.ndarray) -> float:
            return 1 + np.log(s).mean()

        def _v(s: np.ndarray) -> float:
            return np.mean(1 / s)

        def _w(t: float) -> float:
            s = 1 + t * peaks
            us = _u(s)
            vs = _v(s)
            return us * vs - 1

        def _jac_w(t: float) -> float:
            s = 1 + t * peaks
            us = _u(s)
            vs = _v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s**2))
            return us * jac_vs + vs * jac_us

        y_min: float = peaks.min()
        y_max: float = peaks.max()
        y_mean: float = peaks.mean()

        a = -1 / y_max
        if abs(a) < 3 * epsilon:
            epsilon = abs(a) / self._n_points

        a = a + epsilon

        # We look for possible roots
        left_zeros = self._roots_finder(
            _w,
            _jac_w,
            (a + epsilon, -epsilon),
            self._n_points,
            "regular",
        )

        if y_mean > y_min > 0 and not np.isclose(y_mean, y_min):
            b = 2 * (y_mean - y_min) / (y_mean * y_min)
            c = 2 * (y_mean - y_min) / (y_min**2)
            right_zeros = self._roots_finder(
                _w,
                _jac_w,
                (b, c),
                self._n_points,
                "regular",
            )
            # all the possible roots
            zeros = np.concatenate((left_zeros, right_zeros))
        else:
            zeros = left_zeros

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = y_mean
        ll_best = self._log_likelihood(peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = _u(1 + z * peaks) - 1
            sigma = gamma / z
            ll = self._log_likelihood(peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, num: int, gamma: float, sigma: float) -> float:
        """
        Compute the quantile at level 1-q

        Parameters:
            gamma: GPD parameter
            sigma: GPD parameter

        Returns: quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = num * self._proba / self.num_peaks
        if gamma != 0:
            return self._init_threshold + self._key(
                (sigma / gamma) * (pow(r, -gamma) - 1)
            )
        return self._init_threshold - self._key(sigma * np.log(r))

    def initialize(self, data: np.ndarray, init_threshold: float):
        """
        Run the calibration (initialization) step
        """
        self._init_threshold = init_threshold

        # initial peaks
        self._peaks = self._key(
            data[self._key(data) > self._key(self._init_threshold)]
            - self._init_threshold
        )

        self._logger.debug("Initial threshold : %s", self._init_threshold)
        self._logger.debug("Number of peaks : %s", self.num_peaks)
        self._logger.debug("Grimshaw maximum log-likelihood estimation ... ")

        if self._peaks.size:
            gamma, sigma, ll = self._grimshaw(self._peaks)
            self._extreme_quantile = self._quantile(data.size, gamma, sigma)
            self._logger.debug(
                "gamma = %s, sigma = %s, log-likelihood = %s", gamma, sigma, ll
            )
        else:
            self._extreme_quantile = self._init_threshold
            self._logger.info("Initialized with no peaks")
        self._logger.debug(
            "Extreme quantile (probability = %s): %s",
            self._proba,
            self._extreme_quantile,
        )

    def run(self, datum: float, num: int, with_alarm: bool = True) -> Status:
        """
        Run SPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)
        """
        # If the observed value exceeds the current threshold (alarm case)
        if self._key(datum) > self._key(self._extreme_quantile):
            # if we want to alarm, we put it in the alarm list
            if with_alarm:
                return self.Status.ALARM
            # otherwise we add it in the peaks
            self._peaks = np.append(
                self._peaks, self._key(datum - self._init_threshold)
            )
            # and we update the thresholds

            g, s, _ = self._grimshaw(self._peaks)
            self._extreme_quantile = self._quantile(num + 1, g, s)

        # case where the value exceeds the initial threshold but not the alarm ones
        elif self._key(datum) > self._key(self._init_threshold):
            # we add it in the peaks
            self._peaks = np.append(
                self._peaks, self._key(datum - self._init_threshold)
            )
            # and we update the thresholds

            g, s, _ = self._grimshaw(self._peaks)
            self._extreme_quantile = self._quantile(num + 1, g, s)
        else:
            return self.Status.NORMAL
        return self.Status.ABNORMAL

def _merge_overlapping_intervals(intervals):
    """Assumes (start, end) comparable (datetimes or numbers)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def _classify_points_against_windows(points, windows):
    """
    points: list/array of comparable values (e.g., timestamps or integers)
    windows: list of (start, end), inclusive on both ends.
    Returns: boolean mask 'inside' of same length as points.
    """
    if not windows or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    windows = _merge_overlapping_intervals(windows)
    inside = np.zeros(len(points), dtype=bool)
    # Two-pointer scan (windows and points both sorted)
    order = np.argsort(points)
    pts_sorted = np.asarray(points)[order]
    wi = 0
    for pi, p in enumerate(pts_sorted):
        while wi < len(windows) and p > windows[wi][1]:
            wi += 1
        if wi < len(windows) and windows[wi][0] <= p <= windows[wi][1]:
            inside[order[pi]] = True
    return inside



class SPOTBase(ABC):
    """
    The base class for the SPOT algorithm with data management
    """

    # colors for plot
    DEEP_SAFFRON = "#FF9933"
    AIR_FORCE_BLUE = "#5D8AA8"
    _plot_keys = ()

    def __init__(self, logging_level: int = logging.WARNING):
        self._data: np.ndarray = None
        self._init_data: np.ndarray = None
        self._num: int = 0

        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._logger.setLevel(level=logging_level)

    def summary(self) -> dict:
        """
        Summar running status
        """
        report = {
            "name": "Streaming Peaks-Over-Threshold Object",
        }
        if self._data is not None:
            report["Data imported"] = "Yes"
            report["#(initialization values)"] = self._init_data.size
            report["#(stream values)"] = self._data.size
        else:
            report["Data imported"] = "No"
            return report

        if self._num == 0:
            report["Algorithm initialized"] = "No"
        else:
            report["Algorithm initialized"] = "Yes"
            rest = self._num - self._init_data.size
            if rest > 0:
                report["Algorithm run"] = "Yes"
                report["#(observations)"] = f"{rest} ({100 * rest / self._num:.2f} %%)"
            else:
                report["Algorithm run"] = "No"
        return report

    def __str__(self):
        return json.dumps(self.summary(), indent=2, ensure_ascii=False)

    def fit(
        self,
        init_data: Union[np.ndarray, pd.Series, list, int, float],
        data: Union[np.ndarray, pd.Series, list],
    ):
        """
        Import data to SPOT object

        Parameters:
            init_data: initial batch to calibrate the algorithm
            data: data for the run
        """
        if isinstance(data, list):
            self._data = np.array(data)
        elif isinstance(data, np.ndarray):
            self._data = data
        elif isinstance(data, pd.Series):
            self._data = data.values
        else:
            self._logger.warning("This data format (%s) is not supported", type(data))
            return

        if isinstance(init_data, list):
            self._init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self._init_data = init_data
        elif isinstance(init_data, pd.Series):
            self._init_data = init_data.values
        elif isinstance(init_data, int):
            self._init_data = self._data[:init_data]
            self._data = self._data[init_data:]
        elif isinstance(init_data, float) and (0 < init_data < 1):
            r = int(init_data * data.size)
            self._init_data = self._data[:r]
            self._data = self._data[r:]
        else:
            self._logger.warning("The initial data cannot be set")
            return

    def add(self, data: Union[np.ndarray, pd.Series, list]):
        """
        This function allows to append data to the already fitted data

        Parameters:
            data: data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values
        elif not isinstance(data, np.ndarray):
            self._logger.warning("This data format (%s) is not supported", type(data))
            return

        self._data = np.append(self._data, data)

    def initialize(self, level: float = 0.98):
        """
        Run the calibration (initialization) step

        Parameters:
            level: Probability associated with the initial threshold t (default 0.98)
        """
        raise NotImplementedError

    def run(self, with_alarm: bool = True) -> dict:
        """
        Run SPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)
        """
        raise NotImplementedError

    def plot(
        self,
        run_results: dict,
        with_alarm: bool = True,
        title: str = None,
        label: str = None,
	figsize: tuple = (8, 4),
	show: bool = True,
        show_legend: bool = True,
        *,
        # NEW: styling knobs
        data_color: str = "steelblue",
        threshold_color: str = "black",
        true_alarm_color: str = "green",
        false_alarm_color: str = "red",
        shade_color: str = "tab:red",
        shade_alpha: float = 0.15,
        # NEW: context for x-axis
        ts_index: Union[pd.Index, List, np.ndarray, None] = None,
        anomaly_windows: List[Tuple[Union[pd.Timestamp, float, int],
                                    Union[pd.Timestamp, float, int]]] = None,
        # If residuals present, you may want ylabel default to "Z-score"
    ):
        """
        Plot residuals (or raw series) + dynamic thresholds.
        - Data color is controllable (data_color).
        - Threshold forced to 'threshold_color' (default black).
        - Shaded anomaly windows (anomaly_windows).
        - Alarms in green if they fall inside a shaded window, else red.

        Parameters:
          ts_index: Optional index for the x-axis. If provided and datetime-like,
                    windows should be in the same datetime space. If not provided,
                    integer time index is used and anomaly_windows should be in 'index' units.
          anomaly_windows: List of (start, end) intervals to shade.
                           Use your align/merge helpers to produce these in the same coordinate system as ts_index.
        """
        residuals = run_results.get("residuals", None)
        thresholds = run_results.get("thresholds", [])
        alarms = run_results.get("alarms", [])

        # ---- X-axis determination
        full_length = len(self._init_data) + len(self._data)
        if ts_index is not None:
            x = pd.Index(ts_index)
            if len(x) != full_length:
                raise ValueError(
                    f"ts_index length ({len(x)}) must match full series length ({full_length})."
                )
            x_is_datetime = np.issubdtype(x.dtype, np.datetime64) or isinstance(x, pd.DatetimeIndex)
        else:
            x = np.arange(full_length)
            x_is_datetime = False

        # ---- Compose series to plot
        if residuals is not None:
            y = np.asarray(residuals)
            y_label_default = "Z-score"
        else:
            y = np.concatenate([self._init_data, self._data])
            y_label_default = "Raw Value"

        # ---- Figure/Axes
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        # ---- Shade anomaly windows (behind all curves)
        if anomaly_windows:
            # Ensure compatible types with x:
            if x_is_datetime:
                # Expect pandas Timestamps or datetimes
                aw = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in anomaly_windows]
            else:
                # Expect numeric windows in the same "index space" as x
                aw = [(float(s), float(e)) for s, e in anomaly_windows]
            aw = _merge_overlapping_intervals(aw)
            for s, e in aw:
                ax.axvspan(s, e, facecolor=shade_color, alpha=shade_alpha, lw=0, zorder=0)

        # ---- Plot the data series
        ax.plot(x, y, label="Residuals" if residuals is not None else "Raw series",
                color=data_color, zorder=2)

        # ---- Plot threshold in black (dashed), aligned to stream part only
        if thresholds:
            start_idx = len(self._init_data)
            thr_x = x[start_idx : start_idx + len(thresholds)]
            ax.plot(thr_x, thresholds, "--", label="Threshold",
                    color=threshold_color, zorder=3)

        # ---- Plot alarms: split into true/false by window membership
        if with_alarm and alarms:
            # alarm "positions" in x-coordinates and "values" in y
            alarm_stream_idx = np.asarray(alarms)
            alarm_x = x[len(self._init_data) + alarm_stream_idx]
            alarm_y = np.array([
                y[len(self._init_data) + a] for a in alarm_stream_idx
            ])

            # classify by windows
            if anomaly_windows:
                inside_mask = _classify_points_against_windows(alarm_x, aw)
            else:
                inside_mask = np.zeros_like(alarm_x, dtype=bool)

            # scatter (true/false) with distinct colors
            if inside_mask.any():
                ax.scatter(alarm_x[inside_mask], alarm_y[inside_mask],
                           s=32, color=true_alarm_color, label="Alarms (true)", zorder=4)
            if (~inside_mask).any():
                ax.scatter(alarm_x[~inside_mask], alarm_y[~inside_mask],
                           s=32, color=false_alarm_color, label="Alarms (false)", zorder=4)

        # ---- Labels, title, legend
        ax.set_xlabel("Time" if x_is_datetime else "Time Index")
        ax.set_ylabel(label or y_label_default)
        ax.set_title(title or "SPOT Residuals and Dynamic Thresholds", loc="left")

        ax.grid(True, linestyle="--", alpha=0.35)
        
        if show_legend:
            ax.legend(frameon=False)

        if show:
            plt.show()
        return fig, ax

   

class SPOT(SPOTBase):
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
    """

    _plot_keys = ("thresholds",)

    def __init__(
        self, q: float = 1e-4, n_points: int = 10, logging_level: int = logging.WARNING
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
        """
        super().__init__(logging_level=logging_level)
        self._ev = ExtremeValue(q=q, n_points=n_points, logging_level=logging_level)

    def summary(self) -> dict:
        report = super().summary()
        report["Extreme Value"] = self._ev.summary()
        return report

    def initialize(self, level: float = 0.98):
        data = self._init_data
        level = level - np.floor(level)

        # t is fixed for the whole algorithm
        init_threshold = sorted(data)[int(level * data.size)]
        self._ev.initialize(data=data, init_threshold=init_threshold)
        self._num = data.size

    def run(self, with_alarm: bool = True) -> dict:
        """
        Run SPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)

        Returns:
            a dict:
                keys : 'thresholds' and 'alarms'

                'thresholds' contains the extreme quantiles and 'alarms' contains
                the indexes of the values which have triggered alarms

        """
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # list of the thresholds
        thresholds = []
        alarms = []
        # Loop over the stream
        for i, datum in enumerate(self._data):
            if (
                self._ev.run(datum, self._num, with_alarm=with_alarm)
                == ExtremeValue.Status.ALARM
            ):
                alarms.append(i)
            else:
                self._num += 1

            thresholds.append(self._ev.extreme_quantile)  # thresholds record

        return {"thresholds": thresholds, "alarms": alarms}


class biSPOT(SPOTBase):
    """
    This class allows to run biSPOT algorithm on univariate dataset (upper and lower bounds)
    """

    _plot_keys = ("upper_thresholds", "lower_thresholds")

    def __init__(
        self, q: float = 1e-4, n_points: int = 10, logging_level: int = logging.WARNING
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
        """
        super().__init__(logging_level=logging_level)
        self._ev = {
            "upper": ExtremeValue(
                q=q, n_points=n_points, key=_asc_key, logging_level=logging_level
            ),
            "lower": ExtremeValue(
                q=q, n_points=n_points, key=_desc_key, logging_level=logging_level
            ),
        }

    def summary(self) -> dict:
        report = super().summary()
        report["upper Extreme Value"] = self._ev["upper"].summary()
        report["lower Extreme Value"] = self._ev["lower"].summary()
        return report

    def initialize(self, level: float = 0.98):
        data = self._init_data
        level = level - np.floor(level)

        _data = sorted(data)
        # t is fixed for the whole algorithm
        init_thresholds = {
            "upper": _data[int(level * data.size)],
            "lower": _data[int((1 - level) * data.size)],
        }
        for key, ev in self._ev.items():
            ev.initialize(data=data, init_threshold=init_thresholds[key])
        self._num = data.size

    def run(self, with_alarm: bool = True) -> dict:
        """
        Run biSPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)

        Returns:
            a dict:
                keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'

                '*_thresholds' contains the extreme quantiles and 'alarms' contains
                the indexes of the values which have triggered alarms

        """
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # list of the thresholds
        thresholds = {key: [] for key in self._ev}
        alarms = []
        # Loop over the stream
        for i, datum in enumerate(self._data):
            ret = {
                ev.run(datum, self._num, with_alarm=with_alarm)
                for ev in self._ev.values()
            }
            if ExtremeValue.Status.ALARM in ret:
                alarms.append(i)
            else:
                self._num += 1
            for key, ev in self._ev.items():
                thresholds[key].append(ev.extreme_quantile)

        return {
            "upper_thresholds": thresholds["upper"],
            "lower_thresholds": thresholds["lower"],
            "alarms": alarms,
        }


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Moving average of the given data
    """
    mean: List[float] = []
    accumulation: float = data[:window].sum()
    mean.append(accumulation / window)
    for i in range(window, len(data)):
        accumulation = accumulation - data[i - window] + data[i]
        mean.append(accumulation / window)
    return np.array(mean)


class dSPOT(SPOT):
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)
    """

    def __init__(
        self,
        q: float = 1e-4,
        n_points: int = 10,
        depth: int = 10,
        logging_level: int = logging.WARNING,
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
            depth: Number of observations to compute the moving average
        """
        super().__init__(q=q, n_points=n_points, logging_level=logging_level)
        self._depth = depth

    def initialize(self, level: float = 0.98):
        data: np.ndarray = (
            self._init_data[self._depth :]
            - moving_average(self._init_data, self._depth)[:-1]
        )
        level = level - np.floor(level)

        # t is fixed for the whole algorithm
        init_threshold = sorted(data)[int(level * data.size)]
        self._ev.initialize(data=data, init_threshold=init_threshold)
        self._num = data.size

    def run(self, with_alarm: bool = True) -> dict:
        if self._num > self._init_data.size:
            self._logger.warning("Already run; re-initialize before running again")
            return {}

        window = self._init_data[-self._depth:]      # last 'depth' points for baseline
        thresholds = []
        alarms = []
        self._residuals = []                        # new list to store residuals

        for i, datum in enumerate(self._data):
            mean = window.mean()
            residual = datum - mean                 # compute residual (detrended value)
            # Run EVT on the residual:
            status = self._ev.run(residual, self._num, with_alarm=with_alarm)
            if status == ExtremeValue.Status.ALARM:
                alarms.append(i)
            else:
                self._num += 1
                window = np.append(window[1:], datum)  # update moving window with new datum
            thresholds.append(self._ev.extreme_quantile)   # threshold in residual scale (no mean added)
            self._residuals.append(residual)               # save residual
        # Pad initial segment with NaNs so residuals align with full series length:
        init_pad = [np.nan] * len(self._init_data)
        all_residuals = init_pad + self._residuals

        return {"thresholds": thresholds, "alarms": alarms, "residuals": all_residuals}

class OnlinePreprocessor:
    """
    Online preprocessing for DSPOT:
    - EWMA trend filtering
    - Rolling standard deviation normalization
    - Seasonal differencing (on normalized residuals)
    """

    def __init__(self, P=288, alpha=0.05):
        self.P = P  # Seasonal lag
        self.alpha = alpha  # EWMA smoothing factor
        self.T_prev = None  # Previous EWMA value
        self.resid_window = deque(maxlen=P)  # Residual window for rolling std
        self.z_hist = deque(maxlen=P)  # Store normalized residuals for seasonal diff
        self.std_eps = 1e-8  # Avoid division by zero

    def update(self, x_t):
        # Step 1: EWMA detrending
        if self.T_prev is None:
            T_t = x_t
        else:
            T_t = self.alpha * x_t + (1 - self.alpha) * self.T_prev
        self.T_prev = T_t

        resid = x_t - T_t
        self.resid_window.append(resid)

        # Step 2: Variance normalization (rolling z-score)
        if len(self.resid_window) >= 5:
            mean = np.mean(self.resid_window)
            std = np.std(self.resid_window, ddof=1)
        else:
            mean, std = 0.0, None

        if std is None or std < self.std_eps:
            z_t = None
        else:
            z_t = (resid - mean) / std

        if z_t is None:
            self.z_hist.append(0.0)  # Append dummy value to maintain sync
            return None

        # Step 3: Seasonal differencing (on normalized residuals)
        if len(self.z_hist) == self.P:
            z_diff = z_t - self.z_hist[0]
        else:
            z_diff = None

        self.z_hist.append(z_t)

        return z_diff


class ModifiedSPOT(SPOT):
    """
    SPOT + full preprocessing: EWMA + variance normalization + seasonal differencing.
    """

    def __init__(self, q=1e-4, n_points=10, logging_level=30,
                 seasonal_period=288, ewma_alpha=0.05):
        super().__init__(q=q, n_points=n_points, logging_level=logging_level)
        self._preproc = OnlinePreprocessor(P=seasonal_period, alpha=ewma_alpha)
        self._residuals = []

    def initialize(self, level: float = 0.98):
        """
        Initialize EVT model using fully preprocessed z-scores (enforcing i.i.d.).
        """
        data = []

        for x in self._init_data:
            z_t = self._preproc.update(x)
            if z_t is not None:
                data.append(z_t)

        if len(data) < 10:
            raise ValueError("Too few valid preprocessed points. Increase _init_data size.")

        data = np.array(data)
        init_threshold = np.quantile(data, level)
        self._ev.initialize(data=data, init_threshold=init_threshold)
        self._num = len(data)

    def run(self, with_alarm=True):
        """
        Run SPOT with online fully preprocessed input.
        """
        thresholds = []
        alarms = []
        self._residuals = []

        for i, datum in enumerate(self._data):
            z_t = self._preproc.update(datum)

            if z_t is None:
                thresholds.append(np.nan)
                self._residuals.append(np.nan)
                continue

            self._residuals.append(z_t)

            status = self._ev.run(z_t, self._num, with_alarm=with_alarm)

            if status == ExtremeValue.Status.ALARM:
                alarms.append(i)
            else:
                self._num += 1

            thresholds.append(self._ev.extreme_quantile)

        # Pad for alignment with initialization
        init_pad = [np.nan] * len(self._init_data)
        all_residuals = init_pad + self._residuals

        return {
            "thresholds": thresholds,
            "alarms": alarms,
            "residuals": all_residuals
        }

class bidSPOT(biSPOT):
    """
    This class allows to run biDSPOT algorithm on univariate dataset (upper and lower bounds)
    """

    def __init__(
        self,
        q: float = 1e-4,
        n_points: int = 10,
        depth: int = 10,
        logging_level: int = logging.WARNING,
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
            depth: Number of observations to compute the moving average
        """
        super().__init__(q=q, n_points=n_points, logging_level=logging_level)
        self._depth = depth

    def initialize(self, level: float = 0.98):
        data: np.ndarray = (
            self._init_data[self._depth :]
            - moving_average(self._init_data, self._depth)[:-1]
        )
        level = level - np.floor(level)

        _data = sorted(data)
        # t is fixed for the whole algorithm
        init_thresholds = {
            "upper": _data[int(level * data.size)],
            "lower": _data[int((1 - level) * data.size)],
        }
        for key, ev in self._ev.items():
            ev.initialize(data=data, init_threshold=init_thresholds[key])
        self._num = data.size

    def run(self, with_alarm: bool = True):
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # actual normal window
        window: np.ndarray = self._init_data[-self._depth :]

        # list of the thresholds
        thresholds = {key: [] for key in self._ev}
        alarms = []
        # Loop over the stream
        for i, datum in enumerate(self._data):
            mean = window.mean()
            ret = {
                ev.run(datum - mean, self._num, with_alarm=with_alarm)
                for ev in self._ev.values()
            }
            if ExtremeValue.Status.ALARM in ret:
                alarms.append(i)
            else:
                self._num += 1
                window = np.append(window[1:], datum)
            for key, ev in self._ev.items():
                thresholds[key].append(ev.extreme_quantile + mean)

        return {
            "upper_thresholds": thresholds["upper"],
            "lower_thresholds": thresholds["lower"],
            "alarms": alarms,
        }
