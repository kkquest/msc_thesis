"""
Extension of SPOT algorithms that returns dynamic thresholds in the
original data scale instead of the residual scale.

This module reuses the existing implementation from ``spot_copy.py`` to
perform anomaly detection using the dSPOT and ModifiedSPOT algorithms,
but it augments the streaming logic so that the returned thresholds
reflect the same units as the input time series.  

Note that the ``plot`` method inherited from ``SPOTBase`` is used
unchanged; because the returned dictionary from ``run`` does not
contain a ``residuals`` entry, the plot function automatically
selects the raw series for the y‑axis and draws the provided
thresholds on top of it.
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from spot_copy import ExtremeValue, SPOT, dSPOT, ModifiedSPOT, OnlinePreprocessor


class PreprocessorWithInfo(OnlinePreprocessor):
    """
    Extension of :class:`OnlinePreprocessor` that exposes additional
    intermediate values when updating with a new observation.

    The base preprocessor applies three successive steps to each
    observation: an EWMA detrending, a rolling z‑score normalisation
    and a seasonal differencing.  Only the final differenced z‑score
    is returned from the standard :meth:`update` method, making it
    impossible to map an extreme value threshold back to the original
    data units.  This subclass provides :meth:`update_with_info` which
    returns both the differenced z‑score and the auxiliary values
    necessary to invert the transformation:

    * ``T_t``: the exponential moving average at the current time
    * ``mean`` and ``std``: the mean and standard deviation of the
      residual window used for variance normalisation (``std`` may be
      ``None`` when there are too few samples)
    * ``z_prev``: the z‑score from one seasonal period ago; this is
      used to reconstruct the current z‑score from the differenced
      value.

    This method otherwise mirrors the logic in ``OnlinePreprocessor``.
    """

    def update_with_info(
        self, x_t: float
    ) -> Tuple[Optional[float], float, float, Optional[float], Optional[float]]:
        """
        Process a new observation and return detailed preprocessing state.

        Parameters
        ----------
        x_t : float
            The latest observation from the time series.

        Returns
        -------
        tuple
            A 5‑tuple ``(z_diff, T_t, mean, std, z_prev)`` where

            * ``z_diff`` is the seasonal differenced z‑score.  It will
              be ``None`` until enough history has been accumulated
              (when the internal ``z_hist`` deque is shorter than
              ``P``).
            * ``T_t`` is the EWMA trend estimate for ``x_t``.
            * ``mean`` and ``std`` are the mean and sample standard
              deviation of the residual window used for z‑score
              normalisation.  ``std`` may be ``None`` if there are
              fewer than five elements in the window; in that case
              ``z_diff`` will also be ``None``.
            * ``z_prev`` is the historic z‑score from one seasonal
              period prior (i.e. the element removed from the start of
              ``z_hist``).  It is only defined when ``z_diff`` is not
              ``None``.
        """
        # Step 1: EWMA detrending
        if self.T_prev is None:
            T_t = x_t
        else:
            T_t = self.alpha * x_t + (1 - self.alpha) * self.T_prev
        self.T_prev = T_t

        resid = x_t - T_t
        self.resid_window.append(resid)

        # Step 2: Rolling statistics for z‑score
        if len(self.resid_window) >= 5:
            mean = float(np.mean(self.resid_window))
            std = float(np.std(self.resid_window, ddof=1))
        else:
            mean = 0.0
            std = None

        if std is None or std < self.std_eps:
            # Not enough statistics to compute a meaningful z‑score
            z_t = None
        else:
            z_t = (resid - mean) / std

        # Even when no z‑score is available, maintain z_hist alignment
        if z_t is None:
            self.z_hist.append(0.0)
            return None, T_t, mean, std, None

        # Step 3: Seasonal differencing (on z‑scores)
        if len(self.z_hist) == self.P:
            # capture previous z for inversion before appending new one
            z_prev = self.z_hist[0]
            z_diff = z_t - z_prev
        else:
            z_prev = None
            z_diff = None

        # append current z_t into history, automatically discarding the oldest
        self.z_hist.append(z_t)

        return z_diff, T_t, mean, std, z_prev


class dSPOTOriginal(dSPOT):
    """
    ``dSPOT`` variant that returns adaptive thresholds in the original
    data scale.

    The detection logic remains identical to that of
    :class:`~spot_copy.dSPOT`: a moving average baseline is used to
    compute residuals, an EVT model is fitted on the initial residuals
    and updated online, and alarms are raised whenever a new residual
    exceeds the dynamic threshold.  The difference lies in the
    reporting: instead of returning thresholds in the residual domain,
    each threshold is shifted by the baseline mean used to compute
    that residual.  This allows you to plot the threshold directly on
    the raw time series and interpret it in the same units as the
    input data.
    """

    def run(self, with_alarm: bool = True) -> dict:  # type: ignore[override]
        """
        Run dSPOT on the streaming data and return original‑scale thresholds.

        Parameters
        ----------
        with_alarm : bool, optional
            If ``False``, detected extreme values will be treated as
            normal for the purpose of updating the EVT model (no
            alarms will be generated).  Default is ``True``.

        Returns
        -------
        dict
            A dictionary with two entries:

            ``'thresholds'``
                A list of floats (or ``NaN``) representing the dynamic
                threshold at each step, expressed in the same scale as
                the original data.  The list is aligned with the
                streaming data (``self._data``); thus element ``i``
                corresponds to ``self._data[i]``.
            ``'alarms'``
                A list of integer indices into ``self._data`` marking
                observations that triggered an alarm.  Note that the
                indices are relative to the start of the streaming
                portion (i.e. not including the initial batch used for
                calibration).
        """
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # Baseline window for moving average; shape (depth,)
        window: np.ndarray = self._init_data[-self._depth :]
        thresholds_orig: List[float] = []
        alarms: List[int] = []

        # Iterate over the streaming data
        for i, datum in enumerate(self._data):
            mean = float(window.mean())
            residual = datum - mean
            # Pass the residual to the EVT model
            status = self._ev.run(residual, self._num, with_alarm=with_alarm)
            if status == ExtremeValue.Status.ALARM:
                alarms.append(i)
            else:
                # Only update the sample count and moving window when not alarming
                self._num += 1
                window = np.append(window[1:], datum)
            # The EVT model returns a threshold on residuals; shift back to original scale
            thr_residual = self._ev.extreme_quantile
            # When no peaks have been observed yet ``thr_residual`` equals the initial threshold
            # computed from residuals; it is still appropriate to shift by the baseline mean.
            thresholds_orig.append(thr_residual + mean)

        return {"thresholds": thresholds_orig, "alarms": alarms}


class ModifiedSPOTOriginal(ModifiedSPOT):
    """
    ``ModifiedSPOT`` variant that reports thresholds in the original data
    domain.

    This class wraps the fully pre‑processed anomaly detection pipeline
    (EWMA detrending, rolling z‑score normalisation and seasonal
    differencing) defined in :class:`~spot_copy.ModifiedSPOT`.  It
    replaces the internal preprocessor with a variant that exposes
    intermediate statistics so that the dynamic threshold on the
    differenced z‑scores can be mapped back to the raw observation
    scale.  The detection itself remains identical to
    ``ModifiedSPOT``: the EVT model operates on the differenced
    z‑scores produced by the preprocessor, and the adaptation of the
    model is unchanged.
    """

    def __init__(
        self,
        q: float = 1e-4,
        n_points: int = 10,
        logging_level: int = 30,
        seasonal_period: int = 288,
        ewma_alpha: float = 0.05,
    ):
        # Initialise the underlying SPOT machinery
        super().__init__(
            q=q,
            n_points=n_points,
            logging_level=logging_level,
            seasonal_period=seasonal_period,
            ewma_alpha=ewma_alpha,
        )
        # Override the preprocessor with our detailed version
        self._preproc = PreprocessorWithInfo(P=seasonal_period, alpha=ewma_alpha)
        # We won't rely on _residuals for plotting, so we leave it empty
        self._residuals: List[float] = []

    def initialize(self, level: float = 0.98):  # type: ignore[override]
        """
        Initialise the EVT model using fully pre‑processed differenced z‑scores.

        Parameters
        ----------
        level : float, optional
            The initial quantile level (between 0 and 1) used to set
            the EVT threshold.  Default is ``0.98``.

        Raises
        ------
        ValueError
            If fewer than 10 valid differenced z‑scores are obtained
            from the initial data.  This typically indicates that
            ``_init_data`` is too short relative to the seasonal period
            or that the period is too large.
        """
        # Reset preprocessor state
        self._preproc = PreprocessorWithInfo(
            P=self._preproc.P, alpha=self._preproc.alpha
        )

        data: List[float] = []
        # Accumulate differenced z‑scores from the initial batch
        for x in self._init_data:
            z_diff, _, _, _, _ = self._preproc.update_with_info(x)
            if z_diff is not None:
                data.append(float(z_diff))

        if len(data) < 10:
            raise ValueError(
                "Too few valid preprocessed points. Increase _init_data size or reduce seasonal_period."
            )

        arr = np.array(data)
        # Compute initial threshold on the differenced z‑scores
        init_threshold = float(np.quantile(arr, level - np.floor(level)))
        # Initialise EVT with the differenced z‑scores
        self._ev.initialize(data=arr, init_threshold=init_threshold)
        # Number of processed points corresponds to the number of valid z_diff
        self._num = len(arr)

    def run(self, with_alarm: bool = True):  # type: ignore[override]
        """
        Run ModifiedSPOT with detailed preprocessing and return thresholds
        on the original scale.

        Parameters
        ----------
        with_alarm : bool, optional
            Whether to emit alarms when an extreme value is detected
            (default is ``True``).  If ``False`` the EVT model will
            still update but no indices will be stored in ``alarms``.

        Returns
        -------
        dict
            A dictionary with two keys:

            ``'thresholds'``
                A list containing the dynamic thresholds expressed
                directly in the same scale as the input observations.
                Positions where no threshold could be computed (before
                the seasonal period is reached) are filled with
                ``NaN``.
            ``'alarms'``
                A list of indices (relative to the start of
                ``self._data``) where alarms were raised.
        """
        thresholds_orig: List[float] = []
        alarms: List[int] = []

        for i, datum in enumerate(self._data):
            # Perform preprocessing with detailed output
            z_diff, T_t, mean, std, z_prev = self._preproc.update_with_info(datum)

            if z_diff is None:
                # No valid differenced z‑score yet: propagate NaN threshold
                thresholds_orig.append(float('nan'))
                continue

            # Update EVT on the differenced z‑score
            status = self._ev.run(z_diff, self._num, with_alarm=with_alarm)
            if status == ExtremeValue.Status.ALARM and with_alarm:
                alarms.append(i)
            else:
                # Only increment counter when not treating as alarm; matches base class behaviour
                self._num += 1

            # Latest threshold on z_diff scale
            thr_z_diff = self._ev.extreme_quantile
            # Reconstruct corresponding z‑score from the differenced value
            # When z_prev is None, z_diff would also be None; guard for completeness
            if z_prev is None or std is None:
                thresholds_orig.append(float('nan'))
                continue
            z_est = thr_z_diff + z_prev
            # Map back to the residual domain
            resid_threshold = z_est * std + mean
            # Finally shift by the EWMA trend estimate to obtain original scale
            thresholds_orig.append(resid_threshold + T_t)

        return {"thresholds": thresholds_orig, "alarms": alarms}