import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

class AdaptiveForwardGaussianSmoother:
    def __init__(self, min_sigma=0.5, max_sigma=3.0, base_window_size=5, sensitivity=1.0):
        """
        Adaptive causal Gaussian smoother.
        Adjusts sigma and window size based on recent data volatility,
        then applies a causal Gaussian-weighted average to produce the smoothed value.

        Parameters
        ----------
        min_sigma        : minimum std-dev, used for stable data
        max_sigma        : maximum std-dev, used for volatile data
        base_window_size : base window size
        sensitivity      : volatility sensitivity (0.1-5); higher = more reactive
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.base_window_size = base_window_size
        self.sensitivity = sensitivity
        self.history = []  # stores historical data for adaptive computation

    def compute_local_variance(self, new_value, lookback=10):
        """Compute local variance to assess data volatility."""
        if len(self.history) < lookback:
            return 0

        recent_data = self.history[-lookback:]
        recent_data.append(new_value)
        return np.var(recent_data)

    def adapt_parameters(self, current_value):
        """Adaptively adjust sigma and window size based on data volatility."""
        # Compute local volatility
        local_variance = self.compute_local_variance(current_value)

        # Normalise volatility to [0, 1] using an empirical maximum variance
        max_expected_variance = 100.0
        normalized_variance = min(local_variance / max_expected_variance, 1.0)

        # Higher volatility → smaller sigma (preserve detail)
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * (normalized_variance ** self.sensitivity)

        # Dynamic window size — Gaussian coverage strategy:
        #   ±1σ covers ~68% of data
        #   ±2σ covers ~95% of data  (adopted here)
        #   ±3σ covers ~99.7% of data
        window_size = max(3, min(self.base_window_size, int(2 * np.ceil(2 * sigma) + 1)))  # +1 to include centre

        return sigma, window_size

    def create_causal_gaussian_kernel(self, sigma, window_size):
        """Create a causal (forward-looking only) Gaussian kernel."""
        # Use only current and past positions
        x = np.arange(0, window_size)
        kernel = stats.norm.pdf(x, loc=0, scale=sigma)

        # Normalise so weights sum to 1
        kernel = kernel / np.sum(kernel)
        return kernel

    def smooth_value(self, new_value):
        """Smooth a single new data point."""
        # Add to history
        self.history.append(new_value)

        # Adapt parameters
        sigma, window_size = self.adapt_parameters(new_value)

        # Available data window
        available_data = self.history[-window_size:]

        # Fall back to a smaller window if insufficient data
        if len(available_data) < window_size:
            actual_window = len(available_data)
            sigma_adjusted = sigma * (actual_window / window_size)  # scale sigma proportionally
            kernel = self.create_causal_gaussian_kernel(sigma_adjusted, actual_window)
        else:
            kernel = self.create_causal_gaussian_kernel(sigma, window_size)

        # Apply convolution (weighted average)
        smoothed_value = np.dot(available_data, kernel[::-1])

        return smoothed_value, sigma, window_size

    def predict_next(self):
        """Predict the next value."""
        if len(self.history) < 3:  # need at least 3 points
            return np.mean(self.history)  # fall back to mean during warm-up

        # Use the last value to compute adaptive parameters
        sigma, window_size = self.adapt_parameters(self.history[-1])
        available_data = self.history[-window_size:]

        if len(available_data) < window_size:
            actual_window = len(available_data)
            sigma_adjusted = sigma * (actual_window / window_size)
            kernel = self.create_causal_gaussian_kernel(sigma_adjusted, actual_window)
        else:
            kernel = self.create_causal_gaussian_kernel(sigma, window_size)

        # Predict next value as weighted average
        predicted_value = np.dot(available_data, kernel[::-1])
        return predicted_value

    def smooth_array(self, array_A):
        """Smooth an entire array."""
        predict_B = np.zeros_like(array_A)  # predicted values
        array_B   = np.zeros_like(array_A)  # actual smoothed values
        sigmas  = np.zeros_like(array_A)
        windows = np.zeros_like(array_A, dtype=int)

        for i, value in enumerate(array_A):
            # For the first few points use progressive smoothing
            if i < 3:
                # Use simple average during warm-up to avoid over-smoothing
                array_B[i] = np.mean(array_A[:i + 1])
                sigmas[i]  = self.min_sigma
                windows[i] = i + 1
                self.history.append(value)
                if i == 2:
                    predict_B[0] = array_B[0]
                    predict_B[1] = array_B[1]
                    predict_B[2] = array_B[2]
                    predict_B[3] = array_B[3]
                else:
                    continue
            else:
                array_B[i], sigmas[i], windows[i] = self.smooth_value(value)
                if i == len(array_A) - 1:
                    print("predict", self.predict_next())
                predict_B[i] = self.predict_next()  # predict_B[i] is the prediction for index i-1

        return predict_B, array_B, sigmas, windows

    def predict_array(self, array_A):
        """"""
        array_B = np.zeros_like(array_A)
        for i, value in enumerate(array_A):
            # For the first few points use progressive smoothing
            array_B[i] = self.predict_next()

        return array_B


# Run test
if __name__ == "__main__":
    df = pd.read_csv("../results/stacks/ajsp.csv")
    array_o = df.iloc[:, 11]
    array_h = df.iloc[:, 12]
    array_l = df.iloc[:, 13]
    array_c = df.iloc[:, 14]
    gs = AdaptiveForwardGaussianSmoother()
    gs_predict, gs_s, *_ = gs.smooth_array(array_c)
    for i in zip(gs_predict, gs_s, array_c):
        print(i)


    # fig, ax1 = plt.subplots(figsize=(10, 6))
    #
    # # First Y-axis (left)
    # color1 = 'tab:blue'
    # ax1.plot(gs_s, color=color1, marker='o', linewidth=0.5)
    # ax1.set_xlabel('d')
    # ax1.set_ylabel('price', color=color1)
    # ax1.tick_params(axis='y', labelcolor=color1)
    #
    # # Second Y-axis (right)
    # # ax2 = ax1.twinx()
    # # color2 = 'tab:red'
    # # ax2.plot(gs_predict, color=color2, marker='s', linestyle='--', linewidth=0.5)
    # # ax2.set_ylabel('y', color=color2)
    # # ax2.tick_params(axis='y', labelcolor=color2)
    #
    # plt.title('analyse')
    # fig.tight_layout()
    # plt.show()
    # min_len = min(len(gs_predict), len(gs_s))
    #
    # combined_df = pd.concat([
    #     gs_predict.iloc[:min_len, 0],
    #     gs_s.iloc[:min_len, 0]
    # ], axis=1)
    # combined_df.to_csv('filled.csv', index=False)
    # pass
