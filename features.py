import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, stft


def fs_10(signal):
    minimum = np.min(signal)
    maximum = np.max(signal)
    mean = np.mean(signal)
    std_dev = np.std(signal)
    variance = np.var(signal)
    skewness_val = skew(signal)
    kurtosis_val = kurtosis(signal)
    median = np.median(signal)
    zero_crossings = len(np.where(np.diff(np.sign(signal)))[0])
    mean_energy = np.mean(signal ** 2)

    return [minimum, maximum, mean, std_dev, variance, skewness_val, kurtosis_val, median, zero_crossings, mean_energy]


def fs_14(signal):
    moments = [np.mean((signal - np.mean(signal)) ** i) for i in range(3, 7)]
    return fs_10(signal) + moments


def fs_18(signal):
    mean_abs_val = np.mean(np.abs(signal))
    max_scatter_diff = np.max(np.diff(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    mean_abs_deviation = np.mean(np.abs(signal - np.mean(signal)))

    return fs_14(signal) + [mean_abs_val, max_scatter_diff, rms, mean_abs_deviation]


def fs_22(signal):
    std_dev = np.std(signal)
    first_diff = np.mean(np.diff(signal))
    second_diff = np.mean(np.diff(np.diff(signal)))
    first_diff_div_std = first_diff / std_dev
    second_diff_div_std = second_diff / std_dev

    return fs_18(signal) + [first_diff, second_diff, first_diff_div_std, second_diff_div_std]


def low_pass_filter(signal, sample_rate, cutoff):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def extract_basic_features(signal):
    # Basic features: mean, variance, max, min, max-min
    mean_val = np.mean(signal)
    variance_val = np.var(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    max_min_diff = max_val - min_val
    return [mean_val, variance_val, max_val, min_val, max_min_diff]


def fs_low_pass(window, sample_rate):
    features = []
    cutoffs = [0.4, 0.45, 0.5, 1.0, 3.0]

    for cutoff in cutoffs:
        filtered_signal = low_pass_filter(window, sample_rate, cutoff)

        basic_features = extract_basic_features(filtered_signal)
        first_diff = np.diff(filtered_signal)
        second_diff = np.diff(first_diff)

        basic_features_first_diff = extract_basic_features(first_diff)
        basic_features_second_diff = extract_basic_features(second_diff)

        features.extend(basic_features)
        features.extend(basic_features_first_diff)
        features.extend(basic_features_second_diff)

    return features


def get_ecg_features(ecg_data, sample_rate=128, n_fft=256, hop_length=128):
    f, t, Zxx = stft(ecg_data, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    return Zxx.T  # (time, freq)


def get_other_physiological_features(signal, sample_rate=128, sliding_window=1280, feature_size=75):
    # Validate feature size
    if feature_size not in [10, 14, 18, 22, 75]:
        raise ValueError("Invalid feature size. Must be one of [10, 14, 18, 22, 75].")

    # Extract windows
    num_windows = len(signal) - sliding_window + 1
    windows = [signal[i:i + sliding_window] for i in range(num_windows)]

    # Extract features for each window
    features = []
    for window in windows:
        if feature_size == 10:
            features.append(fs_10(window))
        elif feature_size == 14:
            features.append(fs_14(window))
        elif feature_size == 18:
            features.append(fs_18(window))
        elif feature_size == 22:
            features.append(fs_22(window))
        elif feature_size == 75:
            features.append(fs_low_pass(window, sample_rate))

    return np.array(features)
