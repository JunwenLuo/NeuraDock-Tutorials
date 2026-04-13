import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import os
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs  
    low = lowcut / nyq  
    high = highcut / nyq  
    b, a = signal.butter(order, [low, high], btype='band')  
    return b, a  

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):  
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  
    y = signal.filtfilt(b, a, data)  
    return y  
def butter_bandstop(lowcut, highcut, fs, order=5):  
    nyq = 0.5 * fs  
    low = lowcut / nyq  
    high = highcut / nyq  
    b, a = signal.butter(order, [low, high], btype='bandstop')  
    return b, a  

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):  
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)  
    y = signal.filtfilt(b, a, data)  
    return y  
# 定义采样频率（请根据您的实际数据设置）
fs = 250# 采样频率，单位Hz

# 定义质量分析函数
def analyze_50Hz_noise(eeg_data, fs):
    f, Pxx = signal.welch(eeg_data, fs, nperseg=min(len(eeg_data), fs*2))
    idx_50Hz = np.where((f >= 49) & (f <= 51))
    power_50Hz = np.sum(Pxx[idx_50Hz])
    total_power = np.sum(Pxx)
    power_50Hz_ratio = power_50Hz / total_power
    return power_50Hz_ratio

def analyze_emg(eeg_data, fs):
    f, Pxx = signal.welch(eeg_data, fs, nperseg=min(len(eeg_data), fs*2))
    idx_emg = np.where((f >= 20) & (f <= 45))
    power_emg = np.sum(Pxx[idx_emg])
    total_power = np.sum(Pxx)
    power_emg_ratio = power_emg / total_power
    return power_emg_ratio

def analyze_baseline_drift(eeg_data, fs):
    eeg_data = eeg_data - np.mean(eeg_data)  # 去除直流偏移
    f, Pxx = signal.welch(eeg_data, fs, nperseg=min(len(eeg_data), fs*2))
    idx_drift = np.where((f >= 0.1) & (f <= 3))
    power_drift = np.sum(Pxx[idx_drift])
    total_power = np.sum(Pxx)
    power_drift_ratio = power_drift / total_power
    return power_drift_ratio

def analyze_abnormal_values(eeg_data):

    num_abnormal_values = len(np.where((eeg_data<=10)|(eeg_data>=90))[0])
    return num_abnormal_values

def analyze_spectral_features(eeg_data, fs):
    f, Pxx = signal.welch(eeg_data, fs, nperseg=min(len(eeg_data), fs*2))
    total_power = np.sum(Pxx)
    bands = {'delta': (0.5, 4),
             'theta': (4, 8),
             'alpha': (9, 11),
             'beta': (13, 30),
             'gamma': (30, 100)}
    relative_powers = {}
    for band in bands:
        fmin, fmax = bands[band]
        idx_band = np.where((f >= fmin) & (f <= fmax))
        power_band = np.sum(Pxx[idx_band])
        relative_powers[band] = power_band / total_power
    return relative_powers

# 定义函数：计算质量指标
def calculate_quality_metrics(eeg_data, fs):
    n_trials, n_channels, _ = eeg_data.shape
    metrics_list = []
    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            channel_data = eeg_data[trial_idx, ch_idx, :]

            # 计算质量指标
            power_50Hz_ratio = analyze_50Hz_noise(channel_data, fs)
            power_emg_ratio = analyze_emg(channel_data, fs)
            power_drift_ratio = analyze_baseline_drift(channel_data, fs)
            num_abnormal_values = analyze_abnormal_values(channel_data)
            relative_powers = analyze_spectral_features(channel_data, fs)
            alpha_power_ratio = relative_powers['alpha']

            # 存储质量指标
            metrics_list.append({
                'trial': trial_idx,
                'channel': ch_idx,
                '50Hz power': power_50Hz_ratio,
                'muscle artifacts': power_emg_ratio,
                'baseline drift': power_drift_ratio,
                'peak': num_abnormal_values,
                'Alpha amplitude': alpha_power_ratio
            })
    df_metrics = pd.DataFrame(metrics_list)
    return df_metrics

# 定义函数：绘制箱线图
def plot_boxplots(df_metrics, thresholds=None, title='EEG 质量指标箱线图'):
    quality_metrics = ['50Hz power', 'muscle artifacts', 'baseline drift', 'peak','Alpha amplitude']
    plt.figure(figsize=(12, 6))
    df_metrics[quality_metrics].boxplot()
    plt.title(title)
    plt.ylabel('值')
    plt.xticks(rotation=45)
    plt.grid(True)
    

    # 如果提供了阈值，则在箱线图中添加红色阈值线
    if thresholds is not None:
        ax = plt.gca()
        for idx, metric in enumerate(quality_metrics):
            threshold = thresholds[metric]
            x_position = idx + 1  # 箱线图中箱的位置，从1开始
            if metric != 'Delta amplitude':
                # 阈值在 y 轴的上方，需要添加水平线
                ax.axhline(y=threshold, color='red', linestyle='--', xmin=(x_position - 0.5)/len(quality_metrics), xmax=(x_position + 0.5)/len(quality_metrics))
            else:
                # 对于 Delta amplitude，阈值在下方
                ax.axhline(y=threshold, color='red', linestyle='--', xmin=(x_position - 0.5)/len(quality_metrics), xmax=(x_position + 0.5)/len(quality_metrics))
    #plt.ylim(0,2)
    plt.show()


# 定义函数：计算阈值（使用中位数）
'''
def calculate_thresholds(df_metrics):
    quality_metrics = ['50Hz power', 'muscle artifacts', 'baseline drift', 'peak', 'Delta amplitude']
    thresholds = df_metrics[quality_metrics].median() *1.5
    return thresholds
'''    

# 定义函数：计算阈值（使用最大值）
def calculate_thresholds(df_metrics,save_path):
    quality_metrics = ['50Hz power', 'muscle artifacts', 'baseline drift', 'peak','Alpha amplitude']
    thresholds = df_metrics[quality_metrics].quantile(0.8)
    print(thresholds)
    thresholds.to_csv(save_path)
    return thresholds


# 定义函数：执行质量检查
def perform_quality_checks(eeg_data, fs, thresholds):
    n_trials, n_channels, _ = eeg_data.shape
    metrics_list = []
    qualified_trials = []
    for trial_idx in range(n_trials):
        bad_channel_count = 0  # 不合格通道计数
        for ch_idx in range(n_channels):
            channel_data = eeg_data[trial_idx, ch_idx, :]

            # 计算质量指标
            power_50Hz_ratio = analyze_50Hz_noise(channel_data, fs)
            power_emg_ratio = analyze_emg(channel_data, fs)
            power_drift_ratio = analyze_baseline_drift(channel_data, fs)
            num_abnormal_values = analyze_abnormal_values(channel_data)
            relative_powers = analyze_spectral_features(channel_data, fs)
            alpha_power_ratio = relative_powers['alpha']

            # 存储质量指标
            metrics_list.append({
                'trial': trial_idx,
                'channel': ch_idx,
                '50Hz power': power_50Hz_ratio,
                'muscle artifacts': power_emg_ratio,
                'baseline drift': power_drift_ratio,
                'peak': num_abnormal_values,
                'Alpha amplitude': alpha_power_ratio
            })

            # 判断通道是否合格
            is_channel_qualified = True
            if power_50Hz_ratio > thresholds['50Hz power']:
                is_channel_qualified = False
                print("50Hz")
            # if power_emg_ratio > thresholds['muscle artifacts']:
            #     is_channel_qualified = False
            if power_drift_ratio > thresholds['baseline drift']:
                is_channel_qualified = False
                print("baseline")
            if num_abnormal_values > thresholds['peak']:
                is_channel_qualified = False
                print("peak")
            # if alpha_power_ratio > thresholds['Alpha amplitude']:
            #     is_channel_qualified = False
            #     print("alpha")
            if not is_channel_qualified:
                bad_channel_count += 1
        if bad_channel_count <=3:
            qualified_trials.append(eeg_data[trial_idx, :, :])
        # # 判断整个 trial 是否合格
        # if bad_channel_count <=3:
        #     qualified_trials.append(eeg_data[trial_idx, :, :])

    df_metrics = pd.DataFrame(metrics_list)
    qualified_trials = np.array(qualified_trials)
    return df_metrics, qualified_trials
# 主程序
def cal_thresholds(fs = 250,data_path = "data.npy",save_path = "thresholds.csv"):
    qualified_eeg_data = np.load(data_path)
    df_qualified_metrics = calculate_quality_metrics(qualified_eeg_data, fs)
    calculate_thresholds(df_qualified_metrics,save_path)

def quality_control(fs = 250,data_path = "data.npy",save_path = "thresholds.csv"):
    thresholds = pd.read_csv(save_path,index_col=0)
    thresholds = thresholds["0.8"]
    eeg_data_to_check = np.load(data_path) 
    df_metrics, qualified_trials = perform_quality_checks(eeg_data_to_check, fs, thresholds)
    return qualified_trials
# cal_thresholds()