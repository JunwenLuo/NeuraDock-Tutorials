import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

import numpy as np
from scipy.signal import welch
import os
import numpy as np
from numpy.fft import rfft, rfftfreq

def data_reader(file_path:str):
    """用这个工具来读取和解码数据，返回原始数据和marker信息，使用方式data,data_marker = data_reader("1111.txt")，data是一个二维np.array，行是通道数，列是数据长度；marker是一个列表，每行表示事件对应的原始信号行号以及事件标签例如[[1008, 'trial0\n'], [1260, 'trial1-target\n']]，其中第一个数据表示原始数据的第1008行有个marker：'trial0\n'。
       Args:
           file_path: 数据路径.
       Returns:
           原始数据，marker信息
    """
    data1 = []
    data1_marker = []
    with open (file_path) as f:
        d = f.readlines()
        header_line = d[0]

        if not header_line.startswith('HEADER_DEF,'):
            raise ValueError("文件头部格式不正确，必须以 'HEADER_DEF,' 开始。")
        header_parts = header_line.strip().split(',')[1:]
        channel_indices = [i for i, part in enumerate(header_parts) if part == 'C']
        package_size = len(channel_indices)//7


        for i in range(1,len(d)):
            try:
                data_line_split = d[i].strip().split(',')
                data_line = [[float(data_line_split[x]) for x in channel_indices[j*7:j*7+7]] for j in range(package_size)]
                data1 = data1+data_line

            except:
                data1_marker.append([len(data1), d[i]])
    
    data1 = np.array(data1).transpose(1,0)

    return data1,data1_marker



def EEG_quality_check(data_path, segment_len_sec, fs, powerline_freq=50):
    """
    EEG数据质量评分系统 (专家版本)
    
    检测伪影包括: 
    1. 电极脱落 (Flatline)
    2. 肌电干扰 (EMG)
    3. 信号饱和 (Saturation)
    4. 基线漂移 (Baseline Wander)
    5. 工频噪音 (Power-line Noise)

    参数:
    data_path (str): EEG数据文件的路径。
    segment_len_sec (int): 每个数据段的长度（秒）。
    fs (int): 数据的采样率 (Hz)。
    powerline_freq (int): 工频噪音频率 (通常为 50 或 60 Hz)。

    返回:
    tuple: (quality_scores, quality_flags, is_usable_matrix, data)
        - quality_scores (np.array): 形状为 (段数, 通道数) 的评分矩阵 (0-100分)。
        - quality_flags (np.array): 形状为 (段数, 通道数, 5) 的伪影标记矩阵。
        - is_usable_matrix (np.array): 形状为 (段数, 通道数) 的布尔矩阵，True表示可用。
        - data (np.array): 从文件中读取的原始EEG数据。
    """
    # 1. 加载数据
    data, _ = data_reader(data_path)
    
    # 2. 定义参数
    n_channels, n_samples = data.shape
    segment_len_samples = int(segment_len_sec * fs)
    segment_num = n_samples // segment_len_samples
    
    # 3. 初始化结果数组 (使用字典提高可读性)
    ARTIFACT_INDICES = {
        'detachment': 0, 
        'emg': 1, 
        'saturation': 2, 
        'baseline_drift': 3, 
        'powerline': 4
    }
    n_artifact_types = len(ARTIFACT_INDICES)
    quality_flags = np.zeros((segment_num, n_channels, n_artifact_types), dtype=int)
    quality_scores = np.full((segment_num, n_channels), 100.0) # 满分100

    # 4. 定义伪影检测的阈值 (可根据实际情况调整)
    # 饱和/平线
    SATURATION_THRESHOLD = 300      # uV, 信号饱和的峰峰值阈值
    DETACHMENT_STD_THRESHOLD = 2.0  # uV, 平线的标准差阈值
    # 频域阈值
    SIGNAL_FREQ_LOW = 2             # Hz, 主要信号频段
    SIGNAL_FREQ_HIGH = 50           # Hz
    EMG_FREQ_LOW = 30               # Hz, EMG频段
    EMG_FREQ_HIGH = 70            # Hz
    BASELINE_FREQ_HIGH = 1.0        # Hz, 基线漂移频段
    
    # 功率比阈值
    EMG_POWER_RATIO_THRESHOLD = 3     # EMG功率 与 信号功率 的比值
    BASELINE_POWER_RATIO_THRESHOLD = 5.0 # 基线漂移功率 与 信号功率 的比值
    POWERLINE_PEAK_RATIO_THRESHOLD = 5.0 # 工频峰值功率 与 周围频段平均功率 的比值

    # 5. 定义评分系统参数
    SCORE_UNUSABLE_THRESHOLD = 40  # 低于此分数的段被认为完全不可用
    PENALTIES = {
        'emg': 30,
        'baseline_drift': 25,
        'powerline': 20
    }

    # 6. 遍历所有段和通道
    for i in range(segment_num):
        for j in range(n_channels):
            start_idx = i * segment_len_samples
            end_idx = start_idx + segment_len_samples
            segment = data[j, start_idx:end_idx]
            
            if len(segment) < segment_len_samples:
                continue

            # --- 伪影检测 ---
            is_flat = np.std(segment) < DETACHMENT_STD_THRESHOLD
            is_saturated = max(segment) - min(segment) > SATURATION_THRESHOLD

            # a) 电极脱落 (致命)
            if is_flat:
                quality_flags[i, j, ARTIFACT_INDICES['detachment']] = 1
            
            # b) 信号饱和 (致命)
            if is_saturated:
                quality_flags[i, j, ARTIFACT_INDICES['saturation']] = 1
            
            # 如果是致命伪影，直接评0分，跳过后续频域分析
            if is_flat or is_saturated:
                quality_scores[i, j] = 0
                continue
            
            # --- 频域分析 (使用 Welch 方法更稳定) ---
            freqs, psd = welch(segment, fs=fs, nperseg=segment_len_samples, scaling='density')
            
            # 计算各频段总功率
            def get_band_power(f_low, f_high):
                band_indices = np.where((freqs >= f_low) & (freqs < f_high))[0]
                return np.sum(psd[band_indices]) if len(band_indices) > 0 else 1e-10

            signal_power = get_band_power(SIGNAL_FREQ_LOW, SIGNAL_FREQ_HIGH)

            # c) 基线漂移检测
            baseline_power = get_band_power(0, BASELINE_FREQ_HIGH)
            baseline_ratio = baseline_power / signal_power
            if baseline_ratio > BASELINE_POWER_RATIO_THRESHOLD:
                quality_flags[i, j, ARTIFACT_INDICES['baseline_drift']] = 1

            # d) 肌电 (EMG) 干扰检测
            emg_power = get_band_power(EMG_FREQ_LOW, EMG_FREQ_HIGH)
            emg_ratio = emg_power / signal_power
            if emg_ratio > EMG_POWER_RATIO_THRESHOLD:
                quality_flags[i, j, ARTIFACT_INDICES['emg']] = 1
                
            # e) 工频噪音检测
            powerline_band_indices = np.where((freqs > powerline_freq - 1) & (freqs < powerline_freq + 1))[0]
            if len(powerline_band_indices) > 0:
                powerline_peak_power = np.max(psd[powerline_band_indices])
                # 计算周围频段的平均功率作为参考
                surrounding_indices_1 = np.where((freqs > powerline_freq - 5) & (freqs < powerline_freq - 2))[0]
                surrounding_indices_2 = np.where((freqs > powerline_freq + 2) & (freqs < powerline_freq + 5))[0]
                surrounding_power = np.mean(psd[np.concatenate((surrounding_indices_1, surrounding_indices_2))])
                surrounding_power = max(surrounding_power, 1e-10) # 避免除以零
                
                if powerline_peak_power / surrounding_power > POWERLINE_PEAK_RATIO_THRESHOLD:
                    quality_flags[i, j, ARTIFACT_INDICES['powerline']] = 1

    # 7. 计算最终分数
    for i in range(segment_num):
        for j in range(n_channels):
            # 只有在非致命性错误时才计算扣分
            if quality_scores[i, j] > 0:
                score = 100
                if quality_flags[i, j, ARTIFACT_INDICES['emg']] == 1:
                    score -= PENALTIES['emg']
                if quality_flags[i, j, ARTIFACT_INDICES['baseline_drift']] == 1:
                    score -= PENALTIES['baseline_drift']
                if quality_flags[i, j, ARTIFACT_INDICES['powerline']] == 1:
                    score -= PENALTIES['powerline']
                quality_scores[i, j] = max(0, score) # 分数不低于0
    
    # 8. 生成“是否可用”矩阵
    is_usable_matrix = quality_scores >= SCORE_UNUSABLE_THRESHOLD
    
    return quality_scores, quality_flags, is_usable_matrix, data

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_eeg_quality(data, fs, segment_len_sec, quality_flags, artifact_indices):
    """
    专业地可视化EEG信号质量评估结果 (V3: 优化了线条、字体和图例)。

    参数:
    data (np.array): 原始EEG数据 (channels, samples)。
    fs (int): 采样率 (Hz)。
    segment_len_sec (int): 分段长度 (秒)。
    quality_flags (np.array): 伪影标记矩阵 (segments, channels, n_artifact_types)。
    artifact_indices (dict): 将伪影名称映射到其在 quality_flags 中索引的字典。
    """
    n_channels, n_samples = data.shape
    time_axis = np.arange(n_samples) / fs

    # --- 定义可视化参数 ---
    MIN_YLIM_RANGE = 100.0
    Y_PADDING_FACTOR = 0.1
    TICK_FONTSIZE = 12  # <--- 新增：定义刻度字体大小
    LINE_WIDTH = 1.5    # <--- 新增：定义信号线宽度

    idx_to_name = {v: k for k, v in artifact_indices.items()}
    artifact_colors = {
        'saturation':     ('crimson', 0.2),
        'detachment':     ('black', 0.2),
        'emg':            ('orange', 0.2),
        'baseline_drift': ('royalblue', 0.2),
        'powerline':      ('darkviolet', 0.2),
    }

    # --- 创建图表 ---
    fig_height = max(16, n_channels * 1.5)
    fig, axes = plt.subplots(n_channels, 1, figsize=(20, fig_height), sharex=True, sharey=False)
    if n_channels == 1:
        axes = [axes]

    # 将主标题稍微上移，为图例留出空间
    fig.suptitle('EEG Quality Assessment Visualization', fontsize=16, y=0.98)

    # --- 遍历每个通道并绘图 ---
    for j in range(n_channels):
        ax = axes[j]
        channel_data = data[j, :]
        
        # 1. 动态计算并设置Y轴范围
        data_min, data_max = np.min(channel_data), np.max(channel_data)
        if np.isclose(data_min, data_max):
            data_min -= 1; data_max += 1
        data_range = data_max - data_min
        display_range = max(data_range, MIN_YLIM_RANGE)
        padding = display_range * Y_PADDING_FACTOR
        y_center = (data_max + data_min) / 2
        ax.set_ylim(y_center - (display_range / 2) - padding, 
                    y_center + (display_range / 2) + padding)

        # 2. 绘制原始EEG信号 (改动点：加粗线条)
        ax.plot(time_axis, channel_data, color='black', linewidth=LINE_WIDTH) # <--- 改动点
        ax.set_ylabel(f'Ch {j}\n(uV)', rotation=0, labelpad=25, va='center')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # (改动点：放大刻度字体)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE) # <--- 改动点

        # 3. 在背景上绘制伪影标记
        segment_num = quality_flags.shape[0]
        for i in range(segment_num):
            start_time = i * segment_len_sec
            end_time = start_time + segment_len_sec
            segment_flags = quality_flags[i, j, :]
            detected_indices = np.where(segment_flags == 1)[0]
            if len(detected_indices) > 0:
                priority_order = ['saturation', 'detachment', 'emg', 'baseline_drift', 'powerline']
                best_artifact_to_show = None
                for name in priority_order:
                    if artifact_indices[name] in detected_indices:
                        best_artifact_to_show = name
                        break
                if best_artifact_to_show:
                    color, alpha = artifact_colors[best_artifact_to_show]
                    ax.axvspan(start_time, end_time, color=color, alpha=alpha, ec=None, zorder=0)

    # --- 设置坐标轴和图例 ---
    axes[-1].set_xlabel('Time (s)', fontsize=14)
    
    legend_patches = [mpatches.Patch(color=c, alpha=a, label=name.replace('_', ' ').title()) 
                      for name, (c, a) in artifact_colors.items()]
    
    # (改动点：将图例放在顶部水平排列)
    fig.legend(handles=legend_patches,
               loc='lower center',             # 将图例的底部中心对齐
               bbox_to_anchor=(0.5, 0.92),     # 放置在图表顶部中心的位置
               ncol=len(artifact_colors),      # 水平排列
               fontsize=20,
               frameon=False)                  # 去掉边框，更简洁
    
    # 调整布局以防止标签重叠，并为顶部的图例和标题留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.92]) # rect=[left, bottom, right, top]
    plt.show()

import numpy as np
from scipy import signal
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

import numpy as np
from scipy.signal import welch
import os
import numpy as np
from numpy.fft import rfft, rfftfreq
from itertools import combinations

def data_reader(file_path:str):
    """用这个工具来读取和解码数据，返回原始数据和marker信息，使用方式data,data_marker = data_reader("1111.txt")，data是一个二维np.array，行是通道数，列是数据长度；marker是一个列表，每行表示事件对应的原始信号行号以及事件标签例如[[1008, 'trial0\n'], [1260, 'trial1-target\n']]，其中第一个数据表示原始数据的第1008行有个marker：'trial0\n'。
       Args:
           file_path: 数据路径.
       Returns:
           原始数据，marker信息
    """
    data1 = []
    data1_marker = []
    with open (file_path) as f:
        d = f.readlines()
        header_line = d[0]

        if not header_line.startswith('HEADER_DEF,'):
            raise ValueError("文件头部格式不正确，必须以 'HEADER_DEF,' 开始。")
        header_parts = header_line.strip().split(',')[1:]
        channel_indices = [i for i, part in enumerate(header_parts) if part == 'C']
        package_size = len(channel_indices)//7


        for i in range(1,len(d)):
            try:
                data_line_split = d[i].strip().split(',')
                data_line = [[float(data_line_split[x]) for x in channel_indices[j*7:j*7+7]] for j in range(package_size)]
                data1 = data1+data_line

            except:
                data1_marker.append([len(data1), d[i]])
    
    data1 = np.array(data1).transpose(1,0)

    return data1,data1_marker

def global_optimum_indices(matrix: np.ndarray) -> tuple[list[int], list[int]]:

    m, n = matrix.shape

    # 处理转置，确保我们总是穷举较小的维度
    transposed = False
    if m < n:
        matrix = matrix.T
        m, n = n, m  # 更新维度
        transposed = True

    max_area = 0
    best_rows_mask = np.zeros(m, dtype=bool)
    best_cols_indices = []

    for k in range(n, 0, -1):
        if k * m <= max_area:
            break
        for col_indices in combinations(range(n), k):
            sub_matrix_cols = matrix[:, list(col_indices)]
            retained_rows_mask = np.all(sub_matrix_cols, axis=1)
            num_retained_rows = np.sum(retained_rows_mask)
            current_area = num_retained_rows * k
            
            if current_area > max_area:
                max_area = current_area
                best_rows_mask = retained_rows_mask
                best_cols_indices = list(col_indices)

    # 从布尔掩码中获取行索引
    retained_rows_indices = np.where(best_rows_mask)[0].tolist()
    retained_cols_indices = best_cols_indices

    # 如果矩阵被转置过，需要交换回行和列的索引
    if transposed:
        return retained_cols_indices, retained_rows_indices
    else:
        return retained_rows_indices, retained_cols_indices
    
def get_raw_data_indices_to_keep(retained_block_rows: list[int], row_chunk_size: int) -> list[int]:

    raw_indices = []
    for block_index in retained_block_rows:
        start_index = block_index * row_chunk_size
        end_index = start_index + row_chunk_size
        # 使用 range 生成这个块内的所有原始索引
        raw_indices.extend(range(start_index, end_index))
    return raw_indices

def data_selecter(data,data_marker):
    fs = 250  # Sampling frequency
    lowcut = 2
    highcut = 45
    order = 5

    filtered_data = []
    for channel in range(data.shape[0]):
        filtered_channel = butter_bandpass_filter(data[channel, :], lowcut, highcut, fs, order)
        filtered_data.append(filtered_channel)
    filtered_data = np.array(filtered_data)

    
    
    noise_value,ann_onsets, ann_durations = noise_calculation(filtered_data, data_marker)
    noise_types = ["50Hz","EMG","Baseline","outlier"]
    thresh = [10,20,40,2]
    for r in range(4):
        noise_value[:,:,r] /= thresh[r]

    selected_channel = [0,1,2,3,4,5,6]
    bool_arr = noise_value < 10

    mask = np.all(bool_arr, axis=2)
    
    retained_block_rows,retained_rows_indices = global_optimum_indices(mask)
    new_markers = []  
    if data_marker != []:

        good_data_chunks = []      # 存储好段的数据块

        current_length = 0         # 新数据流的当前长度


        # --- 2. 遍历要保留的片段，进行拼接和更新 ---
        for idx in retained_block_rows:
            # a) 获取当前好段的原始起止信息
            onset_s = ann_onsets[idx]
            duration_s = ann_durations[idx]
            
            start_sample = onset_s 
            end_sample = start_sample +duration_s
            
            
            # b) 提取好段数据
            segment_data = data[:, start_sample:end_sample]
            good_data_chunks.append(segment_data)
            
            # c) 查找并更新位于该好段内的marker
            for original_marker_sample, marker_info in data_marker:
                # 检查marker是否落在这个好段的范围内。
                # 注意边界条件：通常包含起始点，不包含结束点。
                if start_sample <= original_marker_sample < end_sample:
                    # 计算marker在本段内的相对位置
                    relative_pos = original_marker_sample - start_sample
                    
                    # 计算marker在新的连续数据流中的绝对位置
                    new_marker_sample = current_length + relative_pos
                    
                    new_markers.append((new_marker_sample, marker_info))
                    
            # d) 更新新数据流的总长度，为下一个好段做准备
            segment_length = end_sample - start_sample
            current_length += segment_length

        # --- 3. 完成最终的拼接 ---
        new_eeg_data = np.concatenate(good_data_chunks, axis=1)
        new_eeg_data = new_eeg_data[retained_rows_indices,:]
    else:
        raw_rows_to_keep = get_raw_data_indices_to_keep(retained_block_rows, 1250)

        raw_rows_to_keep = np.array(raw_rows_to_keep)
        raw_rows_to_keep = raw_rows_to_keep[raw_rows_to_keep<filtered_data.shape[1]]
        new_eeg_data = filtered_data[retained_rows_indices,:]
        new_eeg_data = new_eeg_data[:,raw_rows_to_keep]
    return new_eeg_data, new_markers

def noise_calculation(data,data_marker):
    noise_value = []
    ann_onsets = []
    ann_durations = []
    n_samples = data.shape[1]
    if data_marker!= []:
        

        # 添加从数据开始到第一个marker的片段
        first_marker_idx = data_marker[0][0]
        
        if first_marker_idx > 0:
            ann_onsets.append(0)
            ann_durations.append(first_marker_idx)
            index = 0
            end_index = first_marker_idx
            noise_value_seg = []
            for i in range(7):
                f, Pxx = signal.welch(data[i,index:end_index], 250, nperseg=2048)
                power_50hz = np.sum(Pxx[np.where((f >= 49) & (f <= 51))])
                power_emg = np.sum(Pxx[np.where((f >= 20) & (f <= 40))])
                power_baseline = np.sum(Pxx[np.where((f >= 0.5) & (f <= 4))])
                outlier = np.where((data[i,index:end_index]<=-50) | (data[i,index:end_index]>=50))[0].shape[0]

                noise_value_seg.append([power_50hz,power_emg,power_baseline,outlier])

            noise_value.append(noise_value_seg)
        # 添加marker之间的片段
        for i in range(len(data_marker) - 1):
            start_marker_idx = data_marker[i][0]
            end_marker_idx = data_marker[i+1][0]
            
            onset = start_marker_idx 
            duration = end_marker_idx - start_marker_idx
            
            ann_onsets.append(onset)
            ann_durations.append(duration)
            
            index = start_marker_idx
            end_index = end_marker_idx
            noise_value_seg = []
            for i in range(7):
                f, Pxx = signal.welch(data[i,index:end_index], 250, nperseg=2048)
                power_50hz = np.sum(Pxx[np.where((f >= 49) & (f <= 51))])
                power_emg = np.sum(Pxx[np.where((f >= 20) & (f <= 40))])
                power_baseline = np.sum(Pxx[np.where((f >= 0.5) & (f <= 4))])
                outlier = np.where((data[i,index:end_index]<=-50) | (data[i,index:end_index]>=50))[0].shape[0]

                noise_value_seg.append([power_50hz,power_emg,power_baseline,outlier])

            noise_value.append(noise_value_seg)

            
        # 添加从最后一个marker到数据结束的片段
        last_marker_idx = data_marker[-1][0]
        if last_marker_idx < n_samples:
            ann_onsets.append(last_marker_idx)
            ann_durations.append((n_samples - last_marker_idx))

            
            index = last_marker_idx
            noise_value_seg = []
            for i in range(7):
                f, Pxx = signal.welch(data[i,index:], 250, nperseg=2048)
                power_50hz = np.sum(Pxx[np.where((f >= 49) & (f <= 51))])
                power_emg = np.sum(Pxx[np.where((f >= 20) & (f <= 40))])
                power_baseline = np.sum(Pxx[np.where((f >= 0.5) & (f <= 4))])
                outlier = np.where((data[i,index:]<=-50) | (data[i,index:]>=50))[0].shape[0]

                noise_value_seg.append([power_50hz,power_emg,power_baseline,outlier])

            noise_value.append(noise_value_seg)
    else:

        data_length = data.shape[1]

        for index in range(0,data_length,1250):

            noise_value_seg = []
            for i in range(7):
                end_index = min(index+1250,data_length-1)

                f, Pxx = signal.welch(data[i,index:end_index], 250, nperseg=2048)
                power_50hz = np.sum(Pxx[np.where((f >= 49) & (f <= 51))])
                power_emg = np.sum(Pxx[np.where((f >= 20) & (f <= 40))])
                power_baseline = np.sum(Pxx[np.where((f >= 0.5) & (f <= 4))])
                outlier = np.where((data[i,index:end_index]<=-50) | (data[i,index:end_index]>=50))[0].shape[0]

                noise_value_seg.append([power_50hz,power_emg,power_baseline,outlier])

            noise_value.append(noise_value_seg)
    noise_value = np.array(noise_value)

    return noise_value,ann_onsets, ann_durations

def butter_bandpass_filter(data1, lowcut, highcut, fs, order):  

    nyq = 0.5 * fs  
    low = lowcut / nyq  
    high = highcut / nyq  
    b, a = signal.butter(order, [low, high], btype='band')   
    y = signal.filtfilt(b, a, data1)  
    return y  


