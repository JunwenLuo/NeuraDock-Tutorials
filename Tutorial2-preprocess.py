import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
import matplotlib.patches as mpatches
from itertools import combinations
SAVE_PATH = "preprocess.png"
DATA_PATH = "test1_20250622113115_52s_NeuraDock0228537id_1751938462.txt"


# --- 2. 定义标签和颜色 ---
NOISE_COLOR_MAP = {
    '50Hz Line Noise': 'green',
    'EMG': 'orange',
    'Baseline Wander': 'purple',
    'Outlier/Spike': 'red',
    'Mixed Noise': 'gray'
}

def data_reader(file_path):
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


def butter_bandpass_filter(data1, lowcut, highcut, fs, order):  

    nyq = 0.5 * fs  
    low = lowcut / nyq  
    high = highcut / nyq  
    b, a = signal.butter(order, [low, high], btype='band')   
    y = signal.filtfilt(b, a, data1)  
    return y  


def global_optimum_indices(matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """
    针对“瘦高”或“矮胖”矩阵，返回构成全局最大全True子矩阵的行和列的索引。

    Args:
        matrix: 一个二维布尔型 NumPy 数组。

    Returns:
        一个元组 (retained_rows, retained_cols)，包含要保留的行索引列表和列索引列表。
    """
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
    """
    将保留的行块索引转换为原始数据的行索引列表。

    Args:
        retained_block_rows: `global_optimum_indices` 返回的行块索引列表。
        row_chunk_size: 每个行块包含的原始数据点数 (例如 1250)。

    Returns:
        一个包含所有要保留的原始数据行索引的列表。
    """
    raw_indices = []
    for block_index in retained_block_rows:
        start_index = block_index * row_chunk_size
        end_index = start_index + row_chunk_size
        # 使用 range 生成这个块内的所有原始索引
        raw_indices.extend(range(start_index, end_index))
    return raw_indices

def channel_selecter(noise_value):
    noise_types = ["50Hz","EMG","Baseline","outlier"]
    thresh = [10,20,40,2]

    for r in range(4):
        noise_value[:,:,r] /= thresh[r]


    selected_channel = [0,1,2,3,4,5,6]
    bool_arr = noise_value < 10

    mask = np.all(bool_arr, axis=2)
    
    retained_block_rows,retained_rows_indices = global_optimum_indices(mask)

    return retained_block_rows,retained_rows_indices


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
            ann_onsets.append(index)
            
            noise_value_seg = []
            for i in range(7):
                end_index = min(index+1250,data_length-1)
                ann_durations.append(end_index - index)
                f, Pxx = signal.welch(data[i,index:end_index], 250, nperseg=2048)
                power_50hz = np.sum(Pxx[np.where((f >= 49) & (f <= 51))])
                power_emg = np.sum(Pxx[np.where((f >= 20) & (f <= 40))])
                power_baseline = np.sum(Pxx[np.where((f >= 0.5) & (f <= 4))])
                outlier = np.where((data[i,index:end_index]<=-50) | (data[i,index:end_index]>=50))[0].shape[0]

                noise_value_seg.append([power_50hz,power_emg,power_baseline,outlier])

            noise_value.append(noise_value_seg)
    noise_value = np.array(noise_value)

    return noise_value,ann_onsets, ann_durations


def classify_noise_segment(noise_metrics):
    """根据4个指标和阈值对单个噪声片段进行分类"""
    # noise_metrics: [power_50hz, power_emg, power_baseline, outlier_count]
    labels = []
    if noise_metrics[0] > 10:
        labels.append('50Hz Line Noise')
    if noise_metrics[1] > 10:
        labels.append('EMG')
    if noise_metrics[2] > 10:
        labels.append('Baseline Wander')
    if noise_metrics[3] > 10:
        labels.append('Outlier/Spike')

    if len(labels) > 1:
        return 'Mixed Noise'
    elif len(labels) == 1:
        return labels[0]
    else:
        return 'Clean' # 如果没有噪声超过阈值

data, data_marker = data_reader(DATA_PATH)

# Preprocessing: Apply bandpass filter (2-45Hz) to each channel
fs = 250  # Sampling frequency
lowcut = 2
highcut = 45
order = 5
noise_types = ["50Hz","EMG","Baseline","outlier"]
filtered_data = []
for channel in range(data.shape[0]):
    filtered_channel = butter_bandpass_filter(data[channel, :], lowcut, highcut, fs, order)
    filtered_data.append(filtered_channel)

filtered_data = np.array(filtered_data)

# Calculate noise level and select good channels
noise_value,ann_onsets, ann_durations = noise_calculation(filtered_data, data_marker)
retained_block_rows,retained_rows_indices = channel_selecter(noise_value)


print("retained_block_rows",retained_block_rows)

if data_marker != []:

    good_data_chunks = []      # 存储好段的数据块

    current_length = 0         # 新数据流的当前长度


    # --- 2. 遍历要保留的片段，进行拼接和更新 ---
    print("\n--- 开始处理 ---")
    for idx in retained_block_rows:
        # a) 获取当前好段的原始起止信息
        onset_s = ann_onsets[idx]
        duration_s = ann_durations[idx]
        
        start_sample = onset_s 
        end_sample = start_sample +duration_s
        
        
        # b) 提取好段数据
        segment_data = filtered_data[:, start_sample:end_sample]
        good_data_chunks.append(segment_data)
                
        # d) 更新新数据流的总长度，为下一个好段做准备
        segment_length = end_sample - start_sample
        current_length += segment_length

    # --- 3. 完成最终的拼接 ---
    clean_data = np.concatenate(good_data_chunks, axis=1)
    clean_data = clean_data[retained_rows_indices,:]
else:
    raw_rows_to_keep = get_raw_data_indices_to_keep(retained_block_rows, 1250)

    raw_rows_to_keep = np.array(raw_rows_to_keep)
    raw_rows_to_keep = raw_rows_to_keep[raw_rows_to_keep<filtered_data.shape[1]]
    clean_data = filtered_data[retained_rows_indices,:]
    clean_data = clean_data[:,raw_rows_to_keep]


# raw_rows_to_keep = get_raw_data_indices_to_keep(retained_block_rows, 1250)

# raw_rows_to_keep = np.array(raw_rows_to_keep)
# raw_rows_to_keep = raw_rows_to_keep[raw_rows_to_keep<filtered_data.shape[1]]
# clean_data = filtered_data[retained_rows_indices,:]
# clead_data = clean_data[:,raw_rows_to_keep]
if clean_data.shape[1]/data.shape[1]>=0.7:
    data_quality = "high"
elif 0.4<=data.shape[1]<0.7:
    data_quality = "middle"
else:
    data_quality = "low"
del_chunks =  [x for x in [i for i in range(data.shape[1]//1250+1)] if x not in retained_block_rows]
bool_arr = noise_value > 10
mask_noise = np.all(bool_arr, axis=1)

t = np.linspace(0,filtered_data.shape[1]/250,filtered_data.shape[1])

artifacts = {name: [] for name in ['50Hz', 'EMG', 'Drift', 'Outlier']}
noise_type_list = ['50Hz', 'EMG', 'Drift', 'Outlier']

for channel in range(bool_arr.shape[1]):
    for noise_type in range(bool_arr.shape[2]):
        true_indices = np.where(bool_arr[:,channel,noise_type])[0]

        if len(true_indices)>=0:
            artifacts[noise_type_list[noise_type]]+=[{'channel': channel, 'time': np.arange(i*5, i*5+5) } for i in true_indices]

# --- 2. 设计可视化布局 ---

# 使用科技感的暗色主题
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(30, 12), constrained_layout=True)
fig.suptitle('Signal Quality Report (Left: Raw Data, Right: Pre-processed Data)', fontsize=20, fontweight='bold', color='black')
plt.subplots_adjust(wspace=0.1, hspace=0.8)
# 定义一个4行2列的网格
gs = gridspec.GridSpec(16, 20, figure=fig)

# 分配子图
ax_a = [fig.add_subplot(gs[0, :9]),fig.add_subplot(gs[1, :9]),fig.add_subplot(gs[2, :9]),fig.add_subplot(gs[3, :9]),fig.add_subplot(gs[4, :9]),fig.add_subplot(gs[5, :9]),fig.add_subplot(gs[6, :9])]   # 伪迹概览图
ax_b = fig.add_subplot(gs[8:15, 0:5])  # FFT图
ax_c = fig.add_subplot(gs[8:15, 5:10])  # 统计分布图


ax_d = [fig.add_subplot(gs[0, 10:]),fig.add_subplot(gs[1, 10:]),fig.add_subplot(gs[2, 10:]),fig.add_subplot(gs[3, 10:]),fig.add_subplot(gs[4, 10:]),fig.add_subplot(gs[5, 10:]),fig.add_subplot(gs[6, 10:])]   # 伪迹概览图
ax_e = fig.add_subplot(gs[8:15, 10:15])  # FFT图
ax_f = fig.add_subplot(gs[8:15, 15:])  # 统计分布图

# --- 3. 填充每个子图 ---




n_channels, n_samples = data.shape
channel_names = [f'Ch {i+1}' for i in range(n_channels)]
data_time = np.arange(n_samples) / 250

# (a) 时域图
num_segments = min(len(ann_onsets), len(ann_durations), len(noise_value))

# 遍历每个通道进行绘制和标注
for i in range(n_channels):
    # 1. 绘制信号

    ax_a[i].plot(data_time, data[i], color='black', lw=0.7)
    ax_a[i].set_ylabel(channel_names[i])
    ax_a[i].grid(True, linestyle='--', alpha=0.6)
    
    # 2. 对每个噪声片段，独立判断并标注当前通道
    for idx in range(num_segments):
        onset, duration = ann_onsets[idx], ann_durations[idx]
        
        # 获取当前通道、当前片段的噪声指标
        # 假设 noise_values[idx] 的形状是 (n_channels, n_features)
        channel_specific_noise = noise_value[idx][i]
        print(channel_specific_noise)
        # 对当前通道的噪声进行分类
        label = classify_noise_segment(channel_specific_noise)

        # 如果当前通道在该片段被判定为有噪声，则进行标注
        if label != 'Clean':
            start_time = onset / 250
            end_time = np.min([(onset + duration) / 250,n_samples/250])
            
            color = NOISE_COLOR_MAP[label]
            ax_a[i].axvspan(start_time, end_time, color=color, alpha=0.5, ec='none') # 使用 ec='none' 避免边框
ax_a[0].set_xticks([])


ax_a[1].set_xticks([])
ax_a[2].set_xticks([])
ax_a[3].set_xticks([])
ax_a[4].set_xticks([])
ax_a[5].set_xticks([])

# 创建并添加自定义图例
legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.4) for label, color in NOISE_COLOR_MAP.items() if label != 'Clean']
fig.legend(handles=legend_patches, loc='upper right', fontsize='medium', title='Noise Types')


ax_a[-1].set_xlabel('Time (s)')


# (b) 功率谱密度 (FFT)
ax_b.set_title('Power Spectral Density (PSD)', fontsize=14, fontweight='bold')
colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_channels))
for i in range(n_channels):
    freqs, psd = signal.welch(filtered_data[i], fs, nperseg=fs*2)
    ax_b.semilogy(freqs, psd, label=channel_names[i], color=colors[i], lw=1.5)
ax_b.axvline(50, color='#FFD700', linestyle='--', lw=2, label='50 Hz Line')
ax_b.set_xlabel('Frequency (Hz)', fontsize=12)
ax_b.set_ylabel('Power/Frequency (dB/Hz)')
ax_b.set_xlim(0, 80)
ax_b.legend(fontsize='small')
ax_b.grid(linestyle='--', alpha=0.3)

# (c) 数据统计分布 (Peak-to-Peak)
ax_c.set_title('Peak-to-Peak Amplitude Distribution', fontsize=14, fontweight='bold')
# 将数据分段计算p2p
p2p_values = []
segment_len = fs # 1秒一段
for i in range(n_channels):
    n_segments = len(filtered_data[i]) // segment_len
    segments = filtered_data[i, :n_segments * segment_len].reshape(n_segments, segment_len)
    p2p_values.append(np.ptp(segments, axis=1))

box = ax_c.boxplot(p2p_values, labels=channel_names, patch_artist=True,
                medianprops={'color': 'black', 'linewidth': 2})
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_c.set_ylabel('Peak-to-Peak Amplitude (μV)', fontsize=12)
ax_c.set_xticklabels(channel_names, rotation=45, ha='right')
ax_c.grid(axis='y', linestyle='--', alpha=0.3)
ax_c.set_ylim(0,200)



clean_n_channels, clean_n_samples = clean_data.shape
clean_data_time = np.arange(clean_n_samples) / 250
# 遍历每个通道进行绘制和标注
for index,i in enumerate(retained_rows_indices):
    # 1. 绘制信号
    ax_d[i].plot(clean_data_time, clean_data[index], color='black', lw=0.7)
    ax_d[i].set_ylabel(channel_names[i])
    ax_d[i].grid(True, linestyle='--', alpha=0.6)
    

# (e) 功率谱密度 (FFT)
ax_e.set_title('Power Spectral Density (PSD)', fontsize=14, fontweight='bold')
for index,i in enumerate(retained_rows_indices):
    freqs, psd = signal.welch(clean_data[index], fs, nperseg=fs*2)
    ax_e.semilogy(freqs, psd, label=channel_names[i], color=colors[i], lw=1.5)
ax_e.axvline(50, color='#FFD700', linestyle='--', lw=2, label='50 Hz Line')
ax_e.set_xlabel('Frequency (Hz)', fontsize=12)
ax_e.set_ylabel('Power/Frequency (dB/Hz)')
ax_e.set_xlim(0, 80)
ax_e.legend(fontsize='small')
ax_e.grid(linestyle='--', alpha=0.3)

# (f) 数据统计分布 (Peak-to-Peak)
ax_f.set_title('Peak-to-Peak Amplitude Distribution', fontsize=14, fontweight='bold')
# 将数据分段计算p2p
p2p_values = []
segment_len = fs # 1秒一段
for index,i in enumerate(retained_rows_indices):
    n_segments = len(clean_data[index]) // segment_len
    segments = clean_data[index, :n_segments * segment_len].reshape(n_segments, segment_len)
    p2p_values.append(np.ptp(segments, axis=1))
clean_channel_names = ["Ch "+str(i+1) for i in retained_rows_indices]
box = ax_f.boxplot(p2p_values, labels=clean_channel_names, patch_artist=True,
                medianprops={'color': 'black', 'linewidth': 2})
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_f.set_ylabel('Peak-to-Peak Amplitude (μV)', fontsize=12)
ax_f.set_xticklabels(clean_channel_names, rotation=45, ha='right')
ax_f.grid(axis='y', linestyle='--', alpha=0.3)
ax_f.set_ylim(0,200)




# --- 4. 添加组图标签并美化 ---
fig_labels = ['a', 'b', 'c']
axes = [ax_a[0], ax_b, ax_c]

for i, ax in enumerate(axes):
    # 将标签放在左上角外侧
    ax.text(-0.05, 1.05, fig_labels[i], transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right', color='white')



plt.savefig(SAVE_PATH)
