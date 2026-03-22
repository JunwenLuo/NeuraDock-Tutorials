

from Neuradock_lib import data_reader, butter_bandpass_filter, data_selecter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch  # 引入welch方法用于计算PSD
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# ================= 设置参数 =================
file_path = '1.txt'
fs = 250
epoch_len = 1000  # 截取的长度 (2.4秒)
# num_channels = 7

# ================= 数据读取与预处理 =================
data, data_marker = data_reader(file_path)

# data_marker = data_marker[:10] # 如果需要限制marker数量取消注释
data, data_marker = data_selecter(data, data_marker)
print(f"Markers count: {len(data_marker)}")
num_channels = data.shape[0]

# 滤波处理 (对所有通道进行滤波)
filtered_data = []
for i in range(num_channels):
    # 假设 butter_bandpass_filter 返回一维数组
    filtered_data.append(butter_bandpass_filter(data[i], 2, 80, fs, 5))
filtered_data = np.array(filtered_data) # shape: (num_channels, total_samples)

# ================= 绘图准备 =================
# 创建 num_channels 行，2 列的画布
# 第一列：时域 (Time Domain)
# 第二列：频域 (Frequency Domain / PSD)
fig, axes = plt.subplots(num_channels, 2, figsize=(14, 3 * num_channels))

# 调整布局防止重叠，特别是当行数较多时
plt.subplots_adjust(hspace=0.4)

# 如果只有一个通道，axes shape可能是 (2,)，需要统一处理成 (1, 2)
if num_channels == 1:
    axes = np.expand_dims(axes, axis=0)

# ================= 循环处理每个通道 =================
for ch in range(num_channels):
    data_i_list = []
    
    # 1. 提取 Epoch
    for marker in data_marker:
        start_idx = marker[0]
        end_idx = start_idx + epoch_len
        
        # 边界检查
        if end_idx <= filtered_data.shape[1]:
            segment = filtered_data[ch, start_idx:end_idx]
            data_i_list.append(segment)
    
    # 如果没有提取到数据，跳过
    if not data_i_list:
        print(f"Channel {ch+1}: No valid epochs found.")
        continue

    # shape: (num_epochs, epoch_len)
    epochs_array = np.array(data_i_list) 
    
    # 计算叠加平均 (ERP)
    # axis=0 表示沿着“试次”方向求平均
    channel_mean = np.mean(epochs_array, axis=0)
    
    # ================= 绘制第一列：时域叠加平均图 =================
    ax_time = axes[ch, 0]
    
    # 画出单次试次的淡淡的灰线
    for epoch in epochs_array:
        ax_time.plot(epoch, linewidth=0.5, color='gray', alpha=0.3)
        
    # 画出叠加平均后的粗线
    ax_time.plot(channel_mean, linewidth=1.5, color='blue', label='Mean')
    
    ax_time.set_title(f'Ch {ch+1} - Time Domain (ERP)', fontsize=10)
    ax_time.grid(True, linestyle='--', alpha=0.5)
    
    # ================= 绘制第二列：Marker平均后的PSD图 =================
    ax_psd = axes[ch, 1]
    
    # 计算功率谱密度 (PSD)
    # nperseg 设置为 epoch_len 可以获得该窗口长度下的完整分辨率
    f, Pxx = welch(channel_mean, fs=fs, nperseg=len(channel_mean))
    
    # 绘制 PSD 曲线
    ax_psd.plot(f, Pxx, color='red', linewidth=1.5, label='PSD')
    
    # 设置 PSD 图的属性
    ax_psd.set_title(f'Ch {ch+1} - PSD (of Mean)', fontsize=10)
    ax_psd.grid(True, linestyle='--', alpha=0.5)
    ax_psd.set_ylabel('Power/Frequency')
    
    # 由于滤波范围是 2-45Hz，我们可以限制X轴显示范围让图更清晰（例如 0-60Hz）
    ax_psd.set_xlim(0, 60)

    # =================设置轴标签 (仅最后一行显示) =================
    if ch == num_channels - 1:
        ax_time.set_xlabel('Time (samples)')
        ax_psd.set_xlabel('Frequency (Hz)')

# ================= 保存图片 =================
plt.tight_layout()
save_name = "7_channels_mean_and_psd15hz.pdf"
plt.savefig(save_name)
print(f"Plot saved to {save_name}")
plt.close()