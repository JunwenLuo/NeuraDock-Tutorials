from sqlite3 import DataError
import numpy as np
import mne
import matplotlib.pyplot as plt
from neuradock import data_reader,data_selecter,butter_bandpass_filter
from scipy.signal import welch
import matplotlib as mpl

# 全局设置字体
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False 

data,marker,channel = data_reader(file_path1)
data,marker,channel = data_selecter(data,marker)



alpha_power_data1 = []

for i in range(7):
    data[i] = butter_bandpass_filter(data[i],2,45,250,5)

for i in range(7):
    freqs, psd = welch(data[i],250,nperseg=2048)
    alpha_band = (8, 13)

    # 找到所有在 Alpha 波段内的频率点的索引
    idx_alpha = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
    print(idx_alpha)
    # 使用辛普森法则对Alpha波段的PSD进行积分，以获得总功率
    # 我们需要提供该频段的PSD值和对应的频率值
    alpha_power = np.mean(psd[idx_alpha])
    
    # 将计算出的 Alpha 功率添加到列表中
    alpha_power_data1.append(alpha_power)


alpha_power_data1 = np.array(alpha_power_data1)


data,marker,channel = data_reader(file_path2)
data,marker,channel = data_selecter(data,marker)



alpha_power_data2 = []

for i in range(7):
    data[i] = butter_bandpass_filter(data[i],2,45,250,5)

for i in range(7):
    freqs, psd = welch(data[i],250,nperseg=2048)
    alpha_band = (8, 13)

    # 找到所有在 Alpha 波段内的频率点的索引
    idx_alpha = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
    print(idx_alpha)
    # 使用辛普森法则对Alpha波段的PSD进行积分，以获得总功率
    # 我们需要提供该频段的PSD值和对应的频率值
    alpha_power = np.mean(psd[idx_alpha])
    
    # 将计算出的 Alpha 功率添加到列表中
    alpha_power_data2.append(alpha_power)


alpha_power_data2 = np.array(alpha_power_data2)

alpha_power_data = alpha_power_data2-alpha_power_data1



# --- 步骤 1 & 2: 定义电极系统和数据 (与之前相同) ---
montage_name = 'GSN-HydroCel-128'
montage = mne.channels.make_standard_montage(montage_name)
full_info = mne.create_info(ch_names=montage.ch_names, sfreq=100, ch_types='eeg')
full_info.set_montage(montage)

user_data_1020 = {
    'PO4': alpha_power_data[0], 'O2':  alpha_power_data[1], 'T6':  alpha_power_data[2], 'Oz':  alpha_power_data[3],
    'T5':  alpha_power_data[4], 'O1':  alpha_power_data[5], 'PO3': alpha_power_data[6]
}
channel_map_1020_to_E = {
    'O2': 'E83', 'PO4':  'E76', 'T6':  'E96', 'Oz':  'E75',
    'T5':  'E58', 'PO3':  'E71', 'O1': 'E70'
}


user_data_E_names = {channel_map_1020_to_E[name]: value for name, value in user_data_1020.items()}

# --- 核心修改结束 ---
# --- 步骤 3: 映射数据 (与之前相同) ---
full_ch_names = full_info['ch_names']
full_data_array = np.zeros(len(full_ch_names))
for ch_name, value in user_data_E_names.items():
    idx = full_ch_names.index(ch_name)
    full_data_array[idx] = value


# --- 步骤 4: 绘制地势图并进行自定义裁剪 ---
fig, ax = plt.subplots(figsize=(6, 6))

names_list = [''] * len(full_ch_names)

# 2. 遍历我们的通道映射字典
for standard_name, e_name in channel_map_1020_to_E.items():
    try:
        # 找到 'E' 名称在完整列表中的索引
        idx = full_ch_names.index(e_name)
        # 在该索引位置，填入我们想要显示的 标准名称

        names_list[idx] = standard_name
    except ValueError:
        # 这个警告几乎不会触发
        print(f"警告: 映射的通道 '{e_name}' 在info对象中未找到。")
im, cn = mne.viz.plot_topomap(
    full_data_array,
    full_info,
    axes=ax,
    show=False,
    extrapolate='local',
    image_interp='linear',
    contours=0,
    names=names_list,
    sensors=False, 
    vlim=(-0.8,0.8)
)

# 步骤 6: 添加颜色条和标题
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Alpha 波功率 ($μV^2$)', rotation=270, labelpad=15)
ax.set_title('Alpha 波相较于静息态的变化', fontsize=16)

plt.savefig(save_path)
