from neuradock import data_reader, butter_bandpass_filter,data_selecter

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
file_path = '/Users/lijunling/Documents/Proj/neuradock-V2.1-alpha/appstore/10hz.txt'
fs = 250
left_temporal_idx = 0
right_temporal_idx = 2
alpha_band = (8, 13)
nperseg = 4 * fs
data, data_marker, channel_name = data_reader(file_path)
print(data_marker)
filtered_data = []
for i in range(7):
    filtered_data.append(butter_bandpass_filter(data[i],2,45,250,5))
filtered_data = np.array(filtered_data)
pxx_list = np.zeros(501)
data_list = np.zeros(1000)
for i in data_marker:
    data_i = filtered_data[5,i[0]:i[0]+1000]
    data_list = data_list+data_i


freq,pxx = welch(data_list,250,nperseg=1024)
# pxx_list = pxx_list+pxx
    

plt.plot(freq,pxx)
plt.xlim(0,30)
plt.savefig("10hz.png")
