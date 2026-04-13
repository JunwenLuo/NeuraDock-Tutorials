import socket
import numpy as np
import os

class DataStream:
    def __init__(self, IP, PORT, 
                 buffer_size=1024,
                 total_channels=8,      # 硬件总通道数
                 used_channels=7,       # 实际使用的通道数（取前 N 个）
                 pkg_groups=5,          # 每个网络包包含的时间点数
                 data_group_len=1,      # 每次 __next__ 返回的时间点数
                 save_filepath=None):   # 要保存的本地 txt 文件路径
        """
        在线数据流：从 TCP 服务器接收 CSV 格式的神经信号数据。
        """
        self.ip = IP
        self.port = PORT
        self.buffer_size = buffer_size
        self.total_channels = total_channels
        self.used_channels = used_channels
        self.pkg_groups = pkg_groups
        self.data_group_len = data_group_len
        
        self.save_filepath = save_filepath
        self.file_handle = None
        
        self.is_running = False
        self.socket = None
        self._buffer_str = ""      # 累积未解析的原始字符串
        self._data_buffer =[]     # 累积已解析的时间点（每个是 shape=(used_channels,)）

    def __iter__(self):
        if self.is_running:
            self.close()
        self.is_running = True
        self._connect()
        

        # 重置缓冲区
        self._buffer_str = ""
        self._data_buffer =[]
        return self

    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.ip, self.port))
            self.socket.send(b'start')
            print(f"[DataStream] Connected to {self.ip}:{self.port}")
        except Exception as e:
            print(f"[DataStream] Connection Error: {e}")
            self.is_running = False
            raise

    def close(self):
        self.is_running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        # 安全关闭文件
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None

    def run_DataStream(self):
        self._connect()
        while True:
            # if self.trigger_queue.full():
            #     self.TCP_sent('trigger:'+str(self.trigger_queue.get()))
            self.data = self.socket.recv(self.buffer_size)
            self.data = self.data.decode()

            content = list(map(str,self.data.split(',')[2:self.pkg_groups*self.total_channels+2]))
            # print(self.data)
            # break

            content[7::8] = ["\n","\n","\n","\n","\n"]
            content = ","+",".join(content)
            with open(self.save_filepath,"a") as f:
                f.write(content)

    def __del__(self):
        self.close()

# pp=DataStream("100.79.133.136",9600,save_filepath="1.txt")
# pp.run_DataStream()