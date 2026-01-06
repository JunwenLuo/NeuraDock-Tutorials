
import  socket, threading, queue

import numpy as np


EEG_IP = "192.168.56.1"
EEG_PORT =  9600

class DataStream:
    def __init__(self, IP, PORT, buffer_size=1024, total_channels=8, used_channels=7, pkg_groups=1, data_group_len=1):
        self.ip = IP; self.port = PORT; self.buffer_size = buffer_size
        self.total_channels = total_channels; self.used_channels = used_channels
        self.pkg_groups = pkg_groups; self.data_group_len = data_group_len
        self.is_running = False; self.socket = None; self._buffer_str = ""; self._data_buffer = []

    def __iter__(self):
        if self.is_running: self.close()
        self.is_running = True; self._connect(); self._buffer_str = ""; self._data_buffer = []; return self
    
    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.ip, self.port))
            self.socket.send(b'start')
            print(f"[DataStream] Connected to {self.ip}:{self.port}")
        except Exception as e: print(f"[DataStream] Connect Error: {e}"); self.is_running=False; raise

    def close(self):
        self.is_running = False; 
        if self.socket: 
            try: self.socket.close() 
            except: pass; 
            self.socket=None

    def __next__(self):
        if not self.is_running: raise StopIteration
        while len(self._data_buffer) < self.data_group_len:
            try:
                chunk = self.socket.recv(self.buffer_size)
                if not chunk: raise ConnectionError("Closed")
                self._buffer_str += chunk.decode('utf-8', errors='ignore')
                while True:
                    lines = self._buffer_str.split('\n')
                    if len(lines) < 2: break
                    complete, self._buffer_str = lines[:-1], lines[-1]
                    for line in complete:
                        if not line.strip(): continue
                        fields = line.strip().split(',')
                        if len(fields) < 2 + self.pkg_groups * self.total_channels: continue
                        try:
                            data_vals = list(map(float, fields[2:2 + self.pkg_groups * self.total_channels]))
                            arr = np.array(data_vals, dtype=np.float32).reshape(self.pkg_groups, self.total_channels)
                            for t in range(self.pkg_groups): self._data_buffer.append(arr[t, :self.used_channels].tolist())
                        except: continue
            except socket.timeout: continue
            except Exception: self.close(); raise StopIteration
        res = self._data_buffer[:self.data_group_len]
        self._data_buffer = self._data_buffer[self.data_group_len:]
        return res

class EEGThreadManager:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=100000)
        self.stop_event = threading.Event(); self.stream = None; self.thread = None; self.current_trigger = 0

    def start_stream(self):
        try:
            self.stream = DataStream(EEG_IP, EEG_PORT)
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start(); return True
        except Exception as e: print(f"[Thread Error] {e}"); return False

    def _worker(self):
        print(">> Worker Running.")
        try:
            for data_group in self.stream:
                if self.stop_event.is_set(): break
                if data_group is None: continue
                arr = np.array(data_group)
                marker = np.full((arr.shape[0], 1), self.current_trigger)
                self.data_queue.put(np.hstack([arr, marker]))
        except: pass
        finally: print(">> Worker Stopped.")
    
    def stop_stream(self):
        self.stop_event.set()
        if self.stream: self.stream.close()
        if self.thread: self.thread.join(timeout=1.0)

    def flush_to_buffer(self, buf):
        while not self.data_queue.empty(): 
            try: buf.append(self.data_queue.get_nowait())
            except queue.Empty: break

    def set_trigger(self, code): self.current_trigger = int(code)
