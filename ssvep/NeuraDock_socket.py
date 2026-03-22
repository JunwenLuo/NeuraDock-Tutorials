import socket
import numpy as np

class DataStream(object):
    def __init__(self, IP, PORT, data_length=5):
        '''
        Parameters
        ----------
        IP : str
            IP address of server.
        PORT : int
            port number of server.
        data_length : float, optional
            The default is 5. data length in package
            USB:1
            bluetooth:5
        '''
        super(DataStream,self).__init__()
        self.s        = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port     = PORT
        self.ip       = IP
        self.x_       = []
        self.packagesize   = 512
        self.data_length   = data_length  
        self.s_state       = None
        
        
                    
    def close_DataStream(self):
        self.s_state  = False
        return self.s.close()
    
    def start_DataStream(self):
        self.s        = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.ip,self.port))
        self.DataStream_init()
        self.s_state  = True
        return self.TCP_sent('start')
        

    
    def TCP_sent(self,y):
        return self.s.send(bytes(y,encoding='utf8'))   

    def DataStream_init(self):

        self.packagesize,self.pkgGroups,self.channelNum = 512,self.data_length,8


        
                
    def run_DataStream(self,save_path):
        global IS_CLOSE
        
        if not self.s_state:
            self.start_DataStream()
        _log = 0
        while True:
            try:
                self.data = self.s.recv(self.packagesize)
                self.data = self.data.decode()

                self.x_   +=list(map(float,self.data.split(',')[2:self.pkgGroups*self.channelNum+1]))
                content = list(map(str,self.data.split(',')[2:self.pkgGroups*self.channelNum+2]))
                
                content[7::8] = ["\n"]*self.data_length
                content = ","+",".join(content)
                with open(save_path,"a") as f:
                    f.write(content)
            except:
                continue
    def run_DataStream_realtime(self):
        global IS_CLOSE
        
        if not self.s_state:
            self.start_DataStream()
        _log = 0
        while True:
            self.data = self.s.recv(self.packagesize)
            self.data = self.data.decode()
            time_stamp = self.data.split(",")[0]
            sequence_num = self.data.split(",")[1]
            self.x_   +=list(map(float,self.data.split(',')[2:self.pkgGroups*self.channelNum+1]))
            content = list(map(str,self.data.split(',')[2:self.pkgGroups*self.channelNum+2]))
            
            content[7::8] = ["\n"]*self.data_length

            data = np.array([content[i*8:i*8+7] for i in range(self.data_length)]).reshape((-1,7))
            yield {
                    'data': data,
                    'timestamp': time_stamp,
                    'sequence_num': sequence_num,
                    'package_size': 7*self.data_length
                }
    



pp=DataStream(IP='10.187.255.187', PORT=9600,data_length=5)
pp.run_DataStream("1.txt")