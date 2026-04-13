import numpy as np
import random
from psychopy import visual, core, event
from quality import quality_control
import scipy.signal as signal
import threading
from neuradock_socket import DataStream 
# pp=DataStream(IP='10.187.183.32', PORT=9600,save_path="1.txt")
# pp=DataStream(IP='192.168.56.1', PORT=9600,save_path="1.txt")
pp=DataStream(IP='172.17.144.1', PORT=9600,save_path="1.txt")
# pp=DataStream(IP='100.79.160.249', PORT=9600,save_path="1.txt")
th1 = threading.Thread(target=pp.run_DataStream)
th1.start()
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
def xor(x, y):
    return (x+y)%2

def get_mseq(coef):
    st = coef
    backQ = xor(coef[-1],coef[-2])
    result = [int(backQ)]
    temp = []
    temp.extend(st[:-1])
    temp.insert(0,int(backQ))
    while(st != temp):
        backQ = xor(temp[-1],temp[-2])
        result.append(int(backQ))
        temp = temp[:-1]
        temp.insert(0,int(backQ))
    return result
def th2():
    m1 = get_mseq([0,0,0,0,0,1])
    m1 = [i for i in m1 for j in range(3)]

    mm = [m1,m1[12:]+m1[:12],m1[24:]+m1[:24],m1[36:]+m1[:36]] 
    template = [[],[],[],[]]
    win = visual.Window(fullscr=False,size=(1500,1000), color=(0, 0, 0), units="pix")
    stims = []
    box_text = ["A"]
    texts = []
    for i in range(1):
        texts.append(visual.TextBox2(win,text = f"{box_text[i]}",letterHeight=80,pos =(280,0) ,color=(1, 1, 1)))
    for i in range(3):
        stims.append([visual.ImageStim(win,pos=(i%6*600,0),image="black.png", size=(200, 200),),visual.ImageStim(win,pos=(i%6*600,0),image="white.png", size=(200, 200),)])
    t = 0
    qualified_num = 0
    while t <4:
        for frameN in range(60):
            texts[0].text = str(1-frameN/60)[:3]
            texts[0].draw()
            win.flip()
        for trail in range(4):
            with open("1.txt","a") as f:
                f.write("marker")
                f.write("\n")            
            for frameN in range(189): 
                stims[0][mm[t][frameN]].draw()
                win.flip()
            for frameN in range(60): 
                texts[0].text = str(1-frameN/60)[:3]
                texts[0].draw()
                win.flip()
        if 'escape' in event.getKeys():
            break
        data1 = np.zeros((4,7,788))
        count = -1
        tt = 9999
        with open ("1.txt") as f:
            d = f.readlines()
            for i in range(len(d)):
                if tt<788:
                    try:
                        data1[count,:,tt] = np.array([float(d[i].split(",")[1]),float(d[i].split(",")[2]),float(d[i].split(",")[3]),float(d[i].split(",")[4]),float(d[i].split(",")[5]),float(d[i].split(",")[6]),float(d[i].split(",")[7])])
                        tt = tt+1
                    except:
                        data1[count,:,:] = np.sin(np.arange(0,150,150/788))*200
                        tt = 9999
                if d[i]=='marker\n':
                    tt = 0
                    count+=1
                if count==4:
                    break
        # for trial in range(4):
        #     for channel in range(7):
        #         data1[trial,channel,:] = butter_bandpass_filter(data1[trial][channel],1,45,250,5)
        with open("1.txt","w")as f:
            f.write("")
        np.save("data_offline.npy",data1)
        # qualified_trials = quality_control(data_path="data_offline.npy")
        qualified_trials = data1
        print(qualified_trials.shape)
        if qualified_trials.shape[0]>0:
            template[t].append(qualified_trials)
        qualified_num += qualified_trials.shape[0]
        print(qualified_num)
        if qualified_num>=20:
            t = t+1
            qualified_num = 0
    t1 = np.concatenate(template[0],axis=0)
    t2 = np.concatenate(template[1],axis=0)
    t3 = np.concatenate(template[2],axis=0)
    t4 = np.concatenate(template[3],axis=0)
    print(t1.shape)
    np.save("template1.npy",t1)
    np.save("template2.npy",t2)
    np.save("template3.npy",t3)
    np.save("template4.npy",t4)
    win.close()
    core.quit()
th2()
