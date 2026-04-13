import threading
from psychopy import visual, core, event
from neuradock_socket import DataStream 
from ttca import TTCA
import glob
import numpy as np
import os
import random
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
m1 = get_mseq([0,0,0,0,0,1])
m1 = [i for i in m1 for j in range(3)]

mseqs = [m1,m1[12:]+m1[:12],m1[24:]+m1[:24],m1[36:]+m1[:36]] 

def stimulate(win):
        # 创建窗口
    global selected
    global target
    right = 0
    wrong = 0
    box_text = ["A","B","C","D"]
    
    letters_order = [random.sample(box_text,1),random.sample(box_text,1),
                     random.sample(box_text,1),random.sample(box_text,1),
                     random.sample(box_text,1),random.sample(box_text,1)]
    prompt_box = visual.TextBox2(win,text = "focus on the letters in the following order:",letterHeight=30,pos =(0,450) ,color=(1, 1, 1))
    letter_boxes = []
    for i in range(6):
        letter_boxes.append(visual.TextBox2(win,text = f"{letters_order[i][0]}",letterHeight=30,pos =(i*40,420) ,color=(1, 1, 1)))
    # 创建刺激
    stims = []
    for i in range(4):
        stims.append([visual.ImageStim(win,pos=(i%6*400-600,0),image="black.png", size=(200, 200),),visual.ImageStim(win,pos=(i%6*400-600,0),image="white.png", size=(200, 200),)])
    rects = []
    for i in range(4):
        rects.append(visual.Rect(win,pos=(i%6*400-600,0), size=(200, 200),color=(1, 1, 1)))
    texts = []
    
    for i in range(4):
        texts.append(visual.ImageStim(win,pos=(i%6*400-600,0),image="black.png", size=(200, 200),))
    t = 0
    while t<6:
        for frameN in range(63*2):
            for index,freq in enumerate(range(4)):
                

                texts[index].draw()
            prompt_box.draw()
            for i in range(6):
                letter_boxes[i].draw()
            win.flip()
        with open("1.txt","a") as f:
            f.write("marker")
            f.write("\n")
        for frameN in range(189):  # 假设屏幕刷新率是60Hz
            
            for index,freq in enumerate(range(4)):
                

                stims[index][mseqs[index][frameN]].draw()
            win.flip()
        with open("trigger.txt","w")as f:
            f.write("1")
            
        for frameN in range(63*2):
            for index,freq in enumerate(range(4)):
                

                texts[index].draw()
            prompt_box.draw()
            for i in range(6):
                letter_boxes[i].draw()
            win.flip()
            if "target.txt" in glob.glob("*.txt"):
                
                with open("target.txt")as f:
                    d = int(f.read()[0])
                if box_text[d]==letters_order[t][0]:
                    letter_boxes[t].color=(0,1,0)
                    right+=1
                else:
                    letter_boxes[t].color=(1,0,0)
                    wrong+=1
                t = t+1
                os.remove("target.txt")
                break
    print(f"acurancy:{right}/{right+wrong}")

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

# pp=DataStream(IP='10.187.183.32', PORT=9600,save_path="1.txt")
# pp=DataStream(IP='192.168.56.1', PORT=9600,save_path="1.txt")
pp=DataStream(IP='172.17.144.1', PORT=9600,save_path="1.txt")
# pp=DataStream(IP='100.79.160.249', PORT=9600,save_path="1.txt")
th1 = threading.Thread(target=pp.run_DataStream)
th1.start()
m1 = get_mseq([0,0,0,0,0,1])
print(m1)
m1 = [i for i in m1 for j in range(3)]
mm = [m1,m1[12:]+m1[:12],m1[24:]+m1[:24],m1[36:]+m1[:36]] 
for i in mm:
    print(i)
import matplotlib.pyplot as plt
def fun2():
    m1 = get_mseq([0,0,0,0,0,1])
    m1 = [i for i in m1 for j in range(3)]
    mm = [m1,m1[12:]+m1[:12],m1[24:]+m1[:24],m1[36:]+m1[:36]] 
    mm1 = []
    for i in range(4):
        mm1.append([mm[i][round(j/788*189)-1] for j in range(788)])

    ttst1 = TTCA(np.array(mm1),np.array(mm1))
    ttst1.fit()
    count=0
    while count<6:
        if "trigger.txt" in glob.glob("*.txt"):
            print("========")
            data1 = np.zeros((1,7,788))
            count1 = -1
            tt = 9999
            with open ("1.txt") as f:
                d = f.readlines()
                for i in range(len(d)):
                    if tt<788:
                        try:
                            data1[count1,:,tt] = np.array([float(d[i].split(",")[1]),float(d[i].split(",")[2]),float(d[i].split(",")[3]),float(d[i].split(",")[4]),float(d[i].split(",")[5]),float(d[i].split(",")[6]),float(d[i].split(",")[7])])
                            tt = tt+1
                        except:
                            data1[count1,:,:] = np.sin(np.arange(0,150,150/788))*200
                            tt = 9999
                    if d[i]=='marker\n':
                        tt = 0
                        count1+=1
            with open("1.txt","w")as f:
                f.write("")
            data1 = np.array(data1)
            print(data1.shape)
            ans = ttst1.predict(data1)
            print(ans)
            os.remove("trigger.txt")

            with open("target.txt","w")as f:
                f.write(str(ans[0]))
            count+=1


            
th2 = threading.Thread(target=fun2)
th2.start()
win = visual.Window(fullscr=False,size=(1500,1000), color=(0, 0, 0), units="pix")
stimulate(win)

win.close()