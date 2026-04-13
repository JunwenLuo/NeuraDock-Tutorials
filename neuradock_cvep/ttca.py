import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from quality import quality_control
import os
from scipy.stats import zscore
from mne.decoding.receptive_field import _delay_time_series, _times_to_delays
from sklearn.metrics import accuracy_score
import math

class TTCA():
    def __init__(self, S_train, S_test, winLEN = 4.13, srate = 250.0, alpha = 0.95, tmin = 0.0, tmax = 0.8, n_band = 5, bandmethod=3, jitter=2,filterlearn=False,labels_test = np.arange(4)):
        self.winLEN = winLEN
        self.n_band = n_band
        self.srate = srate
        self.latency = round(0.14*self.srate)
        self.alpha = alpha
        self.tmin = tmin
        self.tmax = tmax
        self.fill_mean = True
        self.bandmethod=bandmethod
        S_train = (S_train-np.min(S_train,axis=-1,keepdims=True))/(np.max(S_train,axis=-1,keepdims=True)-np.min(S_train,axis=-1,keepdims=True))*2-1
        if S_test is not None:
            S_test = (S_test-np.min(S_test,axis=-1,keepdims=True))/(np.max(S_test,axis=-1,keepdims=True)-np.min(S_test,axis=-1,keepdims=True))*2-1
        self.S_train = S_train
        self.S_test = S_test
        self.jitter = jitter
        self.labels_test = labels_test
        self.load_template()

    def load_template(self):
        self.T1 = np.load("template1.npy")
        self.T2 = np.load("template2.npy")
        self.T3 = np.load("template3.npy")
        self.T4 = np.load("template4.npy")

        print("data load is completed")
    def filterbank(self,x,freqInx):

        srate = self.srate/2
        
        if self.bandmethod==0:      #ssvep
            # passband1 = [2, 14, 22, 30, 38, 46, 54, 62, 70, 78]
            # stopband1 = [1, 10, 16, 24, 32, 40, 48, 56, 64, 72]
            # passband2 = [124,124,124,124,124,124,124,124,124,124]
            # stopband2 = [125,125,125,125,125,125,125,125,125,125]
            
            passband1 = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
            stopband1 = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
            passband2 = [90,90,90,90,90,90,90,90,90,90]
            stopband2 = [100,100,100,100,100,100,100,100,100,100]
            
            # passband1 = [6]
            # stopband1 = [4]
            # passband2 = [34]
            # stopband2 = [40]
            
        if self.bandmethod==1:
            passband1=[6,6,6,6,6]
            stopband1=[4,4,4,4,4]
            passband2=[18,26,34,42,50]
            stopband2=[24,32,40,48,56]
        
        elif self.bandmethod==2:
            passband1=[6,14,22,30,38]
            stopband1=[4,10,16,24,32]
            passband2=[18,26,34,42,50]
            stopband2=[24,32,40,48,56]
        
        elif self.bandmethod==3:
            passband1=[6,14,22,30,38]
            stopband1=[4,10,16,24,32]
            passband2=[18,34,50,50,50]
            stopband2=[24,40,56,56,56]
        
        elif self.bandmethod==4:
            passband1=[6,14,22,30,38]
            stopband1=[4,10,16,24,32]
            passband2=[50,50,50,50,50]
            stopband2=[56,56,56,56,56]
        
        
        Wp = [passband1[freqInx]/srate, passband2[freqInx]/srate]
        Ws = [stopband1[freqInx]/srate, stopband2[freqInx]/srate]
        
        [N, Wn]=signal.cheb1ord(Wp, Ws, 3, 40)
        [B, A] = signal.cheby1(N, 0.5, Wn,'bandpass')
        
        filtered_signal = np.zeros(np.shape(x))
        if len(np.shape(x))==2:
            for channelINX in range(np.shape(x)[0]):
                filtered_signal[channelINX,:] = signal.filtfilt(B, A, x[channelINX, :])
        elif len(np.shape(x))==3:
            for epochINX,epoch in enumerate(x):
                for channelINX in range(np.shape(epoch)[0]):
                    filtered_signal[epochINX,channelINX,:] = signal.filtfilt(B, A, epoch[channelINX, :])

        return filtered_signal
    
    def fit(self):
        X = np.concatenate([self.T1,self.T2,self.T3,self.T4],axis=0)
        y = np.array([1 for i in range(self.T1.shape[0])]+[2 for i in range(self.T2.shape[0])]+[3 for i in range(self.T3.shape[0])]+[4 for i in range(self.T4.shape[0])])
        winLENs = round(self.winLEN*self.srate)
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)
        S_train = self.S_train[:,self.latency:self.latency+winLENs]
        self.filters,_=self.getSpatialFilters(X,y)    
        self.epochs = X
        self.labels = y
        
        self.TRF=self.getTRF(self.evokeds, S_train)     #fb*tau
        # print(len(self.TRF[0]))
        # for i in range(5):
        #     plt.plot(self.TRF[i])
        #     plt.show()
        self.rctX = self.getrctX(self.S_test)       #fb*class*T

        return 
    
    def predict(self, X):
        
        winLENs = round(self.winLEN*self.srate)
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)

        rctX = self.rctX[...,self.latency:self.latency+winLENs]
        rctX = rctX.transpose(1,0,-1)      #class*fb*T
        rctXs = _delay_time_series(rctX.transpose(-1,0,1), -self.jitter/self.srate, self.jitter/self.srate, self.srate,fill_mean=True)      #T*class*fb*lag
        rctXs = rctXs.transpose(1,-1,2,0)   #class*lag*fb*T
        
        X_addfb = []
        for fbINX in range(self.n_band):
            X_addfb.append(self.filterbank(X, fbINX))
        X_addfb = np.stack(X_addfb)     # fb*epoch*channel*T
        X_addfb = np.transpose(X_addfb,(1,0,-2,-1))
        
        epochNUM, _, _ = np.shape(X)
        classNUM, lagNUM, _, _ = np.shape(rctXs)

        
        fb_coefs = np.expand_dims(np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)    #-1.25,0.25
        
        result = []
        corrmatrix = np.zeros((epochNUM, classNUM))
        
        for epochINX, epoch in enumerate(X_addfb):
            # rlagINX = np.zeros((classNUM,1)).astype('int')
            for classINX, template in enumerate(rctXs):
                rlag = np.zeros((lagNUM,1))
                for lagINX, laggedtemp in enumerate(template):
                    rtemp = np.zeros((self.n_band,1))
                    for fbINX, (fbepoch, fbtemplate, fbfilter) in enumerate(zip(epoch, laggedtemp, self.filters)):
                        rtemp[fbINX,:] = np.corrcoef(fbtemplate.reshape(1,-1), (fbfilter.T.dot(fbepoch)).reshape(1,-1))[0,1]
                    rlag[lagINX,:] = fb_coefs.dot(rtemp)
                r=np.max(rlag)
                # rlagINX[classINX,:]=np.argmax(rlag)
                corrmatrix[epochINX,classINX]=r
                
            # ================== 【新增的打印逻辑】 ==================
            # 获取当前试次对每一类的相关系数值
            scores = corrmatrix[epochINX, :]
            
            # 使用 Softmax 将相关系数转换为 0~1 的伪概率 (减去最大值是为了防止指数运算溢出)
            exp_scores = np.exp(scores - np.max(scores)) 
            probs = exp_scores / np.sum(exp_scores)
            
            # 格式化打印：保留四位小数以便于观察
            print(f"--- 试次(Epoch) {epochINX+1} ---")
            print(f"原始相关系数 (Raw Corr): {np.round(scores, 4)}")
            print(f"预测概率 (Probability) : {np.round(probs, 4)}")
            # =======================================================
            result.append(self.labels_test[np.argmax(corrmatrix[epochINX,:])])
        self.result = np.stack(result)
        self.corrmatrix = corrmatrix
        
        return self.result
        
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    
    def getSpatialFilters(self,X,y):
        labels = np.unique(y)
        epochNUM, self.channelNUM, _ = np.shape(X)
        
        X_addfb = []
        for fbINX in range(self.n_band):
            X_addfb.append(self.filterbank(X, fbINX))
        X_addfb = np.stack(X_addfb)     #fb*epoch*channel*T
        
        X_classified = []
        for fbX in X_addfb:
            augmentClass = []
            for _, label in enumerate(labels):
                this_class_data =fbX[y == label]
                augmentEpoch = []
                for epoch in this_class_data:
                    augmentEpoch.append(epoch)
                augmentClass.append(np.stack(augmentEpoch))
            X_classified.append(augmentClass)       #fb*class*block*channel*T
        X_classified = np.stack(X_classified)

        augmentEvoked = []
        for fbs in X_classified:
            augmentEvoked.append([con.mean(axis=0) for con in fbs])
        augmentEvoked = np.stack(augmentEvoked)
        
        filters=[]
        for (fbEvoked, fbEpochs) in zip(augmentEvoked, X_classified):
            # norm
            fbEvoked = fbEvoked-np.mean(fbEvoked,axis=-1,keepdims=True)
            fbEvokedFeature = np.mean(fbEvoked, axis=0, keepdims=True)
            betwClass = fbEvoked-fbEvokedFeature
            betwClass = np.concatenate(betwClass,axis=1)    #channel*(block*T)
            # norm
            fbEpochs = [this_class-np.mean(this_class, axis=-1, keepdims=True) for this_class in fbEpochs]
            # allClassEvoked = [this_class-np.mean(this_class, axis=0, keepdims=True) for this_class in fbEpochs]
            allClassEvoked = fbEpochs
            allClassEvoked = [np.transpose(this_class,axes=(1,2,0)) for this_class in allClassEvoked]
            allClassEvoked = [np.reshape(this_class, (self.channelNUM, -1),order='F') for this_class in allClassEvoked]
            allClassEvoked = np.hstack(allClassEvoked)

            
            Hb = betwClass/math.sqrt(len(labels))
            Hw = allClassEvoked/math.sqrt(epochNUM)
            Sb = np.dot(Hb,Hb.T)
            Sw = np.dot(Hw, Hw.T)
            
            C = np.linalg.inv(Sw).dot(Sb)
            lamda, W = np.linalg.eig(C)

            idx=lamda.argsort()[::-1]
            W = W[:,idx]
            filter=np.squeeze(W[:,0:1])
            filters_corrected=filter*np.sign(filter[np.argmax(abs(filter),axis=0)])
            
            filters.append(filters_corrected)
        filters = np.stack(filters)       #fb*channel
        
        
        evokeds=[]
        for _, (fbfilter, fbevoked) in enumerate(zip(filters, augmentEvoked)):
            fbfilter = fbfilter[np.newaxis,:]
            evoked = []
            for _, epoch in enumerate(fbevoked):
                evoked.append(np.squeeze(fbfilter.dot(epoch)))
            evokeds.append(evoked)
        evokeds = np.stack(evokeds)    #fb*class*T
        self.evokeds = evokeds
        
        return filters, evokeds

            
    def getTRF(self,X,S):
        TRF_allepochs=[]
        TRF = []
        _, epochNUM, _, = np.shape(X)
        laggedLEN = len(_times_to_delays(self.tmin,self.tmax,self.srate))
        for fbINX, fbX in enumerate(X):
            fbKernel = np.zeros((epochNUM,laggedLEN))
            fbCov_sr = np.zeros((epochNUM,laggedLEN))
            
            fbS = self.filterbank(S,fbINX)
            for epochINX,(epoch,sti) in enumerate(zip(fbX,fbS)):
                sti = sti[:,np.newaxis]
                epoch = epoch[np.newaxis,:]
                laggedS = _delay_time_series(sti, self.tmin, self.tmax,self.srate,fill_mean=self.fill_mean).squeeze()
                # stimulation whitening
                Cov_ss = laggedS.T.dot(laggedS)
                u,sigma,v = np.linalg.svd(Cov_ss)
                for i in range(len(sigma)):
                    if sum(sigma[0:len(sigma)-i]/sum(sigma)) < self.alpha:
                        sigma = 1/sigma
                        sigma[len(sigma)-i:] = 0
                        break
                sigma_app = np.diag(sigma)
                inv_C = u.dot(sigma_app).dot(v)
                
                
                fbCov_sr[epochINX,:] = np.squeeze(epoch.dot(laggedS))
                fbKernel[epochINX,:] = np.squeeze(epoch.dot(laggedS).dot(inv_C.T))

            TRF_allepochs.append(fbKernel)
            TRF.append(np.mean(fbKernel,axis=0))
            
        self.TRF_allepochs=np.stack(TRF_allepochs)     #fb*class*tau
        TRF = np.stack(TRF)   #fb*tau
        
        return TRF
    

    def getrctX(self, S):

        rctX = []
        for fbINX,fbtrf in enumerate(self.TRF):
            fbrX = []
            fbS = self.filterbank(S,fbINX)
            fbtrf = fbtrf[np.newaxis,:]
            for _, sti in enumerate(fbS):
                sti = sti[:,np.newaxis]
                laggeds = np.squeeze(_delay_time_series(sti,tmin = self.tmin, tmax = self.tmax, sfreq = self.srate))
                fbrX.append(np.squeeze(fbtrf.dot(laggeds.T)))
            rctX.append(fbrX)
        rctX = np.stack(rctX)       #fb*class*T
        return rctX
# def xor(x, y):
#     return (x+y)%2

# def get_mseq(coef):
#     st = coef
#     backQ = xor(coef[-1],coef[-3])
#     result = [int(backQ)]
#     temp = []
#     temp.extend(st[:-1])
#     temp.insert(0,int(backQ))
#     while(st != temp):
#         backQ = xor(temp[-1],temp[-3])
#         result.append(int(backQ))
#         temp = temp[:-1]
#         temp.insert(0,int(backQ))
#     return result
# m1 = get_mseq([0,0,0,0,1])
# m1 = [i for i in m1 for j in range(3)]
# mm = [m1,m1[12:]+m1[:12],m1[24:]+m1[:24],m1[36:]+m1[:36]] 
# mm1 = []
# for i in range(4):
#     mm1.append([mm[i][round(j/385*93)-1] for j in range(385)])

# ttst1 = TTCA(np.array(mm1),np.array(mm1))
# ttst1.fit()
# T1 = np.load("template1.npy")
# T2 = np.load("template2.npy")
# T3 = np.load("template3.npy")
# T4 = np.load("template4.npy")
# print(ttst1.predict(T1))
# print(ttst1.predict(T2))
# print(ttst1.predict(T3))
# print(ttst1.predict(T4))