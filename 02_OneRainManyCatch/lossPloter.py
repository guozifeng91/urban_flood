import numpy as np
import math

# ploting library
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

import os

class Epoch_Loss:
    def __init__(self):
        self.epoch_loss = 0
        self.batch_loss = 0
        self.batch_num = 0
        
    def put_batch(self, batch_loss):
        self.batch_loss = batch_loss
        self.epoch_loss += self.batch_loss
        self.batch_num += 1
    
    def get_batch_loss(self):
        return self.batch_loss
    
    def get_epoch_loss(self):
        return self.epoch_loss / self.batch_num
    
    def clear(self):
        self.epoch_loss = 0
        self.batch_loss = 0
        self.batch_num = 0

class Loss_Ploter:
    def __init__(self):
        self.y_batch=[]
        self.x_batch=[]
        
        self.y_ep_train=[]
        self.y_ep_test=[]
        self.x_ep=[]
        
        self.count = 0
        self.max=0
        self.min=1000
        
        #self.log_min=1
        
        plt.ion()
        
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_yscale("log")
        
        self.ax.axis([0,1,0.1,1])
        self.fig.show()
        
        plt.pause(0.001) # let the GUI run
    
    def put_batch_loss(self, loss):
        self.x_batch.append(self.count)
        self.y_batch.append(loss)
        
        self.max=max(self.max,loss)
        self.min=min(self.min,loss)
        
        self.count += 1

    def put_epoch_loss(self, loss_train, loss_test):
        self.x_ep.append(self.count)
        self.y_ep_train.append(loss_train)
        self.y_ep_test.append(loss_test)
        
        self.max=max(self.max,max(loss_train,loss_test))
        self.min=min(self.min,min(loss_train,loss_test))
    
    def plot(self, num_points=100):
        if (self.count < 2):
            return
        
        # the number that is used to show the epoch
        base = self.count
        if len(self.x_ep) > 0:
            base = self.x_ep[0]
            
        self.ax.cla() # clean the axes
        self.ax.axis([0,self.count/base,self.min,self.max])
        
        # sampling and plot batch loss
        step = 1 + (self.count // num_points)
        x=[]
        y=[]
        
        for i in range(0,self.count,step):
            x.append(i/base)
            y.append(self.y_batch[i])
        
        self.ax.plot(x,y)

        # plot epoch loss
        if (len(self.x_ep)>1):
            self.ax.plot(np.array(self.x_ep)/base,np.array(self.y_ep_train))
            self.ax.plot(np.array(self.x_ep)/base,np.array(self.y_ep_test))
        
        self.ax.set_yscale("log")
        
        plt.draw()
        plt.pause(0.001) # let the GUI run
    
    def load_record(self, model_path, model_name):
        batch_loss_file = model_name + "_batch_loss.csv"
        epoch_loss_file = model_name + "_epoch_loss.csv"
        
        batch_loss_history = np.loadtxt(os.path.join(model_path, batch_loss_file), dtype=np.float32)
        epoch_loss_history = np.loadtxt(os.path.join(model_path, epoch_loss_file), dtype=np.float32)
        
        [self.put_batch_loss(l) for l in batch_loss_history[1]]
        
        self.x_ep = list(epoch_loss_history[0])
        self.y_ep_train = list(epoch_loss_history[1])
        self.y_ep_test = list(epoch_loss_history[2])
        
    def save_record(self, model_path, model_name):
        # save the log of training loss
        np.savetxt(os.path.join(model_path,model_name+"_batch_loss.csv"),np.array([self.x_batch,self.y_batch]))
        np.savetxt(os.path.join(model_path,model_name+"_epoch_loss.csv"),np.array([self.x_ep,self.y_ep_train, self.y_ep_test]))
    
    def close(self):
        plt.close(self.fig)
        
    def savefig(self, filename):
        self.fig.savefig(filename)