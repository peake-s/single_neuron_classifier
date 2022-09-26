import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 


class perceptron:

    def __init__(self,fname):
        self.fname = fname
        self.df = pd.read_csv(self.fname,header=None, names = ["cost", "weight", "type"])
 

    def _normalize(self):    
        self.df['cost'] = (self.df['cost']-self.df['cost'].min())/(self.df['cost'].max()-self.df['cost'].min())
        self.df['weight'] = (self.df['weight']-self.df['weight'].min())/(self.df['weight'].max()-self.df['weight'].min())

    def _select_training_testing(self, test_sz):
        train,test = train_test_split(self.df,test_size=test_sz)
        return (train,test)

    def _output_soft(self,net,k):
        return 2/(1+np.exp(-2*k*net)) - 1

    #pass in learning constant, gain, number iterations, total error goal
    def _train(self,train_data = pd.DataFrame,lc = 0.001,k = 0.1,max_ite = 5000, target_error = 0.001, w =[0.1, 0.1, 0.1]):
        #weights and bias array
        wb = w
        ni = len(wb)
        train = train_data.to_numpy()
        dout = train_data['type'].to_numpy()
        TE = 0.0
        for _ in range(max_ite):
            out = []
            for (idx,pattern) in enumerate(train):
                net = 0.0 
                for i in range(ni):
                    net = net + wb[i]*pattern[i]

                out.append(self._output_soft(net,k))
                err = dout[idx] - out[idx]
                learn = lc * err
                for i in range(ni):
                    wb[i] = wb[i] + learn*pattern[i]
        
        return wb


    def predict(self):
        self._normalize()
        train,test = self._select_training_testing(0.25)
        self._train(train_data=train,lc=0.001,k=0.2,max_ite= 5000,target_error = 0.00001, w = [0.1,0.1,0.1])

def main():
    perc = perceptron('groupA.txt')
    w = perc.predict()
    print(w)

main()