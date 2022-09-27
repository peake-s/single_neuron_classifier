import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

class perceptron:

    def __init__(self,fname):
        self.fname = fname
        self.df = pd.read_csv(self.fname,header=None, names = ["cost", "weight", "type"])
        self.results = []
 

    def _normalize(self):    
        self.df['cost'] = (self.df['cost']-self.df['cost'].min())/(self.df['cost'].max()-self.df['cost'].min())
        self.df['weight'] = (self.df['weight']-self.df['weight'].min())/(self.df['weight'].max()-self.df['weight'].min())

    def _select_training_testing(self, test_sz):
        train,test = train_test_split(self.df,test_size=test_sz)
        return (train,test)

    def _output_soft(self,net,k):
        return 2/(1+np.exp(-2*k*net)) - 1

    #pass in learning constant, gain, number iterations, total error goal
    def _train(self,train_data = pd.DataFrame,lc = 0.001,k = 0.1,max_ite = 5000, target_error = 0.001, nw = 3):
        #weights and bias array
        wb = np.random.uniform(size = nw, low = -0.5, high = 0.5)

        ni = nw
        train = train_data[["weight","cost"]].to_numpy()
        dout = train_data['type'].to_numpy()
        TE = 0.0
        for _ in range(max_ite):
            out = []
            for (idx,p) in enumerate(train):
                net = 0.0
                pattern = [p[0],p[1],1.0]

                for i in range(ni):
                    net = net + wb[i]*pattern[i]

                out.append(self._output_soft(net,k))
                err = dout[idx] - out[idx]
                
                learn = lc * err

                for i in range(ni):
                    wb[i] = wb[i] + learn*pattern[i]
        
        return wb
    
    def _test(self,test_data,weights):
        test = test_data[["weight","cost"]].to_numpy()
        actual = test['type'].to_numpy()
        predicted_correct = 0
        predicted_incorrect = 0

        for i,p in enumerate(test):
            pattern = [p[0],p[1],1.0]
            

    def plot(self,weights):
        
        xint = (0,-weights[2]/weights[1])
        yint = (-weights[2]/weights[0],0)
        m = -(weights[2]/weights[1])/(weights[2]/weights[0])
        y = m*self.df["weight"] + (-weights[2]/weights[1])

        plt.figure()
        plt.scatter(self.df["weight"],self.df["cost"],c=self.df["type"])
        plt.ylabel('cost (USD)')
        plt.xlabel('weight')
        plt.title('Cost vs Weight A')

        plt.plot(self.df["weight"],y,'y-')

        plt.show()

    def predict(self):
        self._normalize()
        train,test = self._select_training_testing(0.25)
        return self._train(train_data=train,lc=0.001,k=0.2,max_ite= 5000,target_error = 0.00001, nw =3)
        

def main():
    perc = perceptron('groupA.txt')
    w = perc.predict()
    perc.plot(w)
    

main()