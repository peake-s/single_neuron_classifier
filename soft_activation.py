import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class perceptron:
    
    def __init__(self,fname):
        self.fname = fname
        self.df = pd.read_csv(self.fname,header=None, names = ["cost", "weight", "type"])
        self.normalized_df = {}

    def normalize(self):    
        self.normalized_df = self.df.copy()
        self.normalized_df[0] = (self.df[0]-self.df[0].min())/(self.df[0].max()-self.df[0].min())
        self.normalized_df[1] = (self.df[1]-self.df[1].min())/(self.df[1].max()-self.df[1].min())

    #performance reasons
    def vectorize(self):
        self.normalized_df.to_numpy()

    def _output_soft(self,net,k):
        return 2/(1+np.exp(-2*k*net)) - 1

    def _select_tr(self):
        pass

    def _select_test(self):
        pass

    def _train_soft(self,np,num_inp,gain,TE=0,k=0.0,weights = [],pattern=[]):
        #selected data
        td = self._select_tr()
        ni = 5000

    def _test(self):
        test = self._select_test()

    def total_error(self):
        pass

    def plot(self):
        line = []
        plt.figure()
        plt.scatter(self.normalized_df['weight'],self.normalized_df['cost'],c=self.normalized_df['type'])
        plt.xlabel('cost (USD)')
        plt.ylabel('weight') 
        plt.title('Weight vs Cost A')
        plt.savefig('../plots/plot_a.png')
