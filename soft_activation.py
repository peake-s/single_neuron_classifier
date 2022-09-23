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

    def _output_soft(self,net,k):
        return 2/(1+np.exp(-2*k*net)) - 1

    def _select_tr(self):
        pass

    def _select_test(self):
        pass
    #use weights from the first assignment?
    def _train_soft(self,nump,gain,iterations = 5000,lc = 0.1,TE=0,k=0.0,weights = [0,0,0],pattern=[]):
        #selected data
        td = self._select_tr()
        td = self.normalized_df
        ni = 3
        k = gain
        w = weights
        nump = td.shape[0]
        alpha = lc
        desired_out = [] 
        #for designated number of iterations
        for ite in range(0,iterations):
            out = [0,0]
            #Loop over the amount of rows in the data
            for (idx,row) in enumerate(nump):
                net = 0
                for i in range(0,ni):
                    net = net + weights[i]*td[row][i]*alpha
                
                out[row] = self._output_soft(net,k)
                err = desired_out[row] - out[row]
                if err <= TE:
                    break 
                learn = alpha*err
                #update the weights
                for i in range(0,ni):
                    w[i] = w[i] + learn * td[row][i]


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

def main():
    #A: y = 1.03353 - 1.07843x
    # => y + 1.07843x < 1.03353
    pass

if __name__ == '__main__':
    main()