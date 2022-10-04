import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

class perceptron:

    def __init__(self,fname):
        self.fname = fname
        self.df = pd.read_csv(self.fname,header=None, names = ["cost", "weight", "type"])
        #going to use to plot all weights
        self.weights_arr = []
        self.predicted_correct = 0
        self.predicted_incorrect = 0
        self.weights_df = pd.DataFrame(columns = ["x1", "x2", "b"])
        self.te = 0
        self.train = None
        self.test = None
 

    def _normalize(self):    
        self.df['cost'] = (self.df['cost']-self.df['cost'].min())/(self.df['cost'].max()-self.df['cost'].min())
        self.df['weight'] = (self.df['weight']-self.df['weight'].min())/(self.df['weight'].max()-self.df['weight'].min())

    def _select_training_testing(self, test_sz):
        train,test = train_test_split(self.df,train_size = 1-test_sz, test_size=test_sz)
        return (train,test)

    def _output_soft(self,net,k):
        #equation from the notes
        return (2/(1+np.exp(-2*k*net))) - 1

    #pass in learning constant, gain, number iterations, total error goal
    def _train_soft(self,train_data = pd.DataFrame,lc = 0.005,k = 0.2,max_ite = 5000, target_error = 0.001, nw = 3):
        #weights and bias array -setting random weights -0.5 - 0.5
        wb = np.random.uniform(size = nw, low = -0.5, high = 0.5)
        bias = 1.0
        print(f"inital weights: {wb}")
        #num inputs = num weights this is entirely unnecessary
        ni = nw
        #seperating the patterns from the output
        train = train_data[["weight","cost"]].to_numpy()
        dout = train_data['type'].to_numpy()
        TE = 0.0
        for _ in range(max_ite):
            out = 0.0
            self.weights_arr.append(wb)
            #bias = bias*wb[2]
            for (idx,p) in enumerate(train):
                net = 0.0

                #this is the input data with a bias
                pattern = [p[0],p[1],bias]

                #finding sum - could maybe just be the dot product
                for i in range(0,ni):
                   net = net + wb[i]*pattern[i]
                
                #shorthand way of doing the loop from above
                #net = np.dot(pattern,wb)

                #find the output based on the net
                out = self._output_soft(net,k)
                
                #error to update learning
                err = dout[idx] - out

                #calculate the total error for this iteration
                TE += (err)**2
                #learning rate = alpha * the error
                learn = lc * err

               #print(f"pattern = {pattern} wb: {wb} error: {err} TE {TE} learn {learn} out {out[idx]}")
                for i in range(0,ni):
                    wb[i] = wb[i] + learn*pattern[i]
                    #print(wb[i])

            if TE <= target_error:
                self.te = TE
                break
            self.te = TE
            TE = 0.0

        return wb

    def _output_hard(self,net):
        if net >= 0:
            sign = 1
        else:
            sign = 0
        return sign
    
    #pass in learning constant, gain, number iterations, total error goal
    def _train_hard(self,train_data = pd.DataFrame,lc = 0.005,k = 0.2,max_ite = 5000, target_error = 0.001, nw = 3):
        #weights and bias array -setting random weights -0.5 - 0.5
        wb = np.random.uniform(size = nw, low = -0.5, high = 0.5)
        bias = 1.0
        print(f"inital weights (x1,x2,bias): {wb}")
        print(f"alpha {lc} gain {k} max iterations {max_ite} target_error total error {target_error}")
        #num inputs = num weights this is entirely unnecessary
        ni = nw
        #seperating the patterns from the output
        train = train_data[["weight","cost"]].to_numpy()
        dout = train_data['type'].to_numpy()
        TE = 0.0
        for _ in range(max_ite):
            out = 0.0
            self.weights_arr.append(wb)
            #bias = bias*wb[2]
            for (idx,p) in enumerate(train):
                net = 0.0

                #this is the input data with a bias
                pattern = [p[0],p[1],bias]

                #finding sum - could maybe just be the dot product
                for i in range(0,ni):
                   net = net + wb[i]*pattern[i]
                
                #shorthand way of doing the loop from above
                #net = np.dot(pattern,wb)

                #find the output based on the net
                out = self._output_hard(net)
                
                #error to update learning
                err = dout[idx] - out

                #calculate the total error for this iteration
                TE += (err)**2
                #learning rate = alpha * the error
                learn = lc * err

               #print(f"pattern = {pattern} wb: {wb} error: {err} TE {TE} learn {learn} out {out[idx]}")
                for i in range(0,ni):
                    wb[i] = wb[i] + learn*pattern[i]
                    #print(wb[i])

            if TE <= target_error and TE > 0.0:
                self.te = TE
                break
            self.te = TE
            TE = 0.0

        return wb
    
    def _test(self,test_data,weights,gain):
        test = test_data[["weight","cost"]].to_numpy()
        actual = test_data['type'].to_numpy()
        self.predicted_correct = 0
        self.predicted_incorrect = 0
        bias = 1.0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for (idx,p) in enumerate(test):
            pattern = [p[0],p[1],bias]
            #finding sum
            net = np.dot(pattern,weights)
            #finding output
            out = self._output_soft(net,gain)
            #testing the outputs
            if out > 0.0 and actual[idx] == 1:
                self.predicted_correct += 1
                true_positive += 1
            elif out <= 0.0 and actual[idx] == 0:
                self.predicted_correct += 1
                true_negative += 1
            elif out <= 0.0 and actual[idx] == 1:
                false_positive += 1
                self.predicted_incorrect += 1
            elif out > 0 and actual[idx] == 0:
                false_negative += 1
                self.predicted_incorrect += 1
        
        print(f"predicted_correct {self.predicted_correct} predicted incorrect {self.predicted_incorrect}")
        print(f"true negative: {true_negative}")
        print(f"false negative: {false_negative}")
        print(f"true positive: {true_positive}")
        print(f"false positive: {false_positive}")

    def line_equatuation(self,weights):
        lines = []
        #assumes weights = [x1*weight,x2*cost,bias]
        for i in range(len(weights)):
            m = -(weights[i][0]/weights[i][1])
            #y intercept
            b = -(weights[i][2]/weights[i][1])
            lines.append([m,b])

        return lines

    def plot_final_weights(self,weights):
        #weights: [x1,x2,bias]
        xint = (-weights[2]/weights[1],0)
        yint = (0,-weights[2]/weights[0])

        #slope
        m = -(weights[0]/weights[1])

        #y intercept
        b = -(weights[2]/weights[1])

        #get values to plot
        vals = [(m * i) + b for i in self.df['weight']]

        plt.figure()
        plt.scatter(self.df["weight"],self.df["cost"],c=self.df["type"])
        plt.ylabel('cost (USD)')
        plt.xlabel('weight')
        plt.title(f"Cost vs Weight {self.fname}")
        plt.plot(self.df['weight'],vals,'b')
        #plt.plot(xint,yint)
        plt.show()

    def _plot_all(self,data,title = '{self.fname}'):
        
        fig,ax = plt.subplots(1,1)
        ax.set_xlim(0,1.2)
        ax.set_ylim(0,1.2)
        plt.scatter(data["weight"],data["cost"],c=data["type"])
        plt.ylabel('cost (USD)')
        plt.xlabel('weight')
        plt.title(title)
        x_lin = np.linspace(0.0,1.0,data.shape[0])
        lines = self.line_equatuation(self.weights_arr)
        for idx,line in enumerate(lines):
            m,b = line

            if idx == len(lines)-1:
                ax.plot(x_lin,m*x_lin + b, c = 'k', ls = '-', lw = 2)
            else:
                ax.plot(x_lin,m*x_lin + b, c = 'r', ls = '--', lw = 1.5)
        
        plt.show()
    
    def plot_all(self):
        self._plot_all(self.test, title = f"Cost vs Weight {self.fname} test data")
        self._plot_all(self.train, title = f"Cost vs Weight {self.fname} train data")


    def predict(self,type = 'soft', lc=0.1,k=0.2,max_ite= 5000,target_error = 0.01, nw = 3):
        self._normalize()
        self.train,self.test = self._select_training_testing(0.25)
        w = []
        if type == 'soft':
            w = self._train_soft(train_data=self.train,lc=lc,k=k,max_ite= max_ite,target_error = target_error, nw = nw)
        elif type == 'hard':
            w = self._train_hard(train_data=self.train,lc=lc,k=k,max_ite= max_ite,target_error = target_error, nw = nw)
        self._test(self.test,w,k)
        return w

#
    def save(self,fname):
        self.df.to_csv(fname,index = False)
    
    #save a csv of the weights
    def save_weights(self):
        self.weights_df = pd.DataFrame(self.weights_arr,columns = ["x1", "x2", "b"])
        self.weights_df.to_csv('a_weights.csv',index = False)

def main():
    perc = perceptron('groupA.txt')
    w = perc.predict(type = 'hard',lc=0.1,k=0.2,max_ite= 5000,target_error = 0.00001, nw = 3)
    print(f"weights: {w} TE: {perc.te}")
    perc.plot_all()
    #perc.save_weights()

if __name__ == '__main__':
    main()