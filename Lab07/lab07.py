import numpy as np
import scipy as sc
import sklearn.datasets as sk

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

def load_iris_binary():
    D, L = sk.load_iris()['data'].T,sk.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D,L,seed=0):
    #holdout method adopted for training set and evaluation set split
    nTrain = int(D.shape[1]*2.0/3.0)  #66% of the dataset used for training
    np.random.seed(seed)    #seed generation 
    idx = np.random.permutation(D.shape[1]) #mask created to apply a random permutation for the split
    idxTrain = idx[0:nTrain]    #use idx to create a new mask useful to capture the training set part
    idxTest=idx[nTrain:] #same but for the evaluation set
    #now we apply the masks to the dataset
    DTR = D[:,idxTrain]
    DTE = D[:,idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR,LTR),(DTE,LTE)


def f(coord):
    x = pow(coord[0]+3,2) + np.sin(coord[0]) + pow(coord[1]+1,2)
    return x


def f_2(coord):
    x = pow(coord[0]+3,2) + np.sin(coord[0]) + pow(coord[1]+1,2)
    grad = np.array([2*(coord[0]+3) + np.cos(coord[0]), 2*(coord[1]+1)])
    return x,grad


    

class logRegClass:
    def __init__(self,DTR,LTR,l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        
        
    def logreg_obj(self,v):
        loss = 0
        #for each possible v value in the current iteration (which corresponds to specific coord
        #obtained by the just tracked movement plotted from the actual Hessian and Gradient values and the previous calculated coord)
        #we extrapolate the w and b parameters to insert in the J loss-function
        
        
        w,b = v[0:-1],v[-1]
        w = vcol(w)
        n = self.DTR.shape[1]
        regularization = (self.l / 2) * np.sum(w ** 2) 
        
        #it's a sample way to apply the math transformation z = 2c - 1 
        for i in range(n):
            
            if (self.LTR[i:i+1] == 1):
                zi = 1
            else:
                zi=-1
            loss += np.logaddexp(0,-zi * (np.dot(w.T,self.DTR[:,i:i+1]) + b))
        
        J = regularization + (1 / n) * loss
        
        return J
    
   
def accuracy (S,LTE):
        count = 0
        n = len(S)
        for i in range(n):
            if S[i] == LTE[i]:
                count += 1
        
        acc = float(count)/n * 100
        return acc
       

if __name__ == "__main__":
    #Numerical optimizations tries
    x,f_min,d = sc.optimize.fmin_l_bfgs_b(f, (0,0),approx_grad=True);
    
    x2,f_min2,d2 = sc.optimize.fmin_l_bfgs_b(f_2, (0,0))
    
    #binary classificator with Logistic Regression applied to the Iris dataset
    l = 0.000001
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    x0 = np.zeros(DTR.shape[0] + 1)
    
    
    logRegObj = logRegClass(DTR, LTR, l) #I created an object logReg with logreg_obj inside
    
    #optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
    #I set approx_grad=True so the function will generate an approximated gradient for each iteration
    params,f_min3,_ = sc.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0,approx_grad=True)
    
    #the function found the coord for the minimal value of logreg_obj and they conrespond to w and b
    
    b = params[-1]
    
    w = np.array(params[0:-1])
    
    S = []
    
   #I apply the model just trained to classify the test set samples
    for i in range(DTE.shape[1]):
        x = DTE[:,i:i+1]
        x = np.array(x)
        x = x.reshape((x.shape[0],1))
        S.append(np.dot(w.T,x) + b)
    
    S = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
    
    acc = accuracy(S,LTE)
    
    print(100-acc)