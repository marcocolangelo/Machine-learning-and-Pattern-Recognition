import numpy as np
import scipy as sc
import sklearn.datasets as sk

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

def load_iris():
    D,L = sk.load_iris()['data'].T,sk.load_iris()['target']
    return D,L


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
    

class logRegClass:
    def __init__(self,DTR,LTR,l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        
       
    
    def logreg_obj(self,v):
        #number of features
        D = self.DTR.shape[0]
        #number of samples
        n = self.DTR.shape[1]
        #number of classes
        K = np.max(self.LTR)+1
        
        #the function accepted only a 1-D array (called v here) 
            #so just now we are allowed to reshape W into a matrix  
        W = v[:D*K].reshape((D, K))
        b = v[D*K:]
        
        #T is the matrix with the n vectors zi (shape k)
        T = np.zeros((K,n))
        T[self.LTR, np.arange(n)] = 1
        #first J's addend 
        first = (self.l / 2) * np.linalg.norm(W) ** 2 
        
        # Compute the matrix of scores S
        S = np.dot(W.T,self.DTR) + b
        # Compute the log-sum-exp of the rows of S
        lse = np.log(np.sum(np.exp(S), axis=0))
        # Compute matrix Y_log
        Y_log = S - lse

        
        second = np.sum(T*Y_log)/n
        J = first - second
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
   
    
    #binary classificator with Logistic Regression applied to the Iris dataset
    l = 0.001
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    #x0 should be equal to DxK + 1
    x0 = np.zeros(DTR.shape[0] * (np.max(LTR)+1) + 1)
    
    
    logRegObj = logRegClass(DTR, LTR, l) #I created an object logReg with logreg_obj inside
    
    #optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
    #I set approx_grad=True so the function will generate an approximated gradient for each iteration
   
    params,f_min,_ = sc.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0,approx_grad=True)
    #the function found the coord for the minimal value of logreg_obj and they conrespond to w and b
    
    b = params[-1]
    
    w = np.array(params[0:-1]).reshape((DTR.shape[0],np.max(LTR)+1))
    
    S = np.zeros((np.max(LTE)+1,DTE.shape[1]))
    
   #I apply the model just trained to classify the test set samples
    for i in range(DTE.shape[1]):
        for c in range(np.max(LTE)+1):
            x = DTE[:,i:i+1]
            x = np.array(x)
            x = x.reshape((x.shape[0],1))
            S[c][i]=(np.dot(w[:,c:c+1].T,x) + b)
    
    pred = np.argmax(S, axis=0)
        
    #S = [i where if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
    
    
    
    acc = accuracy(pred,LTE)
    
    print(100-acc)