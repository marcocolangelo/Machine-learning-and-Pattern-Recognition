import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

def loadFile(fileName):
    matrix= np.zeros((150,4))
    labels = []
    hLabels = {
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }
    with open(fileName,'r') as f:
        for i,riga in enumerate(f):

            elementi=riga.split(',')[0:5]
            label=elementi.pop()
            label=hLabels[label[:-1]]
            labels.append(label)
            

            for j,elemento in enumerate(elementi):
                matrix[i][j] = float(elemento)

    return matrix.T,np.array(labels,dtype=np.int32)   #we want a 4x150 but initially we had a 150x4


def createCenteredSWc(DC):      #for centered data yet
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:,i:i+1],(DC[:,i:i+1]).T)
    C = C/float(DC.shape[1])
    return C


def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)    #broadcasting applied
    return DC



def createSBSW(D,L):
    D0 = D[:,L==0]      #we take all the D rows but only the columns which correspond to the same columns in L where the value is 0   
    D1 = D[:,L==1]
    D2 = D[:,L==2]

    DC0 = centerData(D0)
    DC1 = centerData(D1)
    DC2 = centerData(D2)

    SW0 = createCenteredSWc(DC0)
    SW1 = createCenteredSWc(DC1)
    SW2 = createCenteredSWc(DC2)

    centeredSamples = [DC0,DC1,DC2] 
    allSWc = [SW0,SW1,SW2]
    
    samples = [D0,D1,D2]
    mu = vcol(D.mean(1))

    SB=0
    SW=0

    for x in range(3):
        m = vcol(samples[x].mean(1))
        SW = SW + (allSWc[x]*centeredSamples[x].shape[1]) 
        SB = SB + samples[x].shape[1] * np.dot((m-mu),(m-mu).T)     #here we don't use centered samples because we apply a covariance between classed
                                                                    #and we take the mean off in the formula yet
        
    SB = SB/(float)(D.shape[1])
    SW = SW / (float)(D.shape[1])

    return SB,SW


def LDA1(D,L,m):

    SB, SW = createSBSW(D,L)        
    s,U = sp.linalg.eigh(SB,SW) #we use the scipy function which supports heigbert generalization eigenvectors scomposition 
    W = U[:,::-1][:,0:m]        #we must take the first m columns of U matrix

    return W

def LDA2(D,L,m):

    SB,SW = createSBSW(D,L)
    U,s,_ = np.linalg.svd(SW)
    P1 = np.dot(U,vcol(1.0/(s**0.5))*U.T)       #first transformation (whitening transformation) to apply a samples "CENTRIFICATION"
    SBtilde = np.dot(P1,np.dot(SB,P1.T))
    U,_,_ = np.linalg.svd(SBtilde)              
    P2 = U[:,0:m]                               #second tranformation (samples rotation) to obtain SB diagonalization

    return np.dot(P1.T,P2)


    



    


def plotCross(D,L):
    #the difference with the previous function is set on the way we plot. We use scattering to compare two different features between the several flower species

    D0 = D[:,L==0]      #we take all the D rows but only the columns which correspond to the same columns in L where the value is 0   
    D1 = D[:,L==1]
    D2 = D[:,L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    
    plt.figure()
    plt.scatter(D0[0,:],D0[1,:],label="Iris setosa")
    plt.scatter(D1[0,:],D1[1,:],label="Iris versicolor")
    plt.scatter(D2[0,:],D2[1,:],label="Iris verginica")
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    [D,L] = loadFile('D:\Desktop\I ANNO LM\II SEMESTRE\Machine Learning and Pattern recognition\ML - Lab\Lab04\iris.csv')
    
    W1 = LDA1(D,L,2)
    DW = np.dot(W1.T,D)     #D projection on sub-space W1
    plotCross(DW,L)         #plot function adapted to 2-D representation 

    W2 = LDA2(D,L,2)
    DW2 = np.dot(W2.T,D) 
    plotCross(DW2,L)         #plot function adapted to 2-D representation (plot maybe is flipped because of rotation in the second tranformation?)

    

