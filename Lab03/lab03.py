import numpy as np
import matplotlib.pyplot as plt

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

def createCov(D,mu):   #I don't understand why this function returns a matrix different than the matrix returned by the one below
    mu = 0
    C = 0
    mu = D.mean(1)
    for i in range(D.shape[1]):
        C = C + np.dot(D[:, i:i+1] - mu, (D[:, i:i+1] - mu).T)  #scalar product using numpy
        #with this formule we have just centered the data (PCA on NON CENTERED DATA is quite an unsafe operation) 
    
    C = C / float(D.shape[1])   #where the divider is the dimension N of our data 
    return C

def createCenteredCov(DC):      #for centered data yet
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:,i:i+1],(DC[:,i:i+1]).T)
    C = C/float(DC.shape[1])
    return C

def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)    #broadcasting applied
    return DC

def createP(C,m):
    s, U = np.linalg.eigh(C)      #find eigen-values and eigen-vectors from the covariance matrix (C is always a square matrix)
                                  #remember that you need the first m values from U to obtain P
    P = U[:,::-1][:,0:m]          #this command uses U[:,::-1] to invert the U order and then take all the rows but only the first m columns
                                  #we have to invert the order because we are using the eigh command (only available for sqaure matrix)
                                  #and in this case the eig-values are placed in ascending order
                                  #if we want to use svd (SVC application) we should not invert the U order, just as seen during the lectures 
                                  #because the eig-values are placed in descending order in the s
    #U,S,Vh = np.linalg.svd(C)

    #P2 = U[:,0:m]
    #print(P)
    #print(P2)                    where P=P2

    return P


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

    for i in range(2):
        for j in range(2):      #in this way we can compare the various feature data of the same flower specie
            if i==j:
                continue
            plt.figure()
            plt.scatter(D0[i,:],D0[j,:],label="Iris setosa")
            plt.scatter(D1[i,:],D1[j,:],label="Iris versicolor")
            plt.scatter(D2[i,:],D2[j,:],label="Iris verginica")
            plt.legend()
            plt.show()

    return

if __name__ == "__main__" :
    D,L = loadFile('D:\Desktop\I ANNO LM\II SEMESTRE\Machine Learning and Pattern recognition\Lab\Lab03\iris.csv')
    DC = centerData(D)              #delete the mean component from the data
    C = createCenteredCov(DC)       #calculate the covariance matrix using centered data
    P = createP(C,2)                #we want to project our data on a 2-D sub-space so the second arg of our createP function will be 2 
    DP = np.dot(P.T, D)             #let's project our data on our P 2-D sub-space
                                    #now DP has 2x150 shape
                                    
    plotCross(DP,L)                 #plotting the data on a 2-D cartesian graph