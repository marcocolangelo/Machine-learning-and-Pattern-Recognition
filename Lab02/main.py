import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


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

    return matrix.T,np.array(labels,dtype=np.int32)   #we want a 4x150 but initially we have a 150x4

def plotSingle(D,L):
    #D[:,L==0] is a MASK on the COLUMNS to split out the data according to the flower specie

    D0 = D[:,L==0]      #we take all the D rows but only the columns which correspond to the same columns in L where the value is 0   
    D1 = D[:,L==1]
    D2 = D[:,L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    print(D0)
    print(D1)
    print(D2)

                  #plt.hist(Xaxis,100) #second parameter specifics the bins number, hence the number of white and blue bar used to represent elements frequency
                  #because the number of different elements inside the array is three, adopting less then 5 bins means less comprension of the situation  
    for i in range(4):
        #we repeat the loop 4 times to figure out the data about the four flowers features
        plt.figure()
        plt.xlabel(hFea[i])
        plt.ylabel("number of elements")
        plt.hist(D0[i,:],density=True,alpha = 0.7,label = "Iris-setosa")
        plt.hist(D1[i,:],density=True,alpha = 0.7,label = "Iris-versicolor")
        plt.hist(D2[i,:],density=True,alpha = 0.7,label = "Iris-verginica")
        plt.legend()
        plt.show()

    
    return 

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

    for i in range(4):
        for j in range(4):      #in this way we can compare the various feature data of the same flower specie
            if i==j:
                continue
            plt.figure()
            plt.xlabel(hFea[i])
            plt.ylabel(hFea[j])
            plt.scatter(D0[i,:],D0[j,:],label="Iris setosa")
            plt.scatter(D1[i,:],D1[j,:],label="Iris versicolor")
            plt.scatter(D2[i,:],D2[j,:],label="Iris verginica")
            plt.legend()
            plt.show()

    return

def stats(D,L):
    mean = 0

    #numpy offers a mean function to calculate the mean on a specific dimension (1 means columns, 0 means rows)
    mean = D.mean(1)

    #DC is a vector with data cleaned from mean values to center them
    DC = D - mean.reshape((D.shape[0],1))
    #look at how DC applies a BROADCASTING on D to do the minus operation successfully 

    return DC

if __name__ == "__main__":
    matrix,labels = loadFile('D:\Download\iris.csv')
    #plotSingle(matrix,labels)
    #plotCross(matrix,labels)
    DC = stats(matrix,labels)
    plotCross(DC,labels)



  
