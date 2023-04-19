import lab04_part2 as lb04
import sklearn.datasets as sk 
import numpy as np
import scipy as sc

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

def MVG_model(D,L):
    c0 = []
    c1 = []
    c2 = []
    means = []
    S_matrices = []
    
    for i in range(D.shape[1]):
        if L[i] == 0:
            c0.append(D[:,i])
        elif L[i] == 1:
            c1.append(D[:,i])
        elif L[i] == 2:
            c2.append(D[:,i])
    

    c0 = (np.array(c0)).T
    c1 = (np.array(c1)).T        
    c2 = (np.array(c2)).T    

    c0_cent = lb04.centerData(c0)
    c1_cent = lb04.centerData(c1)
    c2_cent = lb04.centerData(c2)
    
    #you can find optimizations for this part in Lab03
    
    S_matrices.append(lb04.createCenteredCov(c0_cent)) 
    S_matrices.append(lb04.createCenteredCov(c1_cent))
    S_matrices.append(lb04.createCenteredCov(c2_cent))         
    
    means.append(lb04.vcol(c0.mean(1)))
    means.append(lb04.vcol(c1.mean(1)))
    means.append(lb04.vcol(c2.mean(1)))
    
    
    return means,S_matrices,(c0.shape[1],c1.shape[1],c2.shape[1])

def TCG_model(D,L):

    S_matrix = 0
    means,S_matrices,cN = MVG_model(D, L)
    
    cN = np.array(cN)
    
    S_matrices = np.array(S_matrices)
    
    D_cent = lb04.centerData(D)
    
    for i in range(cN.shape[0]):
        
        S_matrix += cN[i]*S_matrices[i]  
    
    S_matrix /=D.shape[1]
    
    return means,S_matrix
    

def loglikelihoods(DTE,means,S_matrices):
    ll0 = []
    ll1 = []
    ll2 = []
    
    for i in range(DTE.shape[1]):
            ll0.append(lb04.loglikelihood(DTE[:,i:i+1] , means[0], S_matrices[0]))
        
            ll1.append(lb04.loglikelihood(DTE[:,i:i+1], means[1], S_matrices[1]))
            
            ll2.append(lb04.loglikelihood(DTE[:,i:i+1], means[2], S_matrices[2]))    
        
    
    return np.array((ll0, ll1,ll2))


def posterior_prob(SJoint):
    
    # Calcola le densità marginali sommando le probabilità congiunte su tutte le classi
    SMarginal = lb04.vrow(SJoint.sum(axis=0))
    
    # Calcola le probabilità posteriori di classe dividendo le probabilità congiunte per le densità marginali
    SPost = SJoint / SMarginal
    
    # Calcola l'array delle etichette previste utilizzando il metodo argmax con la parola chiave axis
    pred = np.argmax(SPost, axis=0)
          
    return pred

def log_post_prob(log_SJoint):
        
    log_SMarginal_sol = np.load('logMarginal_MVG.npy')
    log_SMarginal = lb04.vrow(sc.special.logsumexp(log_SJoint,axis=0))
    #print(np.abs(log_SMarginal - log_SMarginal_sol).max())
    
    #print(log_SMarginal.shape)
    #print(log_SJoint.shape)
    log_SPost = log_SJoint - log_SMarginal
    #log_SPost_sol = np.load('logPosterior_MVG.npy')
    
    log_pred = np.argmax(log_SPost,axis=0)
        
    return log_pred
    

def evaluation(pred,LTE) : 
    
    mask = (pred==LTE)
    
    mask= np.array(mask,dtype=bool)
    
    corr = np.count_nonzero(mask)
    tot = LTE.shape[0]
    
    acc = float(corr)/tot
    
    return acc,tot-corr

def MVG_approach(D,L,DTE):
    #remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes
    
    #using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means,S_matrices,_ = MVG_model(D,L) #3 means and 3 S_matrices -> 1 for each class (3 classes)
    
    #we create a NxNc matrix with the log-likelihoods elements
    #each row represents a class and each column represents a sample
    #so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    log_score_matrix = loglikelihoods(DTE,means,S_matrices)
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(log_score_matrix)*Pc
    SJoint_sol = np.load('SJoint_MVG.npy')
    log_sm_joint = log_score_matrix + np.log(Pc)
    log_sm_joint_sol = np.load('logSJoint_MVG.npy')
    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterio probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)
    
    #simple function to evaluate the accuracy of our model
    acc,_ = evaluation(pred,LTE)  
    acc_2,_=evaluation(log_pred,LTE)
    inacc = 1-acc
    
    return log_pred



def NB_approach(D,L,DTE):
    #remember iris dataset is characterized by elements with 4 characteristics split up into 3 classes
    
    #using the functions built in lab04 we can create the several muc and Sc for the MVG model
    means,S_matrices,_ = MVG_model(D,L) #3 means and 3 S_matrices -> 1 for each class (3 classes)
    
    for i in range(np.array(S_matrices).shape[0]):
        S_matrices[i] = S_matrices[i]*np.eye(S_matrices[i].shape[0],S_matrices[i].shape[1])
    
    #we create a NxNc matrix with the log-likelihoods elements
    #each row represents a class and each column represents a sample
    #so S[i,j] represents the log_likelihood value for that j-th sample bound to the i-th class
    log_score_matrix = loglikelihoods(DTE,means,S_matrices)
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(log_score_matrix)*Pc
    SJoint_sol = np.load('SJoint_MVG.npy')
    log_sm_joint = log_score_matrix + np.log(Pc)
    log_sm_joint_sol = np.load('logSJoint_MVG.npy')
    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterio probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)
    
    #simple function to evaluate the accuracy of our model
    acc,_ = evaluation(pred,LTE)  
    acc_2,_=evaluation(log_pred,LTE)
    inacc = 1-acc
    
    return log_pred

def TCG_approach(D,L,DTE):
    
    
    means,S_matrix = TCG_model(D,L) #3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes
    
    #to recycle yet exiting code (loglikelihoods function), I generated a S_matrices variable cloning three times the S_matrix 
    S_matrices = [S_matrix,S_matrix,S_matrix]
    
    log_score_matrix = loglikelihoods(DTE,means,S_matrices)
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(log_score_matrix)*Pc
    SJoint_sol = np.load('SJoint_TiedMVG.npy')
    
    log_sm_joint = log_score_matrix + np.log(Pc)
    log_sm_joint_sol = np.load('logSJoint_TiedMVG.npy')
    
    log_marginal_sol = np.load('logMarginal_TiedMVG.npy')
    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterio probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)
    
    log_SPost_sol=np.load('logPosterior_TiedMVG.npy')
    SPost_sol=np.load('Posterior_TiedMVG.npy')
    
    #simple function to evaluate the accuracy of our model
    acc,_ = evaluation(pred,LTE)  
    acc_2,_=evaluation(log_pred,LTE)
    inacc = 1-acc

    return log_pred

def K_fold(D,L):
    
        MVG_err = 0
        NB_err=0
        TCG_err=0
    
        for i in range(D.shape[1]):
             #holdout method adopted for training set and evaluation set split
             DTE = D[:,i:i+1]  #1 sample of the dataset used for testing and the other for testing
             
             #find out how to delete a single sample!!!!
             DTR = np.delete(D,i,axis=1)
             LTR = np.delete(L,i)
             LTE = L[i:i+1]
             
             pred_LOO_MVG = MVG_approach(DTR,LTR,DTE)
             _,err = evaluation(pred_LOO_MVG,LTE)
             MVG_err += err
             
             pred_LOO_NB = NB_approach(DTR,LTR,DTE)
             _,err = evaluation(pred_LOO_NB,LTE)
             NB_err += err
             
             pred_LOO_TCG = TCG_approach(DTR, LTR, DTE)
             _,err = evaluation(pred_LOO_TCG,LTE)
             TCG_err+=err
          
        print(MVG_err*100/D.shape[1])
        print(NB_err*100/D.shape[1])
        print(TCG_err*100/D.shape[1])
        
        return 
    
    
if __name__ == "__main__":
   
    D,L = load_iris();
    (DTR,LTR),(DTE,LTE) = split_db_2to1(D, L)
    
    pred_MVG = MVG_approach(DTR,LTR,DTE)
    
    #for so few data we can apply the same code as used for MVG approach and make the S_matrices diagonal using a np.eye()
    pred_Naive_Bayes = NB_approach(DTR,LTR,DTE)
    
    pred_Tied_cov_Gauss = TCG_approach(DTR,LTR,DTE)
    
    #accuracy evaluation system
    print(evaluation(pred_MVG,LTE)[0]*100)
    print(evaluation(pred_Naive_Bayes,LTE)[0]*100)
    print(evaluation(pred_Tied_cov_Gauss,LTE)[0]*100)
    
    #let's try with K-FOLD EVALUATION system (Leave One Out variant)
    K_fold(D,L)
    
    
   
    
    
    
    



