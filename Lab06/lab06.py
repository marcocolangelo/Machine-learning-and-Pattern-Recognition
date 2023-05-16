import load as ld
import numpy as np
import scipy as sc

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

def string_list(data):
    string_list=[]
    
    for w in data:
        w = w.split(' ')
        for ww in w:    
            string_list.append(ww)

    return string_list

def split_trincets(data):
    tr_list=[]
    
   #you have to split the data into trincets
    for w in data.split():
        tr_list.append(w)
    
    return tr_list
            
#keys is the vector of total word inside the training set
#from_occurr is the vector of words from which reading the specific cantic occurrencies
def make_dict(keys,from_occurr):
    #e must be a very small value which should be evaluate on a VALIDATION SET but we adopt a fixed value for now
    e = 0.001
    D = dict.fromkeys(keys,0)
    
    for w1 in D.keys():
        if w1 in from_occurr:
            D[w1] = from_occurr.count(w1) 
    
    #we apply dirichelet a priori distribution approach to manage train data values non present in inf/pur/par train set 
    for w in D.keys():
        D[w] += e;
           
    return D

#it creates a dictionary with trincets as keys and dictionaries (with tr as keys and occurrencies as values) as values
def make_dict_tr_plus_occurr(train_dict,test_tr):
    tr_dict_tot={}
    for t in test_tr:
        evaluation_tr = t.split()
        eval_occurr = make_dict(train_dict, evaluation_tr)
        tr_dict_tot[t] = eval_occurr
    
    return tr_dict_tot


def normalization(dictionary):
    
    p = []
    parameters = dict.fromkeys(dictionary.keys(),dictionary.values())
    
    for w in dictionary.values():
        p.append(w)

    N = np.array(p).sum()
    
    p = np.array(p,dtype=float)/ N
    
    i=0
    
    for w in dictionary:
        parameters[w] = p[i]
        i+=1
        
    return parameters,N




def w_evaluation(parameters):
    
    w = []
    for pi in parameters.keys():
        w.append(np.log(parameters[pi]))
    
    return np.array(w)
    

def log_likelihood(test_trincet,w):
    #lista=[]
    #for x in test_trincet:
     #   lista.append(x)
    values = list(test_trincet.values())
    float_values = [float(value) for value in values]
    vettore = np.array(float_values,dtype=float).reshape(1,-1)
    w2 = np.array(w,dtype=float).reshape(-1,1) 
    print(vettore.shape)
    print(w2.shape)
    result = np.dot(vettore,w2)
    print(result.shape)
    return result


def posterior_prob(SJoint):
    
    # Calcola le densità marginali sommando le probabilità congiunte su tutte le classi
    SMarginal = vrow(SJoint.sum(axis=0))
    
    # Calcola le probabilità posteriori di classe dividendo le probabilità congiunte per le densità marginali
    SPost = SJoint / SMarginal
    
    # Calcola l'array delle etichette previste utilizzando il metodo argmax con la parola chiave axis
    pred = np.argmax(SPost, axis=0)
          
    return pred

def log_post_prob(log_SJoint):
        
    
    log_SMarginal = vrow(sc.special.logsumexp(log_SJoint,axis=0))
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

if __name__ == "__main__":
    e = 0.001
    
    
    lInf, lPur, lPar = ld.load_data()
    lInf_train, lInf_evaluation = ld.split_data(lInf, 4)
    lPur_train, lPur_evaluation = ld.split_data(lPur, 4)
    lPar_train, lPar_evaluation = ld.split_data(lPar, 4)
    
    train = []
    train = string_list(lInf_train + lPur_train + lPar_train)
    
    
    #here we just split up the trincets into single words
    lInf_train = string_list(lInf_train)
    lPur_train = string_list(lPur_train)
    lPar_train = string_list(lPar_train)
    
    
    #we compute the occurrencies for each word from the dictionaty into the class_training set 
    #the result will be a dictionary wich all the dictionary word set as keys and their occurrencies in the respective chapter(Inf,Pur,Par) as values
    inf_occurrencies = make_dict(train,lInf_train)
    pur_occurrencies = make_dict(train,lPur_train)
    par_occurrencies = make_dict(train,lPar_train)
    #we transform these values into frequencies using pi(c,j) = N(c,j)/Nc where j is the j-th word index
    inf_parameters,N_inf = normalization(inf_occurrencies)
    pur_parameters,N_pur = normalization(pur_occurrencies)
    par_parameters,N_par = normalization(par_occurrencies)
    
    
    #we simply map these frequencies in log_frequencies vectors w_inf,w_pur,w_par
    w_inf = w_evaluation(inf_parameters)    #size should always be 12002
    w_pur = w_evaluation(pur_parameters)
    w_par = w_evaluation(par_parameters)
    
    w_cantica=[]
    w_cantica.append(w_inf)
    w_cantica.append(w_pur)
    w_cantica.append(w_par)
    
    #--Test set formation--
    #here we must transform the test data into trincets
    #THIS METHOD IS WRONG BECAUSE IT SPLITS THE DATA INTO SINGLE WORDS!
    #BUT WE WANT TO FORM THE TRINCETS BEFORE
    #so I should form an array of trincets and for each trincets a dictionary with the words as key and the occurrencies as values????
   
    #for the test set the data are splitted into trincets to use as evaluating input (pred = y.T*wc)
    test_data = lInf_evaluation + lPur_evaluation + lPar_evaluation #we don't split them up into single words
        
        
    #we will use the test data as a long list of trincets from all the cantica
    #but we still have to find the single word occurrencies
    
    tr_dict_tot = make_dict_tr_plus_occurr(train,test_data)
    
    
    #--end test set formation--
    
    
    #S is a matrix where rows represent the 3 classes (3 cantica) and each column is a trincets
    #S(i,j) represents the prob of the j-th trincet to be part of the i-th cantica
    S = np.zeros((3,len(test_data)),dtype=float)
    i=0
    j=0
    
    for i in range(3):
        j=0
        for tr in tr_dict_tot.values():
            S[i:i+1,j:j+1] = log_likelihood(tr,w_cantica[i])
            j+=1    
    
    
    
    #adopting broadcasting we can compute JOINT DISTRIBUTION PROBABILITY fx,c(xt,c) = fx|c(xt|c)*Pc(c)
    Pc = 1/3 #we assume all class' Pc(c) are the same 
    #for a misunderstanding thing whit the cov-matrix I called the S matrix sm_joint 
    sm_joint = np.exp(S)*Pc
    log_sm_joint = S + np.log(Pc)
    
    
    #let's compute the POSTERIOR PROBABILITY P(C=c| X = xt) = fx,c(xt,c)/sm_joint.sum(0)
    #be careful! These functions below return prediction labels yet! The Posterio probability 
    #computation is made inside the functions!
    pred = posterior_prob(sm_joint) 
    log_pred = log_post_prob(log_sm_joint)    
    
    
    #forse questo qua non serve, volevo calcolare semlicemente il denominatore per la class posterior prob ma ho preso la formula del lab05 (vedi sopra)
    #for j in len(tr_dict_tot.keys()):
     #   sum(S[:,j:j+1])
    

    #rimane solo da implementare l'evauation in teoria
   evaluation(pred, )
    
    
    