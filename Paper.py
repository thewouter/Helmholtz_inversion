import numpy as np
import pickle
import time
from scipy.stats import uniform, multivariate_normal
from Helmholtz import *
from Sequential_Monte_Carlo import *


def rotation_matrix(K, theta = np.pi/2): # Source algorithm: https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    # input vectors
    v1 = np.array(np.ones(K))
    v2 = np.array([i+2 for i in range(K)])
    # Gram-Schmidt orthogonalization
    n1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(n1,v2) * n1
    n2 = v2 / np.linalg.norm(v2)
    return np.identity(K) + (np.outer(n2, n1) - np.outer(n1, n2))*np.sin(theta) + (np.outer(n1,n1) + np.outer(n2,n2))*(np.cos(theta) - 1)

    
if __name__ == '__main__':
    freq = 2*10**9
    kwargs_data = {"freq" : freq, "h" : 0.5*np.sqrt((1/2**3)**2 * (10**9/(2*10**9))**3), "quad" : True, "char_len" : True, "s" : 0.2, "K" : 100, "data" : True}
    J = get_J(**kwargs_data)
    loc    = np.full(2*J, -1) 
    scale  = np.full(2*J, 2)
    Y_data = np.array([uniform.rvs(loc=loc[i], scale=scale[i]) for i in range(len(loc))]) # [loc[i], loc[i]+scale[i]]
    while np.sum((Y_data-(loc+0.5*scale))**2) < np.sum((0.5*scale)**2)*0.6:
        Y_data = np.array([uniform.rvs(loc=loc[i], scale=scale[i]) for i in range(len(loc))])
    helm_data = forward_observation(Y_data, **kwargs_data)
    print(helm_data)

    var       = abs(np.mean(helm_data)*0.01)
    eta       = multivariate_normal(mean=np.zeros(len(helm_data)), cov=var*np.eye(len(helm_data))).rvs()
    delta_1   = helm_data        # zero noise realisation
    delta_2   = helm_data + eta

    with open("Data/" + time.strftime("%Y%m%d-%H%M%S") + "_Parameters_Simulation.pickle", "wb") as file:
        pickle.dump(Y_data, file)
        pickle.dump(helm_data, file)
        pickle.dump(var, file)
        pickle.dump(eta, file)
        pickle.dump(delta_1, file)
        pickle.dump(delta_2, file)

    kwargs_inv = {"freq" : freq, "h" : np.sqrt((1/2**3)**2 * (10**9/freq)**3), "char_len" : True, "s" : 0.2, "K" : 100, "M" : 1000, "data" : False}
    
    smc_1 = Sequential_Monte_Carlo(delta_1, var, J)
    smc_1.SMC_algorithm(forward_observation, kwargs_inv)

    smc_2 = Sequential_Monte_Carlo(delta_2, var, J)
    smc_2.SMC_algorithm(forward_observation, kwargs_inv)


## Docker
# docker run --rm -ti -v c:\Users\safie\OneDrive\Documenten\Master_Thesis\Code:/opt/project wvharten/dolfinx:0.5.1
# singularity run -B Master_Thesis:/opt/project -B scratch:/scratch Singularity/dolfinx_complex.sif

#cd /opt/project
#python3 script.py