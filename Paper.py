from Sequential_Monte_Carlo import *
import numpy as np
np.random.seed(seed=0)

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
    # freq = 2*10**9
    #
    # kwargs_data = {
    #     "freq" : freq, #  Frequency used
    #     "h" : 0.5*np.sqrt((1/2**3)**2 * (10**9/(2*10**9))**3), #  Spatial discretisation
    #     "quad" : True,
    #     "char_len" : True, #  Whether to use coefficient expansion (35), (34) if false
    #     "s" : 0.2, #  The correlation length in case of (35)
    #     "K" : 100, #  The number of measurement points
    #     "data" : True}
    # kwargs_data
    J = get_J(**kwargs_data)
    # loc    = np.full(2*J, -1)
    # scale  = np.full(2*J, 2)
    # Y_data = np.array([uniform.rvs(loc=loc[i], scale=scale[i]) for i in range(len(loc))]) # [loc[i], loc[i]+scale[i]]
    # while np.sum((Y_data-(loc+0.5*scale))**2) < np.sum((0.5*scale)**2)*0.6:
    #     Y_data = np.array([uniform.rvs(loc=loc[i], scale=scale[i]) for i in range(len(loc))])
    # Use fixed Y-data for reproducibility
    # Generated using `uniform.rvs(loc=-0.75, scale=1.5, size=200)'
    Y_data = np.array([-4.88460446e-01, -1.51959800e-01, -4.54540164e-01,  7.37406938e-01,
       -6.16521604e-01,  4.30568561e-01,  3.95987691e-01,  3.98187299e-01,
       -5.18689386e-01, -3.74723312e-01, -1.70585124e-01,  1.03662907e-01,
       -1.29049291e-01, -6.49483702e-01, -5.69211316e-01, -6.70326282e-01,
        3.32506300e-01, -2.71586716e-01, -2.98590629e-01, -7.45340479e-01,
        4.27742506e-01, -5.60785088e-01, -4.28271183e-02, -3.06299988e-01,
       -3.58225853e-01, -2.77937069e-01, -8.70501978e-02, -5.53583440e-01,
        6.78814864e-01, -7.05448980e-01,  4.87645250e-03, -9.50941329e-02,
       -1.69756416e-02,  2.16446437e-01, -5.25088222e-01, -5.58175670e-01,
       -2.66311360e-01,  3.45824574e-01, -7.13998389e-01,  1.52440081e-01,
       -2.06603480e-01, -6.28080638e-01, -3.27224886e-01, -3.97114450e-02,
        4.84048195e-01, -5.99172455e-01,  3.99811584e-01, -6.30955982e-01,
        6.63223921e-01,  4.62574996e-01, -4.73334841e-02,  3.41913528e-01,
       -3.33208222e-01,  2.70184673e-02, -6.82553396e-01,  2.44553768e-01,
        3.98341085e-01,  5.18372370e-01,  5.13782991e-01, -5.48788733e-02,
       -2.26176119e-02, -6.38536066e-01, -5.63261895e-01,  4.76455234e-01,
        6.02858143e-01,  3.31249157e-01, -5.28147940e-01, -4.72537102e-01,
        2.22335974e-01,  4.99641192e-03, -6.72911265e-01,  4.03512908e-01,
        6.51545248e-01,  4.15634895e-01,  5.36319990e-01, -5.88369981e-01,
        6.66155601e-01,  5.31117261e-01,  1.70789548e-01,  1.57092781e-01,
       -7.36751699e-01, -3.32856241e-01, -2.36177002e-02, -4.95615218e-01,
        2.71876639e-01, -3.53607073e-01,  1.29763337e-01,  6.20747193e-01,
        7.35735506e-01, -3.32477378e-02, -1.74541177e-01, -4.41939393e-01,
       -5.01758121e-01, -1.27673648e-01,  2.89867572e-01,  6.46332103e-01,
       -3.74771379e-01, -2.04668796e-01,  1.69625371e-01,  7.37370042e-01,
        1.51622665e-01,  7.08559807e-02, -1.46084728e-01, -5.40888717e-01,
       -6.33841993e-02, -1.22037806e-01, -5.68576242e-01, -4.47079272e-01,
       -7.93660226e-02,  1.59437866e-01,  5.86180268e-01,  3.44609619e-01,
        1.09405107e-01,  2.86618962e-02,  7.44428770e-01,  2.05400236e-01,
        4.21828612e-01, -7.42213575e-01, -1.98439083e-01,  5.57334442e-01,
        1.31818871e-01, -6.17113170e-01,  9.06057412e-02, -7.10300796e-01,
        4.67275772e-01, -4.84447994e-01,  1.89397601e-01, -4.93956478e-01,
       -2.26143228e-01, -6.33772523e-02, -7.05206667e-01, -4.16018818e-01,
       -1.65298724e-01, -1.92513301e-02,  1.81948783e-02, -6.56776095e-01,
       -7.85450446e-02, -4.18896702e-01, -6.98515448e-01, -3.31422037e-01,
        4.69449474e-01,  2.97148198e-02,  1.93704153e-01, -6.42026588e-01,
       -7.11955671e-01,  2.96919495e-01, -6.25056808e-01,  5.78066989e-01,
        7.18006569e-01,  2.41626418e-01,  2.28224211e-02, -2.72194541e-01,
        3.92370024e-01, -8.85117414e-02, -1.67637616e-01,  6.94778640e-01,
       -3.69726732e-01,  5.64821407e-01, -6.79129444e-04,  6.65071507e-01,
        6.55001047e-01,  2.57383513e-01,  1.08377311e-01, -2.55058301e-01,
       -3.44840601e-01,  5.69497269e-02, -2.69296998e-01, -3.30671988e-01,
       -3.00440803e-01, -2.86190263e-01,  3.79637630e-01,  5.43970459e-01,
        3.30586239e-01,  4.35698748e-01, -2.09968140e-01,  6.12526579e-01,
        4.22070103e-01,  4.87072511e-01, -1.55386036e-01, -2.06426346e-01,
        9.45688130e-02,  1.67744999e-01,  1.97517378e-01, -6.77386986e-02,
        2.77966421e-01, -7.11127318e-01,  2.70535483e-02,  1.76642346e-01,
       -2.30236989e-01, -6.08082269e-01,  9.80255912e-02,  7.75595956e-02,
       -4.70539891e-01,  3.57774428e-02,  6.99529466e-01, -2.75289011e-01,
        3.47603774e-01, -7.18267917e-01, -6.14228964e-01, -4.47704103e-01])
    Y_data = Y_data[:2 * J]

    helm_data = forward_observation(Y_data, **kwargs_data)
    print(helm_data)

    # var       = abs(np.mean(helm_data)*0.001)
    var       = 1e-3  # Fixed value
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

    # kwargs_inv = {"freq" : freq,
    #               "h" : np.sqrt((1/2**3)**2 * (10**9/freq)**3),
    #               "char_len" : True,
    #               "s" : 0.2,
    #               "K" : 100,
    #               "M" : 1000,
    #               "data" : False}
    #
    smc_1 = Sequential_Monte_Carlo(delta_1, var, J, **kwargs_inv)
    smc_1.SMC_algorithm(forward_observation, kwargs_inv)

    smc_2 = Sequential_Monte_Carlo(delta_2, var, J, **kwargs_inv)
    smc_2.SMC_algorithm(forward_observation, kwargs_inv)


## Docker
# docker run --rm -ti -v c:\Users\safie\OneDrive\Documenten\Master_Thesis\Code:/opt/project wvharten/dolfinx:0.5.1
# singularity run -B Master_Thesis:/opt/project -B scratch:/scratch Singularity/dolfinx_complex.sif

#cd /opt/project
#python3 script.py