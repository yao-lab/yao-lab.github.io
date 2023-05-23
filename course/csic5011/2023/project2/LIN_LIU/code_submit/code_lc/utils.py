import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_theme()
def plot_path( gamma_path, save_name='age.png', n=5):
    # gamma_path
    first_time = np.zeros((len(gamma_path[0])))
    for i in range(1, len(gamma_path)):
        new_t = [k for k in range(len(gamma_path[0])) if gamma_path[i,k] !=0 and gamma_path[i-1, k]==0]
        if len(new_t) !=0:
            first_time[new_t] = 1/i
    # import pandas as pd
        
    x = pd.read_csv('college_prefer.csv')
    
    
    final_ind = np.argsort(-first_time)[:n]
    # print(first_time[final_ind])
    # print(final_ind)
    # ssss
    print(x.iloc[final_ind])
    # gamma_path_plot = gamma_path[:, final_ind]
    plt.figure()
    for ind in final_ind:
        plt.plot(np.array(range(gamma_path.shape[0]))/gamma_path.shape[0], gamma_path[:, ind], label=str(ind))
    plt.legend()
    plt.savefig(save_name)
    
def plot_new( non_zero_gamma,nzero_gamma,data_path,save_name='college_new.png'):
    # gamma_path

    data = pd.read_csv(data_path)
    x = data['left'].values
    y = data['right'].values
    

    plt.figure()
    plt.plot(x[non_zero_gamma], y[non_zero_gamma], 'r*', markersize=16,label='Outlier')
    plt.plot(x[nzero_gamma], y[nzero_gamma], 'b*', label='Normal')
    plt.xlabel('Left')
    plt.xlabel('Right')
    plt.legend()
    plt.savefig(save_name)
##################### knockoffs
