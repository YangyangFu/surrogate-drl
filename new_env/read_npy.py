import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_one_ep(num_zone = 1, history_Z_T = None, history_Env_T = None, num_days = 7, fig_path_name = './simulation.png'):

    history_Z_T = history_Z_T#[:,0:180]
    history_Env_T = history_Env_T#[:,0:180]
    
    T_out = []
    T_zone = [[] for nz in range(num_zone)]
    
    T_out.extend(history_Env_T - 273.15)

    T_zone[0].extend(history_Z_T - 273.15)

    t = range(len(T_out))
    T_up = 26.0*np.ones([len(T_out)])
    T_low = 22.0*np.ones([len(T_out)])

    T_up = [30.0 for i in range(len(T_out))]
    T_low = [12.0 for i in range(len(T_out))]

    for i in range(num_days):
        for j in range((19-7)*4):
            T_up[i*24*4 + (j) + 4*7] = 26.0
            T_low[i*24*4 + (j) + 4*7] = 22.0

    colors = [[121/255.0,90/255.0,106/255.0],
        [91/255.0,131/255.0,28/255.0],
        [109/255.0,70/255.0,160/255.0],
        [18/255.0,106/255.0,118/255.0],
        [0/255.0,0/255.0,0/255.0]]
    plt.figure()

    plt.plot(t,T_out,'b')
    
    for nz in range(num_zone):
        plt.plot(t,T_zone[nz],color=colors[nz])
    plt.plot(t,T_up,'r',t,T_low,'r')
    plt.axis([0,len(T_out),10,40])
    plt.xlabel('Simulation step')
    plt.ylabel('Temperature')
    plt.grid()
    plt.show()
    plt.savefig(fig_path_name)

def plot_one_ep_normalized(num_zone = 1, history_Z_T = None, history_Env_T = None, his_cost = None, num_days = 7, fig_path_name = './simulation.png'):

    h = np.array([86400., 273.15+30, 273.15+40,1200., 1000.]+[273.15+40]*3+[1200.]*3)
    l = np.array([0., 273.15+12, 273.15+0,0, 0]+[273.15+0]*3+[0.0]*3)
    #ob = (ob - l)/(h-l)

    history_Z_T = history_Z_T*(30-12) + 273.15+12#[:,0:180]
    history_Env_T = history_Env_T*(40) + 273.15#[:,0:180]
    
    T_out = []
    T_zone = [[] for nz in range(num_zone)]
    
    T_out.extend(history_Env_T - 273.15)

    T_zone[0].extend(history_Z_T - 273.15)

    t = range(len(T_out))
    T_up = 26.0*np.ones([len(T_out)])
    T_low = 22.0*np.ones([len(T_out)])

    T_up = [30.0 for i in range(len(T_out))]
    T_low = [12.0 for i in range(len(T_out))]


    voilation_cnt = 0
    voilation_val = 0.0
    tot_len = len(T_zone[0])

    for i in range(num_days):
        for j in range((19-7)*4):
            tim = i*24*4 + (j) + 4*7
            T_up[tim] = 26.0
            T_low[tim] = 22.0
            if num_zone == 1:
                if T_zone[0][tim] > T_up[tim]:
                    voilation_cnt += 1
                    voilation_val += T_zone[0][tim] - T_up[tim]

                if T_zone[0][tim] < T_low[tim]:
                    voilation_cnt += 1
                    voilation_val += T_low[tim] - T_zone[0][tim]

            else:
                print("implementation of num of zone > 1 needed.")
    
    voilation_rate = voilation_cnt * 1.0 / tot_len
    voilation_val_mean = voilation_val * 1.0 / tot_len
    cost = - np.sum(his_cost)

    colors = [[131/255.0,175/255.0,155/255.0],
        [91/255.0,131/255.0,28/255.0],
        [39/255.0,70/255.0,220/255.0],
        [254/255.0,67/255.0,101/255.0],
        [38/255.0,188/255.0,213/255.0]]
    plt.figure()

    plt.title("vio_rate:"+str(format(voilation_rate, '.2f'))+", avg_vio:"+str(format(voilation_val_mean, '.2f'))+", cost:"+str(format(cost, '.2f'))) 

    plt.plot(t,T_out,color = colors[-1])
    
    for nz in range(num_zone):
        plt.plot(t,T_zone[nz],color=colors[nz])
    plt.plot(t,T_up,color = colors[-2])
    plt.plot(t,T_low,color = colors[-2])
    plt.axis([0,len(T_out),10,40])
    plt.xlabel('Simulation step')
    plt.ylabel('Temperature')
    plt.grid()
    plt.show()
    plt.savefig(fig_path_name)

    return voilation_rate, voilation_val_mean, cost

if __name__ == "__main__":
    l = np.load('./his_obs.npy', allow_pickle=True)
    l_indoor = l[:, 1]
    l_outdoor = l[:, 2]
    plot_one_ep(num_zone = 1, history_Z_T = l_indoor, history_Env_T = l_outdoor, fig_path_name = './simulation.png')

    l = np.load('./his_rew.npy', allow_pickle=True)
    print(l[100:120])