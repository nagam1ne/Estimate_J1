import numpy as np
import matplotlib.pyplot as plt

def plot_losses(np_cross_scores,optimizers_nm,activation,nn_n_hidden,vae_n_hidden,n_z):
    for ind,np_cross_score in enumerate(np_cross_scores):
        print np.array(np.array(np_cross_score).T[0].tolist()).mean(axis=0).min()
        print np.array(np.array(np_cross_score).T[1].tolist()).mean(axis=0).min()
        print np.array(np.array(np_cross_score).T[2].tolist()).mean(axis=0).min()
        print np.array(np.array(np_cross_score).T[3].tolist()).mean(axis=0).min()
        print np.array(np.array(np_cross_score).T[4].tolist()).mean(axis=0).min()
        print("*     *     *     *     *     *     *     *     *     *     *")
        print np.array(np.array(np_cross_scores[0]).T[4].tolist()).mean(axis=0)
        print "++++++++++++++++++++++++++++++"
        

        fig = plt.figure(figsize=[20,5])
        ax_pre = fig.add_subplot(121)
        ax_pre.plot(np.array(np.array(np_cross_score).T[0].tolist()).mean(axis=0))
        ax_pre.plot(np.array(np.array(np_cross_score).T[1].tolist()).mean(axis=0))
        ax_pre.set_ylim([0,2])

        ax_tune = fig.add_subplot(122)
        ax_tune.plot(np.array(np.array(np_cross_score).T[2].tolist()).mean(axis=0))
        ax_tune.plot(np.array(np.array(np_cross_score).T[3].tolist()).mean(axis=0))
        ax_tune.plot(np.array(np.array(np_cross_score).T[4].tolist()).mean(axis=0))
        ax_tune.axhline(y=np.array(np.array(np_cross_score).T[4].tolist()).mean(axis=0).min())
        ax_tune.set_ylim([0,2])
        
        axes = []
        axes.append(ax_tune)
        #fig.suptitle("{},{},{}hidden,{}hidden,{}z".format(optimizers_list[ind],activation,nn_n_hidden,vae_n_hidden,n_z),fontsize=20)
    return axes