if __name__ == '__main__':
    from load_data import load_data
    from plot_losses import plot_losses
    import numpy as np

    train_x, train_y, train_real = load_data()

    all_x = np.array(train_x, dtype=np.float32)
    all_real = np.array(train_real, dtype=np.float32)
    all_y = np.array(train_y, dtype=np.float32).reshape(train_y.shape)
    
    nn_n_hidden = [50]
    vae_n_hidden = [50,50]
    
    grad_clip = 100
    noise_nm = 'none'

    n_z     = 100
    n_batch = 20
    nn_n_epochs  = 30
    vae_n_epochs  = 30
    n_epochs_tuning = 80
    optimizer_nm = 'Adam'
    activation = 'clipped_relu'
    n_iter = 10#32
    gpu=-1

    from sklearn.cross_validation import ShuffleSplit
    from sklearn.cross_validation import KFold
    from multiprocessing import Process, Queue
    import functools
    import math
    import time

    train_labels =[]
    test_labels =[]
    added_nb = all_x.shape[0]/2

    # rs = ShuffleSplit(added_nb, n_iter=n_iter, test_size=0.03, random_state=0)
    rs = KFold(n=added_nb, n_folds=n_iter, shuffle=True,random_state=4)

    for train_label, test_label in rs:
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        tmp_trains = np.array([],dtype='int')
        tmp_tests = np.array([],dtype='int')
        for cnt in range(2):
            tmp_trains =np.append(tmp_trains,train_label)
            train_label +=  added_nb
            tmp_tests =np.append(tmp_tests,test_label)
            test_label +=  added_nb
        train_labels.append(tmp_trains)
        test_labels.append(tmp_tests)
    
    nn_n_hidden_list = [[30,30]]#,[20,20],[50]]
    vae_n_hidden_list = [[20]]#,[20],[30,30]]
    n_z_list = [20]#,20,60]
    grad_clip_list = []
    nm_list = n_z_list
    
    queues = []
    processes = []
    cross_scores =[]

    for ind, hoge in enumerate(nm_list):
        queues.append(Queue())
        processes.extend([Process(target=run_nn_vae,
                              args=(queues[-1],optimizer_nm,np.array(all_x[train_index]),
                                    np.array(all_real[train_index]),np.array(all_y[train_index]),
                                    np.array(all_x[test_index]),np.array(all_y[test_index]),
                                    cnt,nn_n_hidden_list[ind],vae_n_hidden_list[ind],n_z_list[ind],n_batch, nn_n_epochs,
                                    vae_n_epochs,n_epochs_tuning,activation,grad_clip,noise_nm
                                   )) for cnt,[train_index, test_index] in enumerate(zip(train_labels, test_labels))])
        cross_scores.append([])
    list_length = ind +1

    start_time =time.time()
    for process in processes:    
        process.start()

    finish_processes =[]
    while True:
        for ind,process in enumerate(processes):
            if not ind in finish_processes and not process.is_alive():   
                queue_n = int(ind/n_iter)
                cross_scores[queue_n].append(queues[queue_n].get())
                finish_processes.append(ind)
        if len(finish_processes) == len(processes):
            break
    print('total time : {}s'.format(time.time() - start_time))
    
    np_cross_scores = np.array(cross_scores).reshape([list_length,n_iter,5])

    axes=plot_losses(np_cross_scores,optimizer_nm,activation,nn_n_hidden_list,vae_n_hidden_list,n_z_list)
    axes[0].set_ylim(0.6,1.5)
    axes[0].set_ylim(1.18,1.22)