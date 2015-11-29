import time
import math
import sys
import cPickle as pickle
import copy
import os
import six
from NN_VAE import NN_VAE

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers, function
import chainer.functions as F

def run_nn_vae(q,optimizer_nm,train_x,train_real, train_y,test_x,test_y,cross_val,nn_n_hidden,
               vae_n_hidden, n_z, n_batch, nn_n_epochs,vae_n_epochs,n_epochs_tuning,activation,grad_clip,noise_nm,gpu=-1):
    
    # np.random.seed(123) # random値を固定
    
    n_x = train_x.shape[1]
    n_real = train_real.shape[1]
    n_y = train_y.shape[1]

    nn_n_layers = len(nn_n_hidden)
    
    vae_n_hidden_recog =vae_n_hidden
    vae_n_hidden_gen   = vae_n_hidden[::-1]
    vae_n_layers_recog = len(vae_n_hidden_recog)
    vae_n_layers_gen   = len(vae_n_hidden_gen)
    
    
    """NN pre_train"""

    layers = {}

    # Recognition model.
    nn_layer_sizes = [(n_x,nn_n_hidden[0])]
    if nn_n_layers >1:
        nn_layer_sizes += zip(nn_n_hidden[:-1], nn_n_hidden[1:])
    nn_layer_sizes += [(nn_n_hidden[-1], n_real)]

    for i, (n_incoming, n_outgoing) in enumerate(nn_layer_sizes):
        layers['nn_layer_%i' % i] = F.Linear(n_incoming, n_outgoing)

    """VAE pre_train"""

    # Recognition model.
    vae_rec_layer_sizes = [(n_real, vae_n_hidden_recog[0])]
    if vae_n_layers_recog >1:
        vae_rec_layer_sizes += zip(vae_n_hidden_recog[:-1], vae_n_hidden_recog[1:])
    vae_rec_layer_sizes += [(vae_n_hidden_recog[-1], n_z)]

    for i, (n_incoming, n_outgoing) in enumerate(vae_rec_layer_sizes):
        layers['vae_recog_%i' % i] = F.Linear(n_incoming, n_outgoing)

    layers['log_sigma'] = F.Linear(vae_n_hidden_recog[-1], n_z)

    # Generating model.
    vae_gen_layer_sizes = [(n_z, vae_n_hidden_gen[0])]
    if vae_n_layers_recog >1:
        vae_gen_layer_sizes += zip(vae_n_hidden_gen[:-1], vae_n_hidden_gen[1:])
    vae_gen_layer_sizes += [(vae_n_hidden_gen[-1], n_real)]

    for i, (n_incoming, n_outgoing) in enumerate(vae_gen_layer_sizes):
        layers['vae_gen_%i' % i] = F.Linear(n_incoming, n_outgoing)
        
    layers['output'] = F.Linear(n_z, n_y)

    model = NN_VAE(**layers)

    if gpu >= 0:
        cuda.init(gpu)
        model.to_gpu()
    
    # use Adam
    optimizers_dict = {
        "Adam":optimizers.Adam(),"AdaDelta":optimizers.AdaDelta(),
        "AdaGrad":optimizers.AdaGrad(),"MomentumSGD":optimizers.MomentumSGD(),
        "NesterovAG":optimizers.NesterovAG(),"RMSprop":optimizers.RMSprop(),
        "SGD":optimizers.SGD()
    }
    
    optimizer = optimizers_dict[optimizer_nm]
    optimizer.setup(model.collect_parameters())

    total_nn_losses = []
    
    if cross_val >=0:
        print('{}s pre-train start ...'.format(cross_val))
        
    # pre_train_NN start

    for epoch in xrange(1, nn_n_epochs + 1):
        t1 = time.time()
        
        # np.random.seed(123)
        indexes = np.random.permutation(train_x.shape[0])
        
        nn_total_loss = 0.0
        nn_out_list = np.zeros(train_real.shape)
        noisy_train_x = np.array(noisy(noise_nm,train_x),dtype = np.float32)
        for i in xrange(0, train_x.shape[0], n_batch):
            noisy_x_batch = noisy_train_x[indexes[i : i + n_batch]]
            real_batch = train_real[indexes[i : i + n_batch]]

            if gpu >= 0:
                noisy_x_batch = cuda.to_gpu(noisy_x_batch)

            optimizer.zero_grads()

            loss, nn_out = model.nn_forward(
                noisy_x_batch, real_batch, nn_n_layers,nonlinear=activation, gpu=-1,train=True
            )
            
            nn_total_loss += float(loss.data) * len(noisy_x_batch)
            loss.backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()
            nn_out_list[indexes[i : i + n_batch]] = nn_out.data

        total_nn_losses.append(nn_total_loss / train_x.shape[0])
        
    #  pre_train_VAE start
    
    total_vae_losses = []
    
    if cross_val >=0:
        print('{}s tuning start ...'.format(cross_val))
        
    nn_out_list = np.array(nn_out_list, dtype =np.float32)
    noisy_nn_out_list = np.array(noisy(noise_nm,nn_out_list),dtype = np.float32)
    

    for epoch in xrange(1, vae_n_epochs + 1):
        # np.random.seed(123)
        indexes = np.random.permutation(train_x.shape[0])
        total_loss = 0.0
        noisy_nn_out_list = np.array(noisy(noise_nm,nn_out_list),dtype = np.float32)
        for i in xrange(0, train_x.shape[0], n_batch):
            noisy_nn_out_list_batch = noisy_nn_out_list[indexes[i : i + n_batch]]
            nn_out_list_batch =nn_out_list[indexes[i : i + n_batch]]
            real_batch = train_real[indexes[i : i + n_batch]]
            
            if gpu >= 0:
                noisy_nn_out_list_batch = cuda.to_gpu(noisy_nn_out_list_batch)

            optimizer.zero_grads()

            rec_loss, kl_loss, output = model.vae_forward(
                noisy_nn_out_list_batch, real_batch, vae_n_layers_recog,
                vae_n_layers_gen, nonlinear_q=activation, nonlinear_p=activation,train=True)
            loss = rec_loss + kl_loss
            total_loss += float(loss.data) * len(noisy_nn_out_list_batch)
            loss.backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()
        total_vae_losses.append(total_loss /  train_x.shape[0])
    
    #  train_test_NN_VAE start

    total_nn_vae_losses = []
    total_test_losses = []
    total_train_losses = []
    if cross_val >=0:
        print('{}s tuning start ...'.format(cross_val))

    for epoch in xrange(1, n_epochs_tuning + 1):
        noisy_train_x = np.array(noisy(noise_nm,train_x),dtype = np.float32)
        #np.random.seed(123)
        indexes = np.random.permutation(train_x.shape[0])
        total_loss = 0.0
        for i in xrange(0, train_x.shape[0], n_batch):
            noisy_x_batch = noisy_train_x[indexes[i : i + n_batch]]
            y_batch = train_y[indexes[i : i + n_batch]]

            if gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            optimizer.zero_grads()

            loss, predict_score = model.nn_vae_tuning(
                noisy_x_batch, y_batch, nn_n_layers,vae_n_layers_recog,
                nonlinear_q=activation,train=True
            )
            loss = loss ** 0.5
            total_loss += float(loss.data) * len(noisy_x_batch)
            loss.backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()
        total_nn_vae_losses.append(total_loss /  train_x.shape[0])
        
        # test
        
        sum_loss_train = 0
        
        for i in xrange(0, train_x.shape[0], n_batch):
            x_batch = train_x[indexes[i : i + n_batch]]
            y_batch = train_y[indexes[i : i + n_batch]]

            if gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            loss, predict_score = model.nn_vae_tuning(
                x_batch, y_batch, nn_n_layers,vae_n_layers_recog,
                nonlinear_q=activation,train=False
            )
            loss = loss ** 0.5
            sum_loss_train += float(loss.data) * len(noisy_x_batch)
        total_train_losses.append(sum_loss_train/train_x.shape[0])

        x_batch = test_x
        y_batch = test_y

        loss, predict_score =  model.nn_vae_tuning(
            x_batch, y_batch, nn_n_layers,vae_n_layers_recog,
            nonlinear_q=activation, train=False
        )
        loss = loss ** 0.5
        total_test_losses.append(loss.data)
    q.put([total_nn_losses,total_vae_losses,total_nn_vae_losses,total_train_losses,total_test_losses])

import numpy as np
import random
def noisy(noise_typ,matrix):
    if noise_typ == "gauss":
        row,col= matrix.shape
        mean = 0
        var = 0.5
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = matrix + gauss
        return noisy
    elif noise_typ == "s&p":
        prob = 0.01
        output = np.zeros(matrix.shape,np.uint8)
        thres = 1 - prob 
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = matrix.min()
                elif rdn > thres:
                    output[i][j] = matrix.max()
                else:
                    output[i][j] =matrix[i][j]
        return output

    elif noise_typ == "poisson":
        vals = len(np.unique(matrix))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(matrix * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col = matrix.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = matrix + matrix * gauss
        return noisy
    elif noise_typ =="none":
        return matrix