"""Variation AutoEncoder class"""
import os
import time
import numpy as np


from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F

class NN_VAE(FunctionSet):

    def __init__(self, **layers):
        super(NN_VAE, self).__init__(**layers)
    
    def nn_forward(self, x_data, real_data, n_layers ,nonlinear='relu', gpu=-1,train=True):
        inputs = Variable(x_data)
        real = Variable(real_data)
        
        # set non-linear function
        nonlinear_dict = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': F.softplus, 'relu': F.relu,
                     'clipped_relu': F.clipped_relu, 'leaky_relu': F.leaky_relu}
        nonlinear_f= nonlinear_dict[nonlinear]
        
        chain = [inputs]
        
        for i in range(n_layers):
            chain.append(F.dropout(nonlinear_f(getattr(self, 'nn_layer_%i' % i)(chain[-1])),train=train))
        nn_out = getattr(self, 'nn_layer_%i' % n_layers)(chain[-1])
        chain  += [nn_out]
        loss = F.mean_squared_error(nn_out, real)
        
        return loss, nn_out
    
    def vae_forward_whole(self, noisy_h_data, h_data,n_layers_recog, n_layers_gen,
                  nonlinear_q='softplus', nonlinear_p='softplus', gpu=-1,train=True):
        from random import gauss
        
        noisy_inputs = noisy_h_data # For whole
        inputs = Variable(h_data)

        # set non-linear function
        nonlinear_dict = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': F.softplus, 'relu': F.relu,
                     'clipped_relu': F.clipped_relu, 'leaky_relu': F.leaky_relu}
        nonlinear_f_q = nonlinear_dict[nonlinear_q]
        nonlinear_f_p = nonlinear_dict[nonlinear_p]

        chain = [noisy_inputs]

        # compute q(z|x, y)
        for i in range(n_layers_recog):
            chain.append(F.dropout(nonlinear_f_q(getattr(self, 'vae_recog_%i' % i)(chain[-1])),train=train))

        recog_out = getattr(self, 'vae_recog_%i' % n_layers_recog)(chain[-1])

        log_sigma_out = 0.5 * (getattr(self, 'log_sigma')(chain[-1]))
        
        # np.random.seed(123)

        eps = np.random.normal(0, 1, (inputs.data.shape[0], log_sigma_out.data.shape[1])).astype('float32')
        if gpu >= 0:
            eps = cuda.to_gpu(eps)
        eps = Variable(eps)
        z   = recog_out + F.exp(log_sigma_out) * eps

        chain  += [recog_out, z]

        for i in range(n_layers_gen):
            chain.append(F.dropout(nonlinear_f_p(getattr(self, 'vae_gen_%i' % i)(chain[-1])),train=train))

        # chain.append(F.sigmoid(getattr(self, 'vae_gen_%i' % (n_layers_gen))(chain[-1])))
        chain.append(getattr(self, 'vae_gen_%i' % (n_layers_gen))(chain[-1]))
        output = chain[-1]

        rec_loss = F.mean_squared_error(output, inputs)
        KLD = -0.5 * F.sum(1 + log_sigma_out - recog_out**2 - F.exp(log_sigma_out)) / (inputs.data.shape[0]*inputs.data.shape[1])

        return rec_loss, KLD, output
    
    def vae_forward(self, noisy_h_data, h_data,n_layers_recog, n_layers_gen,
                  nonlinear_q='softplus', nonlinear_p='softplus', gpu=-1,train=True):
        from random import gauss
        
        noisy_inputs = Variable(noisy_h_data) #  For non-whole
        inputs = Variable(h_data)

        # set non-linear function
        nonlinear_dict = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': F.softplus, 'relu': F.relu,
                     'clipped_relu': F.clipped_relu, 'leaky_relu': F.leaky_relu}
        nonlinear_f_q = nonlinear_dict[nonlinear_q]
        nonlinear_f_p = nonlinear_dict[nonlinear_p]

        chain = [noisy_inputs]

        # compute q(z|x, y)
        for i in range(n_layers_recog):
            chain.append(F.dropout(nonlinear_f_q(getattr(self, 'vae_recog_%i' % i)(chain[-1])),train=train))

        recog_out = getattr(self, 'vae_recog_%i' % n_layers_recog)(chain[-1])

        log_sigma_out = 0.5 * (getattr(self, 'log_sigma')(chain[-1]))
        
        # np.random.seed(123)

        eps = np.random.normal(0, 1, (inputs.data.shape[0], log_sigma_out.data.shape[1])).astype('float32')
        if gpu >= 0:
            eps = cuda.to_gpu(eps)
        eps = Variable(eps)
        z   = recog_out + F.exp(log_sigma_out) * eps

        chain  += [recog_out, z]

        for i in range(n_layers_gen):
            chain.append(F.dropout(nonlinear_f_p(getattr(self, 'vae_gen_%i' % i)(chain[-1])),train=train))

        # chain.append(F.sigmoid(getattr(self, 'vae_gen_%i' % (n_layers_gen))(chain[-1])))
        chain.append(getattr(self, 'vae_gen_%i' % (n_layers_gen))(chain[-1]))
        output = chain[-1]

        rec_loss = F.mean_squared_error(output, inputs)
        KLD = -0.5 * F.sum(1 + log_sigma_out - recog_out**2 - F.exp(log_sigma_out)) / (inputs.data.shape[0]*inputs.data.shape[1])

        return rec_loss, KLD, output
    
    def nn_vae_tuning(self, x_data, y_data, nn_n_layers, vae_n_layers_recog, 
                      nonlinear_q='softplus', gpu=-1,train=True):
        
        inputs = Variable(x_data)
        y = Variable(y_data)
        
        nonlinear_out = 'relu'

        # set non-linear function
        nonlinear_dict = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': F.softplus, 'relu': F.relu,
                     'clipped_relu': F.clipped_relu, 'leaky_relu': F.leaky_relu,'PReLU' : F.PReLU(True)}
        nonlinear_f_q = nonlinear_dict[nonlinear_q]
        nonlinear_f_out = nonlinear_dict[nonlinear_out]

        chain = [inputs]

        # compute q(z|x, y)
        for i in range(nn_n_layers):
            chain.append(F.dropout(nonlinear_f_q(getattr(self, 'nn_layer_%i' % i)(chain[-1])),train=train))
        nn_out = getattr(self, 'nn_layer_%i' % nn_n_layers)(chain[-1])
        chain  += [nn_out]
        
        for i in range(vae_n_layers_recog):
            chain.append(F.dropout(nonlinear_f_q(getattr(self, 'vae_recog_%i' % i)(chain[-1])),train=train))
        recog_out = getattr(self, 'vae_recog_%i' % vae_n_layers_recog)(chain[-1])
        log_sigma_out = 0.5 * (getattr(self, 'log_sigma')(chain[-1]))
        
        # np.random.seed(123)

        eps = np.random.normal(0, 1, (inputs.data.shape[0], log_sigma_out.data.shape[1])).astype('float32')
        if gpu >= 0:
            eps = cuda.to_gpu(eps)
        eps = Variable(eps)
        z   = recog_out + F.exp(log_sigma_out) * eps
        predict_score = nonlinear_f_out(getattr(self, 'output')(z))
        
        mean_error =  F.mean_squared_error(predict_score, y)
        
        return mean_error, predict_score