import utils, torch, time, os, pickle
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class AIS(object):
    def __init__(self,x_test,params_posterior,decoder,z_current,args,eps_scale=None):
        self.x_in = x_test
        self.params_posterior_in = params_posterior
        self.decoder = decoder
        self.z_current = z_current#sample
        self.args = args
        self.gpu_mode = args.gpu_mode
        self.num_sam = args.num_sam
    
      
        eps=self.args.test_ais_eps
        self.build_model(eps)
    def build_model(self,eps):
        self.batch_size= self.args.batch_size
        output_size= self.args.output_size
        c_dim = self.args.c_dim
        z_dim = self.args.z_dim
        self.x=Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam,784])))
        self.params_posterior = [
            Variable(torch.zeros(p0.size()).cuda(),requires_grad=False)
            for p0 in self.params_posterior_in
        ]
        self.eps_scale = Variable(torch.Tensor(np.ones([self.batch_size*self.num_sam, z_dim])).cuda(),requires_grad=False)
        self.mass = 1.#/self.var0
        mass_sqrt = 1.#/self.std0
   
       
        if self.gpu_mode:
            self.z = Variable(torch.zeros([self.batch_size*self.num_sam, z_dim]).cuda())
            self.p = Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam, z_dim])).cuda())

            self.z_current = Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam, z_dim])).cuda())
            self.p_current = Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam, z_dim])).cuda())
            self.p_rnd = Variable(torch.randn([self.batch_size*self.num_sam, z_dim]).cuda() * mass_sqrt)
        else:
            self.z = Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam, z_dim])),requires_grad=False)
            self.p_rnd = torch.randn([self.batch_size*self.num_sam, z_dim])* mass_sqrt
            self.p = Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam, z_dim])))
            self.z_current = Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam, z_dim])),requires_grad=False)
            self.p_current = Variable(torch.Tensor(np.zeros([self.batch_size*self.num_sam, z_dim]),requires_grad=False))
            
        
        #self.eps = torch.FloatTensor() 
        #self.beta = torch.FloatTensor()
        self.x=self.x_in
        
        # Hamiltoninan
        self.U = self.get_energy(self.z,1)#self.z, beta
        self.V = 0.5 * torch.sum(self.p**2/self.mass, dim=1)
        self.H = self.U + self.V
        self.U_current = self.get_energy(self.z_current,1)#self.z_current, beta
        self.V_current = 0.5 * torch.sum(self.p_current**2/self.mass, dim=1)
        self.H_current = self.U_current + self.V_current
        
        
        # Intialize
        #self.init_batch = [
        #    self.x.assign(self.x_in),
        #   self.eps_scale.assign(self.eps_scale_in)]
       
     
        for (p, p_in) in zip(self.params_posterior, self.params_posterior_in):
            p = p_in
        
        #self.z_current= #self.get_z0(self.params_posterior,self.args.batch_size,self.args.z_dim)
        self.init_hmc =  self.z_current
        self.p_current= self.p_rnd
        self.init_hmc_step = [self.p_current]
        self.z=self.z_current
        self.p=self.p_current
        self.init_hmc_step2 = [
           self.z,
            self.p,
        ]
        # Euler steps
        eps_scaled = self.eps_scale.mul(eps)
        self.z=self.z.add_( eps_scaled.mul(self.p).div(self.mass))
        self.euler_z = self.z
        gradU=[]
        import pdb;pdb.set_trace()
        for i in range(self.U.shape[0]):
            gradU.append(torch.autograd.grad(self.U[i], self.z))
        
        gradU=torch.stack(gradU)
        self.p=self.p-eps_scaled * gradU
        self.euler_p = self.p

        # Accept
         #self.is_accept = torch.cast(tf.random_uniform([batch_size]) < toch.exp(self.H_current - self.H), tf.float32)
        self.is_accept = torch.cast(torch.FloatTensor([batch_size]).uniform_()< toch.exp((self.H_current - self.H),torch.cuda.FloatTensor))
        self.accept_rate = torch.mean(self.is_accept)

        is_accept_rs = self.is_accept.view([batch_size, 1])
        self.update_z = self.z_current.assign(
            is_accept_rs * self.z + (1. - is_accept_rs) * self.z_current
        )
    
    def get_hamiltonian(beta,):
            
        self.U = self.get_energy(self.z, beta)
        self.V = 0.5 * torch.sum((self.p*self.p)/self.mass, dim=1)
        return self.U + self.V
    
    def get_hamiltonian_cur(beta):
        self.U_current = self.get_energy(self.z_current,beta)
        self.V_current = 0.5 * torch.sum((self.p_current*self.p_current)/self.mass, dim=1)
        return self.U_current + self.V_current
    
    
    def update_z(self, beta):
          
        # Accept 
        #TODO Convert tf.random_uniform([batch_size]) to torch code
        H = self.get_hamiltonian(beta)
        H_current = self.get_hamiltonian_cur(beta)
        is_accept = torch.cast(torch.FloatTensor([batch_size]).uniform_() < torch.exp((H_current - H), torch.cuda.FloatTensor))
        self.accept_rate = torch.mean(is_accept)

        is_accept_rs = is_accept.view([batch_size, 1])
        self.z_current = is_accept_rs * self.z + (1. - is_accept_rs) * self.z_current
        return self.z_current, self.accept_rate
        
        
    def euler_step(self,eps, beta):
        
        self.euler_step_z(eps)
        self.euler_step_p(eps,beta)

        #gradU = torch.reshape(torch.gradients(self.U, self.z), [batch_size, z_dim])
        #self.euler_p = self.p.assign_sub(eps_scaled * gradU)
    
    def euler_step_z(self, eps):
        eps_scaled = self.eps_scale * eps
        self.z = self.z + (eps_scaled * self.p/self.mass)
    
    def euler_step_p(self, eps, beta):
        self.U=self.get_energy(self.z, beta)
        self.U.backward()
        gradU = self.z.grad() 
        gradU = gradU.view([batch_size, z_dim])
        self.p = self.p - eps_scaled*gradU
    
    def get_energy(self, z, beta):
        E = beta*self.get_energy1(z) + (1 - beta) * self.get_energy0(z,self.params_posterior)
        return E    
        
    def get_energy1(self, z):
        N, T = z.size()
        decoder_out = self.decoder(z.view([self.batch_size,self.num_sam,-1]))
        #z = z.view([self.num_sam*N,-1])
        recon_x = decoder_out.view([N,-1])
        E = utils.get_reconstr_err(recon_x,self.x, self.args)
        # Prior
        E = E+ torch.sum(
            0.5 * z*z + 0.5 * np.log(2*np.pi), 1
        )

        return E

    def get_energy0(self, z, params_posterior):
        
        N, T = z.size()
        z_mean = params_posterior[0]
        log_z_std = params_posterior[1]
        E = -utils.get_pdf_gauss(z_mean, log_z_std, z)
        return E

    
    def read_batch(self, sess):
        sess.run(self.init_batch)

        
    def evaluate(self):
        is_adaptive_eps = self.config['test_is_adaptive_eps']
        nsteps = self.config['test_ais_nsteps']
        batch_size = self.config['batch_size']
        eps = self.config['test_ais_eps']

        logpx = 0.
        weights = np.zeros([100, batch_size])

        betas = np.linspace(0, 1, nsteps+1)
        accept_rate = 1.

        t = time.time()
        # Initializing hmc 
        #self.z_current = self.get_z0(self.params_posterior) 
        progress = tqdm(range(nsteps), desc="HMC")
        for i in progress:
            f0 = -self.get_energy(self.z_current, betas[i])  # -sess.run(self.U_current, feed_dict={self.beta: betas[i]})
            f1 = -self.get_energy(self.z_current, betas[i+1]) #-sess.run(self.U_current, feed_dict={self.beta: betas[i+1]})
            logpx =logpx+ f1 - f0

            if i < nsteps-1:
                accept_rate = self.run_hmc_step(betas[i+1], eps)
                if is_adaptive_eps and accept_rate < 0.6:
                    eps = eps / 1.1
                elif is_adaptive_eps and accept_rate > 0.7:
                    eps = eps * 1.1
                progress.set_postfix(
                    accept_rate="%.2f" % accept_rate,
                    logw="%.2f+-%.2f" % (logpx.mean(), logpx.std()),
                    eps="%.2e" % eps,
                )
        samples = self.z_current

        return logpx, samples

    def run_hmc_step(self, beta, eps):
        L = 10 # TODO: make configuratble
        
        # Initialize
        #sess.run(self.init_hmc_step)
        # initializing  hmc step 
        self.p_current =[self.p_rnd]
        
        # initializing hmc step2
        #sess.run(self.init_hmc_step2)
        self.z = self.z_current
        self.p = self.p_current

        # Leapfrog steps
        #sess.run(self.euler_p, feed_dict={self.eps: eps/2, self.beta: beta})
        self.euler_step(eps/2, beta)
        for i in range(L+1):
            self.euler_step_z(eps) #BETA may need to be passed in
            #sess.run(self.euler_z, feed_dict={self.eps: eps, self.beta: beta})
            if i < L:
                self.euler_step_p(eps, beta)
                #sess.run(self.euler_p, feed_dict={self.eps: eps, self.beta: beta})
        self.euler_step_p(eps/2, beta)
        #sess.run(self.euler_p, feed_dict={self.eps: eps/2, self.beta: beta})

        # Update Z
        _, accept_rate = self.update_z(beta)
        #_, accept_rate = sess.run([self.update_z, self.accept_rate], feed_dict={self.beta: beta})
        return accept_rate


    def average_weights(self, weights, axis=0):
        nchains = weights.shape[axis]
        logsumw = sp.misc.logsumexp(weights, axis=axis)
        lprob = logsumw - np.log(nchains)
        ess = np.exp(-sp.misc.logsumexp(2*(weights - logsumw), axis=axis))
        return lprob, ess
        
