"""
Followed  https://github.com/znxlwm/pytorch-generative-model-collections Style of Coding
"""

import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
#from torch.distributions.distribution import Distribution
import torch
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)



class DVAE(nn.Module):

    def __init__(self, args):
        super(DVAE, self).__init__()
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.model_type
        self.z_dim = args.z_dim
        self.arch_type = args.arch_type

        # networks init
        self.encoder_init()
        self.decoder_init()

        if self.gpu_mode:
            self.reconstruction_function = nn.BCELoss().cuda()
        else:
            self.reconstruction_function = nn.BCELoss()

        self.reconstruction_function.size_average = False

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)), volatile=True)

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = self.reconstruction_function(recon_x, x) / self.batch_size
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.mean(torch.sum(KLD_element, dim=1).mul_(-0.5), dim=0)
    
        return BCE + KLD


    def decoder_init(self):
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        self.input_height = 28
        self.input_width = 28
        self.output_dim = 1
    

        if self.arch_type == 'conv':
            self.dec_layer1 = nn.Sequential(
                nn.Linear(self.z_dim, 128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.ReLU(),
            )

            self.dec_layer2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:

            self.dec_layer1 = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim*2),
                nn.BatchNorm1d(self.z_dim*2),
                nn.ReLU(),
            )

            self.dec_layer2 = nn.Sequential(
                nn.Linear(self.z_dim*2, self.input_height * self.input_width),
                nn.Sigmoid(),
            )
        utils.initialize_weights(self)
   

    def encoder_init(self):
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1

        if self.arch_type == 'conv':
            self.enc_layer1 = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            self.mu_fc = nn.Sequential(
                nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), self.z_dim),
            )
    
            self.sigma_fc = nn.Sequential(
                nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), self.z_dim),
            )
        else:

            self.enc_layer1 = nn.Sequential(
                nn.Linear(self.input_height*self.input_width, self.z_dim*2),
                nn.BatchNorm1d(self.z_dim*2),
                nn.ReLU(),
                nn.Linear(self.z_dim*2, self.z_dim*2),
                nn.BatchNorm1d(self.z_dim*2),
                nn.ReLU(),
            )

            self.mu_fc = nn.Sequential(
                nn.Linear(self.z_dim*2, self.z_dim),
            )
    
            self.sigma_fc = nn.Sequential(
                nn.Linear(self.z_dim*2, self.z_dim),
            )


        utils.initialize_weights(self)


    def encode(self, x):

        if self.arch_type == 'conv':
            x = self.enc_layer1(x)
            x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        else:
            x = x.view([-1, self.input_height * self.input_width * self.input_dim])
            x = self.enc_layer1(x)
        mean  = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        
        return mean, sigma


    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.gpu_mode :
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def get_latent_var(self, x):

        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        return z


    def decode(self, z):
    
        x = self.dec_layer1(z)
        if self.arch_type == 'conv':
            x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
            x = self.dec_layer2(x)
        else:
            x = self.dec_layer2(x)
            x = x.view(-1, 1, self.input_height, self.input_width)
        return x

    
    def forward(self, x):

        if self.model_name == 'DVAE':
            if self.gpu_mode:
                eps = torch.cuda.FloatTensor(x.size()).normal_(std=0.05)
            else:
                eps = torch.FloatTensor(x.size()).normal_(std=0.05)
            eps = Variable(eps) # requires_grad=False
            x = x.add_(eps)
            #tmp = Distribution.Binomial(x, torch.Tensor(1-std))


        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar
    
    
    def measure_marginal_log_likelihood(model,dataset, subdataset='test', seed=42,minibatch_size=20,num_samples=50):
        print("Measuring {} log likelihood".format(subdataset))
        srng=utils.srng(seed)
        test_x= dataset 
        n_examples = test_x
            


