"""
Followed  https://github.com/znxlwm/pytorch-generative-model-collections Style of Coding
"""

import utils, torch, time, os, pickle
from ais import AIS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from collections import defaultdict
#from torch.distributions.distribution import Distribution

from utils import log_likelihood_samples_mean_sigma, prior_z, log_mean_exp

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
        self.num_sam = args.num_sam
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
            self.sample_z_ = Variable(torch.randn((self.batch_size, 1, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.randn((self.batch_size, 1, self.z_dim)), volatile=True)


    def log_likelihood_estimate(self, recon_x, x, Z, mu, logsig):

        N, C, iw, ih = x.shape
        x_tile = x.repeat(self.num_sam,1,1,1,1).permute(1,0,2,3,4)

        bce = x_tile * torch.log(recon_x) + (1. - x_tile) * torch.log(1 - recon_x)
        log_p_x_z   =  torch.sum(torch.sum(torch.sum(bce, dim=4), dim=3), dim=2)

        log_q_z_x = log_likelihood_samples_mean_sigma(Z, mu, logsig, dim=2)
        log_p_z   = prior_z(Z, dim=2)
        log_ws              = log_p_x_z - log_q_z_x + log_p_z
        #log_ws_minus_max    = log_ws - torch.max(log_ws, dim=1, keepdim=True)[0]
        #ws                  = torch.exp(log_ws_minus_max)
        #normalized_ws       = ws / torch.sum(ws, dim=1, keepdim=True)
        return -torch.mean(torch.squeeze(log_mean_exp(log_ws, dim=1)), dim=0)


    def elbo(self, recon_x, x, mu, logsig):

        N, M, C, iw, ih = recon_x.shape
        x = x.contiguous().view([N*M,C,iw,ih])
        recon_x = recon_x.view([N*M,C,iw,ih])
        BCE = self.reconstruction_function(recon_x, x) / (N*M)
        KLD_element = (logsig - mu**2 - torch.exp(logsig) + 1 )
        #KLD_element = mu.pow(2).add_(logsig.mul_(2).exp()).mul_(-1).add_(1).add_(logsig.mul_(2))
        #KLD = torch.mean(torch.sum(KLD_element, dim=2).mul_(-0.5))
        #KLD_element = (logsig * 2) - (torch.exp(logsig *2)) - mu**2  + 1 
        KLD = - torch.mean(torch.sum(KLD_element* 0.5, dim=2) )

        return BCE + KLD


    def loss_function(self, recon_x, x, mu, logsig):

        N, C, iw, ih = x.shape
        x_tile = x.repeat(self.num_sam,1,1,1,1).permute(1,0,2,3,4)
        #J = - self.log_likelihood_estimate(recon_x, x_tile, Z, mu, logsig)
        J_low = self.elbo(recon_x, x_tile, mu, logsig)
        return J_low



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
                nn.Linear(self.z_dim, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.LeakyReLU(0.2),
                nn.Linear(self.z_dim*4, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                #nn.LeakyReLU(0.2),
                nn.Tanh(),
            )

            self.dec_layer2 = nn.Sequential(
                nn.Linear(self.z_dim*4, self.input_height * self.input_width),
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
                nn.Linear(self.input_height*self.input_width, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.LeakyReLU(0.2),
                nn.Linear(self.z_dim*4, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.LeakyReLU(0.2),
            )

            self.mu_fc = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim),
            )
    
            self.sigma_fc = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim),
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


    def sample(self, mu, logsig):
        #std = logsig.mul(0.5).exp_()
        std = torch.exp(logsig*0.5)
        if self.gpu_mode :
            eps = torch.randn(std.size()).cuda()
        else:
            eps = torch.randn(std.size())
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def get_latent_sample(self, x):

        mu, logsig = self.encode(x)
        z = self.sample(mu, logsig)
        return z


    def decode(self, z):
  
        N,T,D = z.size()
        x = self.dec_layer1(z.view([-1,D]))

        if self.arch_type == 'conv':
            x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
            x = self.dec_layer2(x)
        else:
            x = self.dec_layer2(x)
            x = x.view(-1, 1, self.input_height, self.input_width)
        return x.view([N,T,-1,self.input_width, self.input_height])

    
    def forward(self, x, testF=False):

        if self.model_name == 'DVAE' and not testF:
            if self.gpu_mode:
                eps = torch.randn(x.size()).cuda() * 0.025
            else:
                eps = torch.randn(x.size()) * 0.025
            eps = Variable(eps) # requires_grad=False
            x = x.add_(eps)
            #tmp = Distribution.Binomial(x, torch.Tensor(1-std))

        mu, logsig = self.encode(x)
        mu  = mu.repeat(self.num_sam,1,1).permute(1,0,2)
        logsig = logsig.repeat(self.num_sam,1,1).permute(1,0,2)

        z = self.sample(mu, logsig)
        res = self.decode(z)
        return res, mu, logsig, z
   
    
    def get_z0(self, batch_size,z_dim,z_mean, z_std):
        z_std = torch.exp(z_std)
        return z_mean + z_std * torch.randn([batch_size, z_dim])
    
     
    
    def testing_ais(self,recon_batch,x_,mu,logvar,args):

        
        decoder = self.decode   
        log_dir = args.log_dir
        
        logstd = 0.5*logvar
        logstd = logstd.view([self.num_sam*self.batch_size,-1])
        mu = mu.view([self.num_sam*self.batch_size,-1])
        
        z = self.get_z0(args.batch_size,args.z_dim,mu,logvar)
        z = z.view([self.num_sam*self.batch_size,-1])
        
        eval_dir = args.eval_dir
        z_dim = args.z_dim
        stats = defaultdict(list)

        results_dir = os.path.join(eval_dir, "results")

        batch_size = args.batch_size
        ais_nchains = args.test_ais_nchains
        test_nais = args.test_nais
        
        x_tile = self.x.repeat(self.num_sam,1,1,1,1).permute(1,0,2,3,4).contiguous()
        x = x_tile.view([self.num_sam*N,-1])
        
        params_posterior = [mu, logstd]
        ais=AIS(x,params_posterior,decoder,z,args)
        
        
        progress_ais = tqdm(range(ais_nchains), desc="AIS")
        for j in progress_ais:
            ais_res[j], ais_samples[j] = ais.evaluate()
            ais_lprob, ais_ess = ais.average_weights(ais_res[:j+1], axis=0)
            progress_ais.set_postfix(
                lprob="%.2f+-%.2f" % (ais_lprob.mean(), ais_lprob.std()),
                ess="%.2f/%d" % (ais_ess.mean(), j+1)
            )
        stats["ais_mean"].append(np.mean(ais_lprob))
        stats["ais_std"].append(np.std(ais_lprob))
        stats["ais_ess_mean"].append(np.mean(ais_ess))
        stats["ais_ess_std"].append(np.std(ais_ess))
        return np.mean(ais_ess),np.std(ais_ess)
        
        
        
        
        
        

