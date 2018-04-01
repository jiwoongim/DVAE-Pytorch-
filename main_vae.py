import argparse, os, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import utils 
from DVAE import DVAE


def train(model, args, data_loader_tr, data_loader_vl):

    if args.gpu_mode:
        model.cuda()

    print('---------- Networks architecture -------------')
    utils.print_network(model)
    print('-----------------------------------------------')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))



    train_hist = {}
    train_hist['tr_loss'] = []
    train_hist['vl_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    model.train()
    print('training start!!')
    start_time = time.time()
    for epoch in range(args.epoch):

        epoch_start_time = time.time()
        for iter, (x_, y_) in enumerate(data_loader_tr):
            if iter * args.batch_size < 50000:
                if iter == data_loader_tr.dataset.__len__() // args.batch_size:
                    break

                if args.gpu_mode:
                    x_ = Variable(x_.cuda())
                else:
                    x_ = Variable(x_)

                # Update DVAE network
                optimizer.zero_grad()

                recon_batch, mu, logvar, Z = model(x_)
                loss = model.loss_function(recon_batch, x_, mu, logvar)
                train_hist['tr_loss'].append(loss.data[0])
        
                loss.backward()
                optimizer.step()

                #if ((iter + 1) % 100) == 0:
                #    print("Epoch: [%2d] [%4d/%4d] loss (elbo): %.8f"%
                #            ((epoch + 1), \
                #            (iter + 1), \
                #            len(data_loader_tr.dataset) // args.batch_size, \
                #            loss.data[0]))

        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        visualize_results(model, epoch+1, args)


        for iter, (x_, y_) in enumerate(data_loader_vl):
            if iter * args.batch_size <= 10000:
                if iter == data_loader_vl.dataset.__len__() // args.batch_size:
                    break

                if args.gpu_mode:
                    x_ = Variable(x_.cuda())
                else:
                    x_ = Variable(x_)

                recon_batch, mu, logvar, Z = model(x_)
                elbo = model.loss_function(recon_batch, x_, mu, logvar)
                lle  = model.log_likelihood_estimate(recon_batch, x_, Z, mu, logvar)
                train_hist['vl_loss'].append(lle.data[0])


                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] Train loss: %.8f Valid  lle %.8f  Elbo (loss) %.8f" %
                            ((epoch + 1), \
                            (iter + 1), \
                            len(data_loader_vl.dataset) // args.batch_size, \
                            train_hist['tr_loss'][-1],\
                            lle.data[0],\
                            elbo.data[0]))

 

        if epoch % 25 :
            save(model, epoch, args.save_dir, args.dataset, \
                    args.model_type, args.batch_size, train_hist)

    train_hist['total_time'].append(time.time() - start_time)
    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % \
                    (np.mean(train_hist['per_epoch_time']),
                    epoch, train_hist['total_time'][0]))
    print("Training finish!... save training results")

    save(model, epoch, args.save_dir, args.dataset, args.model_type, \
                                    args.batch_size, train_hist)
    utils.generate_animation(args.result_dir + '/' + args.dataset + '/' \
                    + args.model_type + '/' + args.model_type, epoch)
    utils.loss_plot(train_hist, os.path.join(args.save_dir, args.dataset, \
                                    args.model_type), args.model_type)


def visualize_results(model, epoch, args, sample_num=100, fix=True):
    model.eval()

    if not os.path.exists(args.result_dir + '/' + args.dataset + '/' + args.model_type):
        os.makedirs(args.result_dir + '/' + args.dataset + '/' + args.model_type)

    tot_num_samples = min(sample_num, args.batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    if fix:
        """ fixed noise """
        samples = model.decode(model.sample_z_)
    else:
        """ random noise """
        if args.gpu_mode:
            sample_z_ = Variable(torch.rand((args.batch_size, args.z_dim)).cuda(), volatile=True)
        else:
            sample_z_ = Variable(torch.rand((args.batch_size, args.z_dim)), volatile=True)

        samples = model.sample(sample_z_)

    N,T,C,IW,IH = samples.size()
    samples = samples.view([N,C,IW,IH])
    if args.gpu_mode:
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
    else:
        samples = samples.data.numpy().transpose(0, 2, 3, 1)

    utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      args.result_dir + '/' + args.dataset + '/' + args.model_type + '/' + args.model_type + '_epoch%03d' % epoch + '.png')


def save(model, epoch, save_dir, dataset, model_type, batch_size, train_hist):
    save_dir = os.path.join(save_dir, dataset, model_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, model_type + '_encoder_epoch' + str(epoch)+'_batch_sz' + str(batch_size)+'.pkl'))

    with open(os.path.join(save_dir, model_type + '_history.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)


def load(model, save_dir, dataset='MNIST', model_type='VAE'):
    save_dir = os.path.join(save_dir, dataset, model_type)
    model.load_state_dict(torch.load(os.path.join(save_dir, model_type + '.pkl')))




"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of DVAE collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_type', type=str, default='VAE',
                        choices=['VAE', 'DVAE'],
                        help='The type of VAE')#, required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--arch_type', type=str, default='fc',\
                        help="'conv' | 'fc'")
    parser.add_argument('--z_dim', type=float, default=128)
    parser.add_argument('--num_sam', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()


    # declare instance for VAE
    if args.model_type == 'VAE' or args.model_type == 'DVAE':
        model = DVAE(args)
    else:
        raise Exception("[!] There is no option for " + args.model_type)


    # load dataset
    if args.dataset == 'mnist':
        data_loader_tr = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
                                                            transform=transforms.Compose(
                                                                [transforms.ToTensor()])),
                                                batch_size=args.batch_size, shuffle=False)

        data_loader_vl = DataLoader(datasets.MNIST('data/mnist', train=False, download=True,
                                                            transform=transforms.Compose(
                                                                [transforms.ToTensor()])),
                                                batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'fmnist':
        data_loader_tr = DataLoader(datasets.FashionMNIST('data/fashion-mnist', train=True, download=True,
                                                            transform=transforms.Compose(
                                                                [transforms.ToTensor()])),
                                                batch_size=args.batch_size, shuffle=False)

        data_loader_vl = DataLoader(datasets.FashionMNIST('data/fashion-mnist', train=False, download=True,
                                                            transform=transforms.Compose(
                                                                [transforms.ToTensor()])),
                                                batch_size=args.batch_size, shuffle=False)



    # launch the graph in a session
    train(model, args, data_loader_tr, data_loader_vl)
    print(" [*] Training finished!")

    # visualize learned generator
    #visualize_results(args.epoch)
    print(" [*] Testing finished!")




if __name__ == '__main__':
    main()

