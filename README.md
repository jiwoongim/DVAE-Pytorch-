# Denoising Criterion for Variational Auto-encoding Framework (Pytorch Version of DVAE)
Python (Theano) implementation of Denoising Criterion for Variational Auto-encoding Framework code provided 
by Daniel Jiwoong Im, Sungjin Ahn, Roland Memisevic, and Yoshua Bengio.
Denoising criterion injects noise in input and attempts to 
generate the original data. This is shown to be advantageous.
The codes include training criterion which corresponds to a 
tractable bound when input is corrupted. For more information, see 

```bibtex
@article{Im2016dvae,
    title={Denoising Criterion for Variational Framework},
    author={Im, Daniel Jiwoong and Ahn,Sungjin and Memisevic, Roland and Bengio, Yoshua},
    journal={http://arxiv.org/abs/1511.06406},
    year={2016}
}
```

If you use this in your research, we kindly ask that you cite the above arxiv paper


## Dependencies
Packages
* [numpy](http://www.numpy.org/)
* [Pytorch](http://pytorch.org/)


## How to run
Entry code for one-bit flip and factored minimum probability flow for mnist data are 
```
    - /main_vae.py
```


