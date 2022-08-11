## Extended Fourier Neural Operator

### Requirements
- PyTorch >=1.8.0. Pytorch 1.8.0 starts to support complex numbers and it has a new implementation of FFT
- torchinfo
- yaml
- numpy
- scipy
- ray

### files

- run: containing excutable python files
    - `efno_field_2d.py` is the EFNO for predicting electirc and magnetic field
    - `efno_2d.py` is the EFNO for predicting apparent resistivity and phase
    - `efno_2d_invariant.py` is the EFNO for super-resolution
    - `fourier_2d.py` is the Fourier Neural Operator([Li et al., 2021](https://arxiv.org/abs/2010.08895))
    - ``ECNN_2d.py` is the extended dense convolutional  encoder-decoder network (modified from [Zhu et. al, 2019](https://github.com/cics-nd/pde-surrogate.git)) 
    - `cofigure.yml` is the configuration for `efno_2d.py`, `efno_2d_invariant.py` and `fourier_2d.py`
    - `cofigure.yml` is the configuration for `efno__field2d.py`
    - `cofigure_CNN.yml` is the configuration for `ECNN_2d.py`
- scripts: some auxiliary python fiels
- model: trained model file
- Log: log file
- temp: if stop early, you can file model file here.

### Data

shared in google drive: https://drive.google.com/drive/folders/12nnzinkdz84tAYOOEqsJTzKOgy1MUqpo?usp=sharing

or you can generate training and testing samples by using python files in  `data_gen`.

- `gaussian_random_fields.py`:using gaussian random filed with different length scale. 

- `MT2D_secondary.py` is compute 2D MT response by using secondary filed method (SFM), Parallel version

- `model_random.py`generate conductivity structures and compute the apparent resistivity and phase by using finite difference method.

### Usage
```shell
cd ./run
python efno_2d.py random_best
```
