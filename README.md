# MELD

This repository contains a python package that implements _MELD_, 
a moment estimation method for generalized Dirichlet latent variable model.
For the details of the method and the model, please see http://arxiv.org/abs/1603.05324.

# Introduction

MELD stands for *M*oment *E*stimation for generalized *L*atent *D*irichlet variable models.


## The model

The generalized latent Dirichlet variable model in MELD assumes the *p* dimensional observation 
**_y_**<sub>_i_</sub> is a mixture of *k* latent components, with mixture weight denoted as
a Dirichlet distributed latent variable **_x_**<sub>_i_</sub>. This model is also known as
a *mixed membership model*. In contrast to previous mixed membership models, the new model
allows each coordinate of **_y_**<sub>_i_</sub> to take different variable types, including
categorical, continuous and integer-valued variables.

## Parameter estimation

The parameter estimation method developed in MELD uses a moment method known as generalized 
method of moments (GMM). By encoding each coordinate of **_y_**<sub>_i_</sub> by a dummy variable, 
MELD calculates the *cross* moment matrices or tensors among variables. Parameter estimation
is conducted using a coordinate descent algorithm.

# About the folders

- `data` contains the simulated categorical data with *n = {50,100,200,500,1000}* observations. 
The number of categorical variables is *20*. Each categorical variable could take *4* different 
categories. 
- `code` contains the implementation of MELD and a python script showing how to use MELD.




