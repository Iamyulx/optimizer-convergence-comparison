# Optimizer Convergence Comparison

Implementation and comparison of several deep learning optimizers, including **custom implementations from scratch** and PyTorch implementations.

## Optimizers Implemented

From scratch:

- SGD
- Adam
- AdamW

PyTorch:

- torch.optim.Adam
- torch.optim.AdamW

## Experiment

A simple linear regression model is trained on a synthetic dataset.  
The convergence behavior of each optimizer is compared using **MSE loss**.


## Observations

- SGD converges quickly but can be unstable depending on learning rate.
- Adam shows stable convergence due to adaptive moment estimation.
- AdamW introduces decoupled weight decay which improves generalization in large models.

## Project Structure

optimizers.py → optimizer implementations.
model.py → linear regression model.
train.py → training loop.
plot_results.py → optimizer comparison plot.


## Requirements


torch
matplotlib


## Educational Purpose

This repository demonstrates how popular optimizers work internally, providing a deeper understanding of modern deep learning training dynamics.
