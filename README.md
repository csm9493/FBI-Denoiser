# FBI-Denoiser: Fast Blind Image Denoising for Source-Dependent Noise

------

This is an implementation code for FBI-Denoiser.

It contains training codes for PGE and FBI Net and other baselines, such as, D-BSN and N2V.

For reproducing D-BSN and N2V, we downloaded the official code of it and made a change to apply it to our dataset.

We cannot upload datasets for training a model because the size of dataset is too big to upload.

### The final version of our code will be updated soon!

------

## 1. Train PGE-Net

Check 'train_est.sh'. It contains five different shall scripts to train PGE-Net on each dataset

## 2. Train FBI-Net (Unsupervised)

Check 'train_fbi_emse.sh'. It contains five shall scripts to train PGE-Net on each dataset.

## 3. Train FBI-Net (Supervised)

Check 'train_fbi_mse.sh'. It contains five shall scripts to train PGE-Net on each dataset and seven shall scripts for ablation study.

