# FBI-Denoiser

The official code of FBI-Denoiser: Fast Blind Image Denoiser for Poisson-Gaussian Noise (Oral, CVPR 2021) [[arxiv]](https://arxiv.org/pdf/2105.10967.pdf).

## Quick Start

### 1. Requirements

```
$ pip install -r requirements.txt
$ mkdir weights result_data
```

### 2. Experiment with a synthetic noise using the FiveK dataset

1) Download FiveK[1] raw training and test dataset from [[this link]](https://drive.google.com/file/d/13RsccRrTi3zyNjQYdJ94E8WNbuJ8qG01/view?usp=sharing).
2) Locate 'Fivek_test_set.mat' and 'Fivek_training_set.hdf5' in './data' directory.

```
./data
      /Fivek_test_set.mat
      /Fivek_training_set.hdf5 
```

3) Generate the synthetic Poisson-Gaussian noise dataset 

: Run './data/generate_fivek_synthetic_dataset.ipynb' and choice parameters(\alpha and \sigma) for the Poisson-Gaussian noise.

4) Train PGE-Net and FBI-Net sequentially.

: Adjust \alpha and \sigma in 'train_FBI_PGE_Net_synthetic_noise.sh', and run it.


### 2. Load and evalute with pretrained weights of FBI-Net

1) Download FiveK[1], FMD[2], SIDD[3] and DND[4] test dataset from [[this link]](https://drive.google.com/file/d/139dSklT10_sCg4t5HL1z1cWY8x5seVBP/view?usp=sharing).

2) Locate downloaded datasets in './data' directory.

```
./data
      /test_CF_FISH.mat 
      /test_CF_MICE.mat
      /test_TP_MICE.mat
      /test_DND.mat
      /test_SIDD.mat
      /Fivek_test_set.mat
```

3) Download pretrained weights of FBI-Net from [[this link]](https://drive.google.com/file/d/134EdFVulZkmT7OyYk-a0oCWO0N8kQfpQ/view?usp=sharing).

4) Locate downloaded weights in './weights' directory.

5) (for synthetic noise datasets) Generate the synthetic Poisson-Gaussian noise test dataset 

: Run './data/generate_fivek_synthetic_dataset.ipynb' and choice parameters(\alpha and \sigma) for the Poisson-Gaussian noise.

6) Evaluate pretrained FBI-Net with test datasets

: Run './evaluate_pretrained_fbi_net.sh'.


#### Evaluation results
|                  |     FiveK synthetic Poisson-Gaussian noises     |                              |                              |               |
|:----------------:|:------------------------------:|:----------------------------:|:----------------------------:|:-------------:|
| Noise parameters | \alpha = 0.01, \sigma = 0.0002 | \alpha = 0.01, \sigma = 0.02 | \alpha = 0.05, \sigma = 0.02 | Mixture noise |
|      FBI-Net     | 44.44 / 0.9569                               |   43.08 / 0.9384                           | 39.00 / 0.9099                             | 46.91 / 0.9704              |

| Real Noise Datasets |       FMD      |       FMD       |       FMD      |      SIDD          |   DND  |
|:-------------------:|:--------------:|:---------------:|:--------------:|:--------------:|:---:|
|                     |     CF FISH    |     CF MICE     |     TP MICE    |      Validation Dataset      | Validation Dataset |
|       FBI-Net       | 32.29 / 0.8873 | 38.31 / 0.99637 | 33.93 / 0.9087 | 51.26 / 0.9273 |  39.65 / 0.8928   |

### 3. Load and evalute with pretrained weights of PGE-Net

1) Download FiveK[1], FMD[2], SIDD[3] and DND[4] test dataset from [[this link]](https://drive.google.com/file/d/139dSklT10_sCg4t5HL1z1cWY8x5seVBP/view?usp=sharing).

2) Locate downloaded datasets in './data' directory.

```
./data
      /test_CF_FISH.mat 
      /test_CF_MICE.mat
      /test_TP_MICE.mat
      /test_DND.mat
      /test_SIDD.mat
      /Fivek_test_set.mat
```
3) Download pretrained weights of PGE-Net from [[this link]](https://drive.google.com/file/d/134EdFVulZkmT7OyYk-a0oCWO0N8kQfpQ/view?usp=sharing).

4) Locate downloaded weights in './weights' directory.

5) (for synthetic noise datasets) Generate the synthetic Poisson-Gaussian noise test dataset 

: Run './data/generate_fivek_synthetic_dataset.ipynb' and choice parameters(\alpha and \sigma) for the Poisson-Gaussian noise.

6) Evaluate pretrained PGE-Net with test datasets

: Run './evaluate_pretrained_pge_net.sh'.

#### Evaluation results
|         |                  |              FiveK             |             FiveK            |             FiveK            |     FiveK     |   FMD   |   FMD   |   FMD   |     SIDD     |      DND     |
|:-------:|:----------------:|:------------------------------:|:----------------------------:|:----------------------------:|:-------------:|:-------:|:-------:|:-------:|:------------:|:------------:|
|         | Noise parameters | \alpha = 0.01, \sigma = 0.0002 | \alpha = 0.01, \sigma = 0.02 | \alpha = 0.05, \sigma = 0.02 | Mixture Noise | CF FISH | CF MICE | TP MICE | Validation | Validation |
| PGE-Net |    \alpha_hat    |      0.0101                          |           0.0142                   |       0.0398                       |    0.0028           |   0.0357      |   0.0120      |  0.023       |     0.0086         |    0.0.0016         |
| PGE-Net |    \sigma_hat    |      0.000006                          |         0.005267                     |        0.003603                      |   0.003851            |   0.000417      | 0.000697       |  0.00025       |     0.003954         |    0.0.006833          |

## QnA
### 1. Where is the code for generating the synthetic Poissian Gaussian noise?

: Check add_noise() and random_noise_levels() in './data/generate_fivek_synthetic_dataset.ipynb'

### 2. The estimation result of PGE-Net for \sigma is quite underestimated than the true value of \sigma.

: The detailed dicussion and experimental results are proposed in Section 5.2 of the paper. We showed that, even though \sigma is underestimated, it did not significantly affect the denoising performance of using GAT+BM3D and FBI-Net (see Table 4 and 5 of the paper).

### If you have any questions or problems to run this code, please mail to sungmin.cha@snu.ac.kr. Thank you!

## Citation

```
@inproceedings{byun2021fbi,
  title={FBI-Denoiser: Fast Blind Image Denoiser for Poisson-Gaussian Noise},
  author={Byun, Jaeseok and Cha, Sungmin and Moon, Taesup},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5768--5777},
  year={2021}
}
```

## Reference

[1] A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images [[arxiv]](https://arxiv.org/abs/1812.10366)

[2] The Berkeley Segmentation Dataset and Benchmark [[link]](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

[3] A High-Quality Denoising Dataset for Smartphone Cameras [[link]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Abdelhamed_A_High-Quality_Denoising_CVPR_2018_paper.pdf)

[4] Natural Image Noise Dataset [[link]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Brummer_Natural_Image_Noise_Dataset_CVPRW_2019_paper.pdf)

