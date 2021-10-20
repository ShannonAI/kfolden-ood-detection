# kfolden-ood-detection

The repository contains the code for the recent research advances at [Shannon.AI](http://www.shannonai.com). 

**kFolden: k-Fold Ensemble for Out-Of-Distribution Detection** <br>
Xiaoya Li, Jiwei Li, Xiaofei Sun, Chun Fan, Tianwei Zhang, Fei Wu, Yuxian Meng and Jun Zhang<br>
EMNLP 2021, [paper](https://arxiv.org/pdf/2108.12731)<br>
If you find this repository helpful, please cite the following:
```tex 
 @article{li2021k,
  title={$ k $ Folden: $ k $-Fold Ensemble for Out-Of-Distribution Detection},
  author={Li, Xiaoya and Li, Jiwei and Sun, Xiaofei and Fan, Chun and Zhang, Tianwei and Wu, Fei and Meng, Yuxian and Zhang, Jun},
  journal={arXiv preprint arXiv:2108.12731},
  year={2021}
}
```

## Benchmarks 

In this paper, we construct semantic shift and non-semantic shift benchmarks for out-of-distribution detection. <br>
You can download the benchmarks following this [guidline](./data/README.md). 
And this repository also contains [code](./data/preprocess) and [scripts](./scripts/data_preprocess) for generating our benchmarks from their original datafiles.   <br>
The unzipped dataset directory should have the following structure: <br>

```text
<benchmark-name>
├── dev
│   ├── id_dev.csv
│   └── ood_dev.csv
├── test
│   ├── id_test.csv
│   └── ood_test.csv
└── train
    └── train.csv
```

Every dataset directory contains three subdirectories `train/`, `dev/`, and `test/`, each containing the randomly sampled training, development, and testing subsets, respectively. <br>
For example, the testing set for in-distribution can be found in the `<benchmark-name>/test/id_test.csv` file. 
And the `<benchmark-name>/test/ood_test.csv` file contains out-of-distribution test data samples. <br>
More details can be found in the [paper](https://arxiv.org/pdf/2108.12731.pdf) (Section 5 and Appendix). 


## Requirements

If you are working on a GPU machine with CUDA 10.1, please run the following command to setup environment. <br> 

```bash 
$ conda create -n kfolden-env python=3.6
$ conda activate kfolden-env
$ pip3 install -r requirements.txt 
$ pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
``` 

**Notice**: please check your CUDA version and install compatible pytorch. Please refer to [https://pytorch.org/](https://pytorch.org/) for more details.  


## Training 


## Evaluation 



### Contact 

If you have any issues or questions about this repo, please feel free to contact **xiaoya_li [AT] shannonai.com** .<br>
Any discussions, suggestions and questions are welcome !
