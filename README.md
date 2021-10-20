# kfolden-ood-detection

The repository contains the code of the recent research advances at [Shannon.AI](http://www.shannonai.com). 

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

## Install Requirements

If you are working on a GPU machine with CUDA 10.1, please run the following command to setup environment. <br> 

```bash 
$ conda create -n kfolden-env python=3.6
$ conda activate kfolden-env
$ pip3 install -r requirements.txt 
$ pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
``` 

## Dataset 

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



### Contact 

If you have any issues or questions about this repo, please feel free to contact **xiaoya_li [AT] shannonai.com** .<br>
Any discussions, suggestions and questions are welcome !
