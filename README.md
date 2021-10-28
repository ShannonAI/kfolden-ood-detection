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
This repository contains [code](./data/preprocess) and [scripts](./scripts/data_preprocess) for generating our benchmarks from their original datafiles.   <br>
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
And the `<benchmark-name>/test/ood_test.csv` file contains out-of-distribution test data instances. <br>
More details can be found in the [paper](https://arxiv.org/pdf/2108.12731.pdf) (Section 5 and Appendix). 


## Requirements

If you are working on a GPU machine with CUDA 10.1, please run the following command to setup environment. <br> 

```bash 
$ conda create -n kfolden-env python=3.6
$ conda activate kfolden-env
$ pip3 install -r requirements.txt 
$ pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
``` 

**Notice**: please check your CUDA version and install compatible pytorch referring to [pytorch.org](https://pytorch.org/).  



## 1. Download Glove, BERT, and RoBERTa 

Before start training models on benchmark datasets, please first download required files (e.g., Glove, BERT, RoBERTa).  

1.1 For CNN/LSTM, run [bash ./scripts/download/word_embedding.sh](./scripts/download/word_embedding.sh) to obtain Glove-300D the vocab file and the weight file. <br>

1.2 For Pretrained Masked LM, run [bash ./scripts/download/pretrained_lm.sh](./scripts/download/pretrained_lm.sh) 
to obtain BERT and RoBERTa model files.

## 2. Train and Evaluate 

2.1 For CNN/LSTM models, scripts for reproducing experimental results can be found under the `./scripts/<dataset_name>/` folder. <br>
Training and evaluation procedure are defined in [./task/train_nn.py](./task/train_nn.py).  <br>

2.2 For pretrained MLM models, scripts for reproducing experimental results can be found under the `./scripts/<dataset_name>/` folder. <br>
Training and evaluation procedure are defined in [./task/finetune_plm.py](./task/finetune_plm.py). <br>

Note that you need to change `DATA_DIR`, `BERT_DIR`, `OUTPUT_DIR` to your own dataset path, bert model path and log path, respectively.  <br> 

During training, the model trainer will automatically evaluate on the dev set every `val_check_interval` epochs,
and save the topk checkpoints to `$OUTPUT_DIR`. <br> 
After training, you can find the best checkpoint on the dev set according to the evaluation results in `$OUTPUT_DIR/train_log.txt`. <br> 
When treating $K$ labels as out-of-distribution samples, the model trainer saves $k$ under the directory `OUTPUT_DIR`. 

**Note**: <br>
[1]: `REPO_PATH`: The path to the `kfolden-ood-detection` repository. <br> 
[2]: `DATA_DIR`: The path to benchmark directory. <br> 
[3]: `OUTPUT_DIR`: The directory for saving training logs and intermediate checkpoints. <br>


### Contact 

If you have any issues or questions about this repo, please feel free to contact **xiaoya_li [AT] shannonai.com** .<br>
Any discussions, suggestions and questions are welcome !
