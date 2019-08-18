# SRQA

SRQA: Synthetic Reader for Factoid Question Answering

[中文说明](#chinesehead)
## Introduction
This is a QA system designed for machine reading comprehension. And we have performed it on WebQA dataset. You can train, test, and evaluate this system following this instruction.
## Requirements
- python3.6
- pytorch >= 1.0
- sru[cuda]
- pandas
- jieba

Note：
- If you want to run *server.py*, the other two python package *flask* and *requests* is needed.
- The model need to use *SRU*, so you need to run *"pip install sru[cuda]"* to install python package cupy, pynvrtc, etc.

<h2 id="data">Files Detail</h2> 

**code_WebQA**: contain most code for training and testing our system on WebQA dataset. And a flask server is also included.
  
**data**: contain dataset and other necessary data. The directory format is like:
```
./
├── chars.txt  *Chinese character dictionary*
├── refined_lbs.pkl  *the file used when refining labels*
├── test.ann.json.gz  *file from original WebQA dataset*
├── test.ir.json.gz  *file from original WebQA dataset*
├── training.json.gz  *file from original WebQA dataset*
├── validation.ann.json.gz  *file from original WebQA dataset*
└── validation.ir.json.gz  *file from original WebQA dataset*
```

**evaluation**: contain codes related to the evaluation on WebQA dataset.

**model**: to save checkpoints.

**logs**: to save training logs.
## Prepare Data
- WebQA dataset: This is a Factoid Question Answering dataset described in the paper [Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering](https://arxiv.org/abs/1607.06275). And we can download this dataset from [google drive](https://drive.google.com/file/d/1P6Asn8LPECGXCuIjfWci_NYJsdNYP1m7/view?usp=sharing). 下After downloading, extract the relevant data in the compressed package to the data folder of the project according to the directory format in [Files Detail] (#data).
- [Other related data:](https://drive.google.com/open?id=17OAzI9Z7sB8SKrawWa0Ldem862cQTEI4)
  - chars.txt, refined_lbs.pkl: Download to the *data* folder of this project.
  - model_WebQA.pt: Download to the *data* folder of this project (this is the pre-trained model loaded by default, which can be ignored if using the local trained model).

## Usage
### Change the commend direction to ./code_WebQA/
    cd ./code_WebQA/
### Run test for single sample
    python test_sample.py
The program will load model_WebQA.pt in the folder *model*.

A single-sentence test can be implemented by entering a question (up to a maximum of 20 words) and a passage (up to a maximum of 100 words) on the command line as instructed.
### Train a new model
    python main.py
The training parameters of the model, such as train_id, debug, etc., can be changed in *parameters.py*.

The model files are saved in the *model* folder and creates a new folder corresponding to the train_id in *parameters.py* to save the model.

The trained log will be saved in the *logs* folder and will also be output on the command line.
### 运行flask服务程序

Run *server.py* in a new command window:
```
python server.py
```
Run *clinet.py* in another command window:
```
python clinet.py
```
So that you can observe the operation of the *server* and the *clinet*.

## Reference

If you find this repo useful, please consider citing:
```
@inproceedings{wang2018a3net,
  title={A3Net: Adversarial-and-Attention Network for Machine Reading Comprehension},
  author={Wang, Jiuniu and Fu, Xingyu and Xu, Guangluan and Wu, Yirong and Chen, Ziyan and Wei, Yang and Jin, Li},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={64--75},
  year={2018},
  organization={Springer}
}
```


<h1 id="chinesehead">SRQA 中文机器阅读理解项目</h1> 

SRQA: Synthetic Reader for Factoid Question Answering

事实类问题的综合阅读理解
## 介绍
本系统针对WebQA数据集实现了中文机器阅读理解的功能，包括在数据集上训练、测试，以及利用训练好的模型进行单句测试。
## 运行环境
- python3.6
- pytorch >= 1.0
- sru[cuda]
- pandas
- jieba

注意：
- 如果运行服务server.py，还需要flask和requests。
- 模型使用SRU，因此需要cupy、pynvrtc等依赖，使用"pip install sru[cuda]"指令可以安装全部支持。

<h2 id="data_cn">文件结构</h2> 

code_WebQA: 包含了处理WebQA的代码，支持训练、句测试以及flask服务。
  
data: 用于存放WebQA数据集，数据目录如下：
```
./
├── chars.txt  中文字符字典
├── refined_lbs.pkl  对label进行优化的本地文件
├── test.ann.json.gz  WebQA数据集提供的数据
├── test.ir.json.gz  WebQA数据集提供的数据
├── training.json.gz  WebQA数据集提供的数据
├── validation.ann.json.gz  WebQA数据集提供的数据
└── validation.ir.json.gz  WebQA数据集提供的数据
```

evaluation: 与WebQA测试相关的配套代码

model: 保存训练好的模型
## 下载数据
- WebQA数据集：在论文 [Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering](https://arxiv.org/abs/1607.06275)中讨论的数据集。 可以从 [google drive](https://drive.google.com/file/d/1P6Asn8LPECGXCuIjfWci_NYJsdNYP1m7/view?usp=sharing)中下载。下载后将压缩包中的data文件夹中的相关数据按照[文件结构](#data_cn)中的目录形式解压到本项目的data文件夹下。
- [其他相关数据：](https://drive.google.com/open?id=17OAzI9Z7sB8SKrawWa0Ldem862cQTEI4)
  - chars.txt, refined_lbs.pkl: 下载到本项目的data文件夹下。
  - model_WebQA.pt: 下载到本项目的data文件夹下（这是项目默认载入的预训练模型，若使用本地训练模型则可以忽略）。


## 使用方法
### 将系统目录调整到 ./code_WebQA/
    cd ./code_WebQA/
### 运行单句测试程序
    python test_sample.py
程序会载入文件夹model中的model_WebQA.pt。

按照指示在命令行中输入问题（处理上限为20个单词）和文档（处理上限为100个单词）,即可实现单句测试。
### 训练新模型
    python main.py
通过改变parameters.py中的参数可以控制模型的训练条件，如train_id，debug等参数。

训练时会将模型保存在model文件夹下，并且新建与parameters.py中的train_id相对应的文件夹来保存模型。

训练的log会保存在code/logs文件夹下，同时也会在命令行输出。
### 运行flask服务程序

在新的窗口中运行server.py:
```
python server.py
```
在另一个窗口中运行clinet.py:
```
python clinet.py
```
即可观察到server以及clinet的运行情况。

## 参考文献

如果您觉得本项目对您有帮助，可以引用如下文献：
```
@inproceedings{wang2018a3net,
  title={A3Net: Adversarial-and-Attention Network for Machine Reading Comprehension},
  author={Wang, Jiuniu and Fu, Xingyu and Xu, Guangluan and Wu, Yirong and Chen, Ziyan and Wei, Yang and Jin, Li},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={64--75},
  year={2018},
  organization={Springer}
}
```


