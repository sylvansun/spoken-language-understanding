### 创建环境

    conda create -n slu python=3.6
    source activate slu
    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install transformers
### 模型训练和测试
  我们一共提交了两个模型，其中随canvas系统提交的模型model.bin是作业报告中Substitute+Denoise+RNN+CRF对应的模型，随jbox链接提交的模型model_combined_1.bin是作业报告中BertTRNN+CRF对应的模型（使用了Bert作为预训练模型，体积较大，需要从 [https://jbox.sjtu.edu.cn/l/V1tOYk](https://jbox.sjtu.edu.cn/l/V1tOYk) 下载模型并放置 `./` 目录下）。
  
  
  model.bin的测试命令为：
  ```shell
  python scripts/slu_minimodel_best.py --device 1 --crf --testing 
  ```
  model_combined_1.bin的测试命令为：
  ```shell
  python scripts/slu_combined.py --device 1  --crf --model_path model_combined_1.bin --testing
  ```
  运行此测试命令时，会自动复现开发集上的评分结果，并会同时基于助教提供的predict函数在根目录下生成对应的prediction.json文件
    
  欲训练和复现我们提供的model，可以分别在命令行执行如下代码。
  model.bin的训练命令为：
  ```shell
  python scripts/slu_minimodel_best.py --device 1 --crf 
  ```
  model_combined_1.bin的训练命令为：
  ```shell
  python scripts/slu_combinded.py --device 1 --lr 1e-4 --crf --model_path model_combined_1.bin
  ```  
  除此之外，我们还保留并提供了实验过程中使用到的其他模型和对应的脚本，均可以在scripts文件夹下找到。

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/examples/*.py`:读取数据代码文件夹，不同的训练脚本需要不同的读取方式
+ `utils/batches/*.py`:将数据以批为单位转化为输入代码文件夹，不同的训练脚本需要不同的转化方式
+ `model/slu_baseline_tagging.py`:baseline模型
+ `scripts/slu_baseline.py`:主程序脚本

### 有关预训练语言模型

本次代码中没有加入有关预训练语言模型的代码，如需使用预训练语言模型我们推荐使用下面几个预训练模型，若使用预训练语言模型，不要使用large级别的模型
+ Bert: https://huggingface.co/bert-base-chinese
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext
+ Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext
+ MacBert: https://huggingface.co/hfl/chinese-macbert-base

### 推荐使用的工具库
+ transformers
  + 使用预训练语言模型的工具库: https://huggingface.co/
+ nltk
  + 强力的NLP工具库: https://www.nltk.org/
+ stanza
  + 强力的NLP工具库: https://stanfordnlp.github.io/stanza/
+ jieba
  + 中文分词工具: https://github.com/fxsjy/jieba
