# 多模态情感分析大作业
这是“当代人工智能”第五次大作业的代码
## Setup
### 代码使用AutoDL云服务器运行
- GPU - RTX 4090
- torch(1.11.0)、python(3.8)、cuda(11.3)：
### requirements：
- pandas==2.0.3
- numpy==1.22.4
- Pillow==9.1.1
- scikit-learn==1.3.0
- scipy==1.10.1
- tokenizers==0.13.3
- torch @ http://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
- torchvision @ http://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
- tornado==6.1
- tqdm==4.65.0
- transformers==4.30.2

You can simply run：
```shell
pip install -r requirments.txt
```

## Repository structure
由于不能上传超过25M的文件，所以训练的模型bert，bert_tokenizer, resnet-50， bertSelf.pt没法上传
```python
|-- 实验五数据 # 本次实验用到的数据
    ｜-- data # 文本与图像
|-- README.md # 本项目的介绍
|-- data_util.py # 数据处理的文件，里面包含了构建dataset类与dataloader的代码
|-- df_for_test.csv # 整理好的用于预测数据集
|-- df_for_train.csv # 整理好的用于训练的数据集
|-- main.py # 主函数
|-- model.py  # 构建多模态模型的代码
|-- prediction.py  # 用于预测的代码
|-- requirements.txt # 创建好云服务器后需要的依赖
|-- test_with_labels.txt # 本次实验的提交答案
|-- train.py # 用于训练的模型的代码
```

## Train and Predict
### parameters setting
```python
python ./main.py -h
```
```python
--model MODEL         #选择使用的模型
--lr LR               #设置学习率
--weight_decay WEIGHT_DECAY  #设置权重衰减
--warmup WARMUP       #预热学习率步数                      
--epochs EPOCHS       #设置训练轮数
--batch_size BATCH_SIZE  #批量大小
                        
```

### Train the model
#### Multimodal Fusion
模型分别是**Concatenation, Additive Attention, Multi-Layer Fusion, CL-Multi-Layer Fusion**,
Multi-Layer Fusion
```shell
python ./main.py --model concat
```
Concatenation
```shell
python ./main.py --model additive
```
Additive Attention
```shell
python ./main.py --model mlf
```
CL-Multi-Layer Fusion
```shell
python ./main.py --model cl
```
#### Ablation experiment
消融是对于Multi-Layer Fusion而言的
text—only
```shell
python ./main.py --model text_only
```
image—only
```shell
python ./main.py --model image_only
```
### Predict
```shell
python ./main.py --model test
```

## References

[1]  Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. CLMLF:a contrastive learning and multi-layer fusion method for multimodal sentiment detection. In Findings of the Association for Computational Linguistics: NAACL 2022. Association for Com- putational Linguistics, 2022.

[2] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

https://github.com/liyunfan1223/multimodal-sentiment-analysis

https://github.com/Link-Li/CLMLF








