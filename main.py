import pandas as pd
import numpy as np
from train import train
from prediction import predict
from model import BertResNetModel, BertREsnet_additive, BertSelfAttention, CL,BertSelfAttention_pre
import argparse
from transformers import BertConfig
import torch

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='选择使用的模型')
    parser.add_argument('--image_only', help='MLF仅使用图片')
    parser.add_argument('--do_test', help='使用训练后的模型对测试集进行预测')
    parser.add_argument('--concatenation', help='直接拼接法')
    parser.add_argument('--additive_attention', help='Additive Attention模型')
    parser.add_argument('--mlf', help='Multi-Layer Fusion')
    parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
    parser.add_argument('--weight_decay', default=1e-2, help='设置权重衰减', type=float)
    parser.add_argument('--epochs', default=8, help='设置训练轮数', type=int)
    parser.add_argument('--batch_size', default=32, help='批量大小', type=int)
    parser.add_argument('--warmup', default=20, help='预热学习率步数',type=int)
    parser.add_argument('--seed', default=233, help='设置随机种子', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    df_for_train = pd.read_csv('./df_for_train.csv')
    df_for_test = pd.read_csv('./df_for_test.csv')
    config = BertConfig('./bert')

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)
    """在种子不变的情况下保证结果一致"""
    torch.backends.cudnn.deterministic = True

    args = set_args()
    if args.model == 'mlf':
        model = BertSelfAttention(config=config)
        train(model=model , 
              df_for_train=df_for_train, 
              augment=False, 
              num_train_epochs=args.epochs, 
              batch_size=args.batch_size,
              warmup_steps=args.warmup,
              weight_decay=args.weight_decay,
              learning_rate=args.lr,
              unfusion=0
            )
    elif args.model == 'text_only':
        model = BertSelfAttention(config=config)
        train(model=model , 
              df_for_train=df_for_train, 
              augment=False, 
              num_train_epochs=args.epochs, 
              batch_size=args.batch_size,
              warmup_steps=args.warmup,
              weight_decay=args.weight_decay,
              learning_rate=args.lr,
              unfusion=1
            )
    elif args.model == 'image_only':
        model = BertSelfAttention(config=config)
        train(model=model , 
              df_for_train=df_for_train, 
              augment=False, 
              num_train_epochs=args.epochs, 
              batch_size=args.batch_size,
              warmup_steps=args.warmup,
              weight_decay=args.weight_decay,
              learning_rate=args.lr,
              unfusion=2
            )
    elif args.model == 'concat':
        model = BertResNetModel(num_labels=3, text_pretrained='./bert')
        train(model=model , 
              df_for_train=df_for_train, 
              augment=False, 
              num_train_epochs=args.epochs, 
              batch_size=args.batch_size,
              warmup_steps=args.warmup,
              weight_decay=args.weight_decay,
              learning_rate=args.lr,
              unfusion=0
            )


    elif args.model == 'additive':
        model = BertREsnet_additive(num_labels=3, text_pretrained='./bert')
        train(model=model , 
              df_for_train=df_for_train, 
              augment=False, 
              num_train_epochs=args.epochs, 
              batch_size=args.batch_size,
              warmup_steps=args.warmup,
              weight_decay=args.weight_decay,
              learning_rate=args.lr,
              unfusion=0
            )
        
    elif args.model == 'cl':
        model = CL(config=config, temperature=0.02)
        train(model=model , 
              df_for_train=df_for_train, 
              augment=True, 
              num_train_epochs=args.epochs, 
              batch_size=args.batch_size,
              warmup_steps=args.warmup,
              weight_decay=args.weight_decay,
              learning_rate=args.lr,
              unfusion=0
            )
        
    elif args.model == 'test':
        model = BertSelfAttention_pre(config=config)
        model.load_state_dict(torch.load('./bertSelf.pt'))

        predictions = predict(model=model, 
                              df_for_test=df_for_test,
                              batch_size=args.batch_size,
                              )
        
    

        
        
        
    


    

