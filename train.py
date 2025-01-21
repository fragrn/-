import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW,RMSprop, Adam
from tqdm import tqdm, trange
from time import perf_counter
from transformers import AutoModel, AutoTokenizer, get_scheduler,BertModel,BertPreTrainedModel, BertLayer,BertConfig
from data_util import dataloader_for_train
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



max_seq_length = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, df_for_train, augment, num_train_epochs, batch_size, learning_rate, warmup_steps, weight_decay, unfusion):
    train, valid = train_test_split(df_for_train, test_size=0.1, random_state=42)
    train_dataloader, valid_dataloader = dataloader_for_train(train=train, valid=valid, augment=augment, batch_size=batch_size)
    bert_tokenizer = AutoTokenizer.from_pretrained('./bert_tokenizer')
    model = model.to(device)

    if augment == False:
        t_total = len(train_dataloader) * num_train_epochs


        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)


        criterion = nn.CrossEntropyLoss()

        model.train()


        train_epochs_loss = []
        valid_epochs_loss = []
        train_acc = []
        train_f1 = []
        train_precision = []
        train_recall = []
        val_acc = []
        val_f1 = []
        val_precision = []
        val_recall = []


        start = perf_counter()
        for epoch_num in trange(num_train_epochs, desc='Epochs'):
            epoch_total_loss = 0
            
            train_epoch_loss = []
            train_pred_results = []
            train_target_results = []
            
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                # print(batch)
                b_text, b_labels, b_imgs = batch
               
            
                b_inputs = bert_tokenizer(
                    list(b_text), truncation=True, max_length=max_seq_length,
                    return_tensors="pt", padding='max_length'
                )

    
                b_labels = b_labels.to(device)
                b_imgs = b_imgs.to(device)
                b_inputs = b_inputs.to(device)
            
                model.zero_grad()
                if unfusion == 1:
                    b_imgs = None
                elif unfusion == 2:
                    b_inputs == None
                
                b_logits = model(b_inputs, b_imgs)
                loss = criterion(b_logits, b_labels)
                loss.backward()

                optimizer.step()
                scheduler.step()
                
                train_epoch_loss.append(loss.item())
                b_logits = torch.max(b_logits,1)[1]
                train_pred_results += b_logits.detach().cpu().tolist()
                train_target_results += b_labels.detach().cpu().tolist()
                
            train_epochs_loss.append(np.average(train_epoch_loss))
            
            acc = accuracy_score(train_pred_results,train_target_results)
            f1 = f1_score(train_pred_results, train_target_results, average='weighted')
            recall = recall_score(train_pred_results, train_target_results, average='weighted')
            precision = precision_score(train_pred_results, train_target_results, average='weighted')
            
            train_acc.append(acc)
            train_f1.append(f1)
            train_precision.append(precision)
            train_recall.append(recall)
            
            
            print('epoch =', epoch_num+1)
            print("    train_loss = {}, train_acc = {}, train_f1 = {}, train_recall={}, train_precision={}".format(np.average(train_epoch_loss), acc, f1, recall, precision))
            print('    learning rate =', optimizer.param_groups[0]["lr"])
            
            
            with torch.no_grad():
                model.eval()
                val_epoch_loss = []
                val_pred_results = []
                val_target_results = []
                
                    
                for batch in tqdm(valid_dataloader):
                    b_text, b_labels, b_imgs = batch
                    b_inputs = bert_tokenizer(list(b_text), truncation=True, max_length=max_seq_length, return_tensors="pt", padding='max_length')
                    b_labels = b_labels.to(device)
                    b_imgs = b_imgs.to(device)
                    b_inputs = b_inputs.to(device)
                    
                    if unfusion == 1:
                        b_imgs = None
                    elif unfusion == 2:
                        b_inputs == None

                    b_logits = model(b_inputs, b_imgs)
                    val_loss = criterion(b_logits, b_labels)
                    
                    val_epoch_loss.append(val_loss.item())
                    b_logits = torch.max(b_logits,1)[1]
                    
                    
                    val_pred_results += b_logits.detach().cpu().tolist()
                    val_target_results += b_labels.detach().cpu().tolist()


                    
                valid_epochs_loss.append(np.average(val_epoch_loss))
                
                acc = accuracy_score(val_pred_results, val_target_results)
                f1 = f1_score(val_pred_results, val_target_results, average='weighted')
                recall = recall_score(val_pred_results, val_target_results, average='weighted')
                precision = precision_score(val_pred_results, val_target_results, average='weighted')
                if acc > 0.74:
                    print('saving_model')
                    torch.save(model.state_dict(), './model.pt')
                val_acc.append(acc)
                val_f1.append(f1)
                val_precision.append(precision)
                val_recall.append(recall)
                
                print("    valid loss = {}, val_acc = {}, val_f1 = {}, val_recall={}, val_precision={}".format(np.average(val_epoch_loss), acc, f1, recall, precision))
            
        end = perf_counter()
        resnet_training_time = end- start
        print('Training completed in ', resnet_training_time, 'seconds')

    else:
        t_total = len(train_dataloader) * num_train_epochs


        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)


        criterion = nn.CrossEntropyLoss()

        model.train()


        train_epochs_loss = []
        valid_epochs_loss = []
        train_acc = []
        train_f1 = []
        train_precision = []
        train_recall = []
        val_acc = []
        val_f1 = []
        val_precision = []
        val_recall = []


        start = perf_counter()
        for epoch_num in trange(num_train_epochs, desc='Epochs'):
            acc, nums = 0, 0
            train_epoch_loss = []
            train_pred_results = []
            train_target_results = []
            
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                # print(batch)
                b_text, b_text_aug, b_labels, b_imgs = batch
               
            
                b_inputs = bert_tokenizer(
                    list(b_text), truncation=True, max_length=max_seq_length,
                    return_tensors="pt", padding=True
                )
                b_inputs_aug = bert_tokenizer(
                    list(b_text_aug), truncation=True, max_length=max_seq_length,
                    return_tensors="pt", padding='max_length'
                )
                
                temp_labels = [b_labels-0, b_labels-1, b_labels-2]
                target_labels = []
                for i in range(3):
                    temp_target_labels = []
                    for j in range(temp_labels[0].size(0)):
                        if temp_labels[i][j] == 0:
                            temp_target_labels.append(j)
                    target_labels.append(torch.LongTensor(temp_target_labels[:]))

                b_labels = b_labels.to(device)
                b_imgs = b_imgs.to(device)
                b_inputs = b_inputs.to(device)
                b_inputs_aug = b_inputs_aug.to(device)
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].to(device)
             
                model.zero_grad()
                # augment_text, augment_image, text, image, labels, target_labels
                

                b_logits, l_pos_neg, cl_lables, cl_self_loss = model(b_inputs_aug, b_imgs, b_inputs, b_imgs, b_labels, target_labels)


                loss = criterion(b_logits, b_labels)
                cl_loss = criterion(l_pos_neg, cl_lables)


                # epoch_total_loss += loss.item()

                # Perform a backward pass to calculate the gradients
                loss = (loss + cl_loss*0.5 + cl_self_loss*0.5) / b_logits.size(0)
                loss.backward()


                optimizer.step()
                scheduler.step()
                
                train_epoch_loss.append(loss.item())
       
                b_logits = torch.max(b_logits,1)[1]
                train_pred_results += b_logits.detach().cpu().tolist()
                train_target_results += b_labels.detach().cpu().tolist()
                
                
                acc += sum(b_logits == b_labels).cpu()
                nums += b_labels.size()[0]
                
              

            train_epochs_loss.append(np.average(train_epoch_loss))
            acc = accuracy_score(train_pred_results,train_target_results)
            f1 = f1_score(train_pred_results, train_target_results, average='weighted')
            recall = recall_score(train_pred_results, train_target_results, average='weighted')
            precision = precision_score(train_pred_results, train_target_results, average='weighted')
            
            train_acc.append(acc)
            train_f1.append(f1)
            train_recall.append(recall)
            train_precision.append(precision)
            
            print('epoch =', epoch_num)
            # print("    train acc = {:.3f}%".format(100 * acc / nums))
            print("    train_loss = {}, train_acc = {}, train_f1 = {}, train_recall={}, train_precision = {}".format(np.average(train_epoch_loss), acc, f1, recall, precision))
            # print('    epoch_loss =', epoch_total_loss)
            # print('    avg_epoch_loss =', avg_loss)
            print('    learning rate =', optimizer.param_groups[0]["lr"])
            
            
            with torch.no_grad():
                model.eval()
                val_epoch_loss = []
                val_pred_results = []
                val_target_results = []
                acc, nums = 0, 0
                    
                for batch in tqdm(valid_dataloader):
                    b_text,b_text_aug, b_labels, b_imgs = batch
                    # b_labels = one_hot(b_labels)
                    b_inputs = bert_tokenizer(list(b_text), truncation=True, max_length=max_seq_length, return_tensors="pt", padding=True)
                    b_inputs_aug = bert_tokenizer(list(b_text_aug), truncation=True, max_length=max_seq_length, return_tensors="pt", padding=True)
                    b_labels = b_labels.to(device)
                    b_imgs = b_imgs.to(device)
                    b_inputs = b_inputs.to(device)
                    
                    b_logits = model(None, None, text=b_inputs, image=b_imgs, labels=None, target_labels=None)
                    val_loss = criterion(b_logits, b_labels)
                    
                    val_epoch_loss.append(val_loss.item())
                 
                    b_logits = torch.max(b_logits,1)[1]
                    
                    
                    val_pred_results += b_logits.detach().cpu().tolist()
                    val_target_results += b_labels.detach().cpu().tolist()
                    # print(b_labels)
                    acc += sum(b_logits == b_labels).cpu()
                    nums += b_labels.size()[0]
                    
                valid_epochs_loss.append(np.average(val_epoch_loss))

                acc = accuracy_score(val_pred_results, val_target_results)
                f1 = f1_score(val_pred_results, val_target_results, average='weighted')
                recall = recall_score(val_pred_results, val_target_results, average='weighted')
                precision = precision_score(val_pred_results, val_target_results, average='weighted')

                if acc > 0.74:
                    print('saving_model')
                    torch.save(model.state_dict(), './cl_fusion_model.pt')
                val_acc.append(acc)
                val_f1.append(f1)
                val_recall.append(recall)
                val_precision.append(precision)

                
                print("    valid loss = {}, val_acc = {}, val_f1 = {}, val_recall={}, val_precision={}".format(np.average(val_epoch_loss), acc, f1, recall, precision))
            
        end = perf_counter()
        resnet_training_time = end- start
        print('Training completed in ', resnet_training_time, 'seconds')
        
