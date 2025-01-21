import numpy as np
import pandas as pd
from PIL import Image
import os
# from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_scheduler,BertModel,BertPreTrainedModel, BertLayer,BertConfig,ResNetModel,ResNetForImageClassification
from transformers.models.roberta.modeling_roberta import RobertaEncoder
import transformers
# import torchvision.transforms as transforms







max_seq_length = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




"""Concatenation"""
class BertResNetModel(nn.Module):
    def __init__(self, num_labels, text_pretrained='./bert', image_pretrained="./resnet-50" ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_pretrained)
        self.visual_encoder = ResNetModel.from_pretrained(image_pretrained)
        self.image_hidden_size = 2048
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(49152, 768)

        self.classifier = nn.Linear(self.text_encoder.config.hidden_size + self.image_hidden_size, num_labels)
        self.text_classifier = nn.Linear(self.text_encoder.config.hidden_size, num_labels)
        self.image_classifier= nn.Linear(self.image_hidden_size, num_labels)
    def forward(self, text, image):
        if (text is not None) and (image is not None):
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state[:, 0, :]

            img_feature = self.visual_encoder(image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            #print(img_feature.shape)
            features = torch.cat((text_feature, img_feature), 1)
            #print(features.shape)

            logits = self.classifier(features)
            # print(logits.shape)

            return logits
        
        elif text is not None:
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state
            # print(text_feature.shape)
            
            logits = self.text_classifier(self.linear(self.flatten(text_feature)))
            return logits
        
        else:
            img_feature = self.visual_encoder(image).last_hidden_state.view(-1, 49, 2048).max(1)[0]
            logits = self.image_classifier(img_feature)
            
            return logits



"""additive attention"""
class BertREsnet_additive(nn.Module):
    def __init__(self, num_labels, text_pretrained='./bert', image_pretrained="./resnet-50" ):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(text_pretrained)
        self.image_encoder = ResNetModel.from_pretrained(image_pretrained)
        self.hidden_size = 768
        self.img_linear = nn.Linear(in_features=2048, out_features=self.hidden_size)
        # self.txt_linear = nn.Linear(in_features=768, out_features=self.hidden_size)
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=0)
        self.wq = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.W = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.align = nn.Linear(in_features=49, out_features=max_seq_length)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self.hidden_size*max_seq_length, num_labels)

    


    def forward(self, text, image):
        if (text is not None) and (image is not None):
            # print(text.shape)
            fusion_features = 0
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state   #[batch_size, 128, 768]
            # print(text_feature.shape)
            # print(text_output.last_hidden_state.shape)
            img_feature = self.image_encoder(image).last_hidden_state.view(-1, 49, 2048)
            img_aligned = self.img_linear(img_feature)  # [batch_size, 49, 768]
            img_aligned = self.align(img_aligned.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, 128, 768]
            # print(text_feature.shape)
            # img_activated = self.tanh(img_aligned)
            # print(text_feature.shape)
            # print(img_activated.shape)
            # features = torch.cat((text_feature, img_activated), 1)
            for i in range(int(text_feature.shape[0])):
                alpha_1 = self.soft(self.wq(text_feature[i])/np.sqrt(self.hidden_size))
                alpha_2 = self.soft(self.wq(img_aligned[i])/np.sqrt(self.hidden_size))
                # print(alpha_1.shape)
                # print(alpha_2.shape)
                features = self.W(self.tanh(alpha_1 * text_feature[i] + alpha_2*img_aligned[i]))
                if(i == 0):
                    fusion_features = features.unsqueeze(0)
                else:
                    fusion_features = torch.cat((fusion_features, features.unsqueeze(0)), 0)
                # print(fusion_features.shape)
            
            fusion_features = self.flatten(fusion_features)
            logits = self.classifier(fusion_features)
            # print(logits.shape)

            return logits
        
        elif text is not None:
            fusion_features = 0
            text_output = self.text_encoder(**text)
            text_feature = text_output.last_hidden_state
            
            for i in range(int(text_feature.shape[0])):
                alpha_1 = self.soft(self.wq(text_feature[i])/np.sqrt(self.hidden_size))
                # alpha_2 = self.soft(self.wq(img_aligned[i])/np.sqrt(self.hidden_size))
                # print(alpha_1.shape)
                # print(alpha_2.shape)
                features = self.W(self.tanh(alpha_1 * text_feature[i]))
                if(i == 0):
                    fusion_features = features.unsqueeze(0)
                else:
                    fusion_features = torch.cat((fusion_features, features.unsqueeze(0)), 0)
                # print(fusion_features.shape)

            fusion_features = self.flatten(fusion_features)
            logits = self.classifier(fusion_features)
            # print(logits.shape)

            return logits
                                  
        else:
            fusion_features = 0        
            img_feature = self.image_encoder(image).last_hidden_state.view(-1, 49, 2048)
            img_aligned = self.img_linear(img_feature)  # [batch_size, 49, 768]
            img_aligned = self.align(img_aligned.permute(0, 2, 1)).permute(0, 2, 1)                      
            
            for i in range(int(img_feature.shape[0])):
                # alpha_1 = self.soft(self.wq(text_feature[i])/np.sqrt(self.hidden_size))
                alpha_2 = self.soft(self.wq(img_aligned[i])/np.sqrt(self.hidden_size))
                # print(alpha_1.shape)
                # print(alpha_2.shape)
                features = self.W(self.tanh(alpha_2*img_aligned[i]))
                if(i == 0):
                    fusion_features = features.unsqueeze(0)
                else:
                    fusion_features = torch.cat((fusion_features, features.unsqueeze(0)), 0)
                # print(fusion_features.shape)

            fusion_features = self.flatten(fusion_features)
            logits = self.classifier(fusion_features)
            # print(logits.shape)

            return logits
                                  

"""MLF"""

class BertSelfAttention(BertPreTrainedModel):
    def __init__(self, config, train_dim=768):
        super().__init__(config)
        self.bert = BertModel.from_pretrained('./bert')
        self.resnet = ResNetModel.from_pretrained("./resnet-50")
        self.comb_attention = BertLayer(config)
        self.train_dim = train_dim
        self.W = nn.Linear(in_features=2048, out_features=config.hidden_size)
        self.image_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential (
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(self.train_dim*2, self.train_dim // 2),
            nn.Linear(self.train_dim // 2, 3)
        )
        
        self.classifier_single =  nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(self.train_dim, self.train_dim // 2),
            nn.Linear(self.train_dim // 2, 3)
        )
        
    def forward(self,text_input, image_input):
        if (image_input is not None) and (text_input is not None):
           
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state
            image_features = self.resnet(image_input).last_hidden_state.view(-1, 49, 2048).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1)

            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state], 1)

            # 设置图像的attention_mask
            attention_mask = text_input.attention_mask
            
            image_attention_mask = torch.ones((attention_mask.size(0), 1)).to(device)
            attention_mask = torch.cat([image_attention_mask, attention_mask], 1).unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000

            # transformerEncoder
            image_text_attention_state = self.comb_attention(image_text_hidden_state, attention_mask)[0]

            image_pooled_output = self.image_pool(image_text_attention_state[:, 0, :])
            text_pooled_output = self.text_pool(image_text_attention_state[:, 1, :])
            final_output = torch.cat([image_pooled_output, text_pooled_output], 1)


            out = self.classifier(final_output)
            
            return out
            
        elif image_input is None:
            
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state
            attention_mask = text_input.attention_mask
            
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000
            
            
            attention_state = self.comb_attention(text_hidden_state, attention_mask)[0]
            text_pooled_output = self.text_pool(attention_state[:, 0, :])
            
            out = self.classifier_single(text_pooled_output)
            
            return out
            
            
            
        elif text_input is None:
            
            image_features = self.resnet(image_input).last_hidden_state.view(-1, 49, 2048).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1)
            
            image_attention_mask = torch.ones((image_hidden_state.size(0), 1)).to(device)
            attention_mask = (1.0 - image_attention_mask) * -10000
            attention_state = self.comb_attention(image_hidden_state, attention_mask)[0]
            image_pooled_output = self.image_pool(attention_state[:, 0, :])
            out = self.classifier_single(image_pooled_output)
            
            return out


class BertSelfAttention_pre(BertPreTrainedModel):
    def __init__(self, config, train_dim=768):
        super().__init__(config)
        self.bert = BertModel.from_pretrained('./bert')
        self.resnet = ResNetModel.from_pretrained("./resnet-50")
        self.comb_attention = BertLayer(config)
        self.train_dim = train_dim
        self.W = nn.Linear(in_features=2048, out_features=config.hidden_size)
        self.image_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential (
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.train_dim*2, self.train_dim // 2),
            nn.Linear(self.train_dim // 2, 3)
        )
        
        
    def forward(self,text_input, image_input):
        if (image_input is not None) and (text_input is not None):
           
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state
            image_features = self.resnet(image_input).last_hidden_state.view(-1, 49, 2048).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1)

            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state], 1)

            # 设置图像的attention_mask
            attention_mask = text_input.attention_mask
            
            image_attention_mask = torch.ones((attention_mask.size(0), 1)).to(device)
            attention_mask = torch.cat([image_attention_mask, attention_mask], 1).unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000

            # transformerEncoder
            image_text_attention_state = self.comb_attention(image_text_hidden_state, attention_mask)[0]

            image_pooled_output = self.image_pool(image_text_attention_state[:, 0, :])
            text_pooled_output = self.text_pool(image_text_attention_state[:, 1, :])
            final_output = torch.cat([image_pooled_output, text_pooled_output], 1)


            out = self.classifier(final_output)
            
            return out
            
        


class MLF(nn.Module):
    def __init__(self, config, train_dim=768):
        super(MLF, self).__init__()
        self.bert = BertModel.from_pretrained('./bert')
        self.resnet = ResNetModel.from_pretrained('./resnet-50')
        self.hidden_size = 768
        self.train_dim = train_dim
        self.config= config
        self.gelu = nn.GELU()
        self.soft = nn.Softmax()
        self.tanh = nn.Tanh()
        self.text_change = nn.Sequential(
            nn.Linear(self.hidden_size, self.train_dim),
            nn.Tanh()   
        )
        self.image_change = nn.Sequential(
            nn.Linear(2048, self.train_dim),
            nn.Tanh()
        )
        # parse.add_argument('-tran_dim', type=int, default=768, help='Input dimension of text and picture encoded transformer')
        # self.image_change = nn.Linear(in_features=2048, out_features=768)
        # self.encodeLayer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.hidden_size // 64)
        self.TransformerEncoder = RobertaEncoder(config=config)
        self.output_attention = nn.Sequential(
                nn.Linear(self.train_dim, self.train_dim // 2),
                nn.GELU(),
                nn.Linear(self.train_dim // 2, 1)
            )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.train_dim, self.train_dim // 2),
            nn.GELU(),
            nn.Linear(self.train_dim // 2, 3)

        )
       
    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask
    


    def forward(self, text_input, image_input):
        text_features = self.bert(**text_input).last_hidden_state
        image_features = self.resnet(image_input).last_hidden_state.view(-1, 49, 2048).contiguous()
        image_features = self.image_change(image_features)

        # image_encode 之前需要获得图片的attention_mask
        image_attenion = torch.ones((text_input.attention_mask.size(0), 49)).to(device)
        extended_attention_mask = self.get_extended_attention_mask(image_attenion)
        image_encoded = self.TransformerEncoder(image_features,
                                                attention_mask=None,
                                                head_mask=None,
                                                encoder_hidden_states=None,
                                                encoder_attention_mask=extended_attention_mask,
                                                past_key_values=None,
                                                output_attentions=self.config.output_attentions,
                                                output_hidden_states=self.config.output_hidden_states,
                                                return_dict= self.config.use_return_dict
                                                ).last_hidden_state  #[batch_size, seq_length(49), hidden_size]

        ### concat之前是否需要激活函数？？？？论文源码里面有
        text_features = self.text_change(text_features)
        text_image_features = torch.cat((text_features, image_encoded), dim=1)  ### [batch_size, 64+49, 768]

        text_image_attention = torch.cat((image_attenion, text_input.attention_mask), dim=1)
        extended_attention_mask = self.get_extended_attention_mask(text_image_attention)
        # text_image_state_encoded = self.TransformerEncoder(text_image_hidden_state)
        text_image_encoded = self.TransformerEncoder(text_image_features,
                                                     attention_mask=extended_attention_mask,
                                                     encoder_hidden_states=None,
                                                     encoder_attention_mask=extended_attention_mask,
                                                     past_key_values=None,
                                                     output_attentions=self.config.output_attentions,
                                                     output_hidden_states=self.config.output_hidden_states,
                                                     return_dict= self.config.use_return_dict
                                                     ).last_hidden_state
        
        text_image_output = text_image_encoded.contiguous()

        text_image_mask = text_image_attention.permute(1, 0).contiguous()
        text_image_mask = text_image_mask[0:text_image_output.size(1)]
        text_image_mask = text_image_mask.permute(1, 0).contiguous()

        text_image_alpha = self.output_attention(text_image_output)
        text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
        text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

        text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)

        return text_image_output
        

class CL(nn.Module):
    def __init__(self, config, temperature, train_dim=768):
        super(CL, self).__init__()
        self.fusion_model = MLF(config=config)
        self.temperature = temperature
        self.train_dim = train_dim
        self.linear_change = nn.Sequential(
            nn.Linear(self.train_dim, self.train_dim),
            nn.GELU(),
            nn.Linear(self.train_dim, self.train_dim)

        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.train_dim, self.train_dim // 2),
            nn.GELU(),
            nn.Linear(self.train_dim // 2, 3)

        )

    def forward(self, augment_text, augment_image, text, image, labels, target_labels):
        org_text_image_features = self.fusion_model(text, image)
  
        output = self.classifier(org_text_image_features)

        if augment_text != None:

            # org_text_image_features = self.fusion_model(text, image)
            aug_text_image_features = self.fusion_model(augment_text, augment_image)
            org_res_change = self.linear_change(org_text_image_features)
            aug_res_change = self.linear_change(aug_text_image_features)

            l_pos_neg = torch.einsum('nc,ck->nk', [org_res_change, aug_res_change.T])
            cl_lables = torch.arange(l_pos_neg.size(0))
            cl_lables = cl_lables.to(device)

            l_pos_neg /= self.temperature

            l_pos_neg_self = torch.einsum('nc,ck->nk', [org_res_change, org_res_change.T])
            l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)
            l_pos_neg_self = l_pos_neg_self.view(-1)

            cl_self_labels = target_labels[labels[0]]
            for index in range(1, org_text_image_features.size(0)):
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[index]] + index*labels.size(0)), 0)

            l_pos_neg_self = l_pos_neg_self / self.temperature
            cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)

            return output, l_pos_neg, cl_lables, cl_self_loss
        else:
            return output





"""BERT"""     
class VLBertModel(nn.Module):

    def __init__(self, num_labels, text_pretrained='bert-base-uncased'):
        super().__init__()

        self.num_labels = num_labels
        self.text_encoder = AutoModel.from_pretrained(text_pretrained)
        self.classifier = nn.Linear(
        self.text_encoder.config.hidden_size, num_labels)
        


    def forward(self, text):
        output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        logits = self.classifier(output.last_hidden_state[:, 0, :]) # CLS embedding
        return logits
    

"""ResNet-50"""
class ResNet_only_model(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.resnet = ResNetForImageClassification.from_pretrained("./resnet-50")
        self.linear = nn.Linear(1000, 3)
        self.classifier = nn.Linear(
            1000, num_labels
        )
        
        
    def forward(self, image):
        image_features = self.resnet(image).logits
        # image_changed = self.linear(image_features)
        
        logits = self.classifier(image_features)
        
        return logits