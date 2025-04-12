import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# current_dir = os.getcwd()
# src_path = os.path.join(current_dir, 'src')
# os.chdir(src_path)
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import sys
sys.path.append('/disk/SYZ/Xray-Diffsuion')
from src.open_clip import create_model_and_transforms, get_mean_std
import torch
### Logistic Regression Classifier    
from sklearn.linear_model import LogisticRegression

mean, std = get_mean_std()

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Text_Encoder(object):
    def __init__(self, model_name, pretrained_weights, device, text_encoder_name, context_length=77, freeze=True, checkpoint_path=None):
        super(Text_Encoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.device = device
        self.model, _, _ = create_model_and_transforms(
                            model_name,
                            pretrained_weights,
                            precision='amp',
                            device = device,
                            force_quick_gelu=True,
                            mean=mean, std=std,
                            inmem=True,
                            text_encoder_name=text_encoder_name,)
        self.BioCLIPTextEncoder = self.model.encode_text
        self.context_length = context_length
        self.checkpoint_path = checkpoint_path
        self.classifier = Classifier(input_dim=512, num_classes=3).to(self.device)
        if self.checkpoint_path is not None:
            self.classifier.load_state_dict(torch.load(self.checkpoint_path)['model'])
        
        # 冻结模型参数
        if freeze:
            self.freeze()
    
    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, text):
        tokens = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=self.context_length,
                    padding='max_length',
                    truncation=True,
                    )
        # print(f"text: {text}")
        # print("tokens.keys():", tokens.keys())
        # print("tokens:", tokens.tokens())
        # print("tokens.input_ids:", tokens.input_ids)
        input_ids = (tokens.input_ids).to(self.device)
        # print("input_ids.shape:", input_ids.shape)
        outputs = self.BioCLIPTextEncoder(input_ids)
        z = outputs.view(-1, 1, 512)
        if self.checkpoint_path is not None:
            labels = self.classifier(z)
            labels = labels.argmax(dim=1)
            labels_list = [str(tensor.cpu().numpy()) for tensor in labels]
            # print(f"labels: {labels_list}")
            tokens = self.tokenizer(
                                labels_list,
                                return_tensors='pt',
                                max_length=self.context_length,
                                padding='max_length',
                                truncation=True,
                                )
            input_ids = (tokens.input_ids).to(self.device)
            # print("input_ids.shape:", input_ids.shape)
            outputs = self.BioCLIPTextEncoder(input_ids).to(self.device)  
            lz = outputs.unsqueeze(1)
            # print(f"z shape: {z.shape}", f"lz shape: {lz.shape}")
            # condz = torch.cat((z, lz), dim=1)
            condz = z + lz
            return condz   # B, 1, 512
        else:
            return z   # B, 1, 512


if __name__ == '__main__':
    text_encoder = Text_Encoder(model_name = 'ViT-B-16-quickgelu', 
                               pretrained_weights = "/disk/syz/UniMed-CLIP-main/unimed_clip_vit_b16.pt", 
                               device = torch.device('cuda:0'),
                               text_encoder_name = "/disk/syz/UniMed-CLIP-main/BiomedNLP-BiomedBERT-base-uncased-abstract", 
                               context_length = 77,
                               freeze=True)
    text = "The cervical vertebrae are seven small vertebrae between the skull and thorax. Each has a foramen in the transverse process. C1 (atlas) lacks a vertebral body and has large transverse processes. C2 (axis) features a large dens extending superiorly."
    outputs= text_encoder(text)
    print(f"outputs.shape: {outputs.shape}, outputs.dtype: {outputs.dtype}")#1, 512
    
    