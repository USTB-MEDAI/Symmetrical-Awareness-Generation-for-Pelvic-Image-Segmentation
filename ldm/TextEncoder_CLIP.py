import torch 
import torch.nn as nn             
from transformers import CLIPTokenizer, CLIPTextModel

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        l.weight.data = l.weight.data.bfloat16()
        if l.bias is not None:
            l.bias.data = l.bias.data.bfloat16()


class Text_Encoder(nn.Module):
    '''
    clip-vit-large-patch14为模型参数,需要提前单独下载并保存于本地
    '''
    def __init__(self,version='/home/syz/Xray-Diffsuion/clip-vit-large-patch14', device='cuda', max_length=77,freeze=True):
        super(Text_Encoder, self).__init__()
        # 定义文本的tokenizer和transformer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        print(f"type of tokenizer: {type(self.tokenizer)}")
        print(f"type of transformer: {type(self.transformer)}")
        
        self.device = device 
        self.max_length = max_length
        # self.fc1 = nn.Linear(768, 4*16*16*16) # 线性层降维

        # 冻结模型参数
        if freeze:
            self.freeze()
        
        self.transformer = self.transformer.to(torch.bfloat16)
        #self.convert_to_fp16()
    
    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
                                  
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        # self.tokenizer.apply(convert_module_to_f16)
        self.transformer.apply(convert_module_to_f16)

            
    def forward(self,text):
        # 对输入图片进行分词并编码,长度不足时直接padding到77
        batch_encoding = self.tokenizer(text,truncation=True,max_length=self.max_length,return_length=True,
                                        return_overflowing_tokens=False,padding='max_length',return_tensors='pt')
        
                
        # 拿出input_ids然后传入transformer进行特征提取d
        tokens = batch_encoding['input_ids'].to(self.device)
        # print(f"type of tokens: {tokens.dtype}")
        # 将输入张量转换为 bfloat16
        #tokens = tokens.to(torch.long)
        # print(f"type of tokens: {tokens.dtype}")
        outputs = self.transformer(input_ids=tokens, output_hidden_states=False)
        out = outputs.last_hidden_state #out shape [1,77,768]
        # print(f"type of out: {out.dtype}")        
        # 线性层降维
        # out = self.fc1(out) # out shape [1,77,4*16*16*16]
        # out = out.view(1,77,4,16,16,16) # out shape [1,4,16,16,16]
        # out = out.mean(dim=1)  # 在时间步维度（dim=1）取平均，形状变为 (1, 4, 16, 16, 16)
        
        return out


    
if __name__ == '__main__':
    
    #text = "The left pelvic bone articulates with the sacrum at the sacroiliac joint and with the right pelvic bone at the pubic symphysis. It has an acetabulum for the hip joint and a large obturator foramen."
    #text = "The sacrum is a single bone formed by five fused vertebrae. It is triangular, articulating with L5 above and the coccyx below. It has L-shaped facets for pelvic bones and sacral foramina for spinal nerve passage."
    #text = "The right pelvic bone articulates with the sacrum at the sacroiliac joint and with the left pelvic bone at the pubic symphysis. It has an acetabulum for the hip joint and a large obturator foramen."
    text = "The bones of the pelvis consist of the right and left pelvic (hip) bones, the sacrum."
    # 实例化模型
    model = Text_Encoder(device='cuda:1')
    # 输入文本进行特征提取
    out = model(text)
    #池化特征
    pooled_embedding = out.mean(dim=1)
    print(f"Input text: {text}")
    print(f"Output_type: {type(out)}")
    #print(f"Pooled embedding: {pooled_embedding}")
    print(f"Pooled embedding shape: {pooled_embedding.shape}")
    print(f"Output shape: {out.shape}") # 输出为[batch_size,seq_len,hidden_size] [1,77,768]