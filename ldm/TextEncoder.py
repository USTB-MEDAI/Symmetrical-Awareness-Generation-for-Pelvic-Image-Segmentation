import torch 
import torch.nn as nn             
from transformers import CLIPTokenizer, CLIPTextModel

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="/home/syz/Xray-Diffsuion/clip-vit-large-patch14", device="cuda:0", 
                 max_length=77, checkpoint_path=None):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.checkpoint_path = checkpoint_path
        self.classifier = Classifier(input_dim=768*77, num_classes=3)
        if self.checkpoint_path is not None:
            self.classifier.load_state_dict(torch.load(self.checkpoint_path)['model'])
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        self.classifier = self.classifier.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        # if not isinstance(text, str):
        #     print(f"Warning: text is not a string, converting to string: {text}")
        #     text = str(text)
        # print(f"type of text: {type(text)}")
        # print(f"texts: {text}")
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        # print(f"tokens: {tokens}")  
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        if self.checkpoint_path is not None:
            labels = self.classifier(z)
            labels = labels.argmax(dim=1)
            labels_list = [str(tensor.cpu().numpy()) for tensor in labels]
            # print(f"labels: {labels_list}")
            labels_encoding = self.tokenizer(labels_list, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            labels_tokens = labels_encoding["input_ids"].to(self.device)        
            outputs = self.transformer(input_ids=labels_tokens)
            lz = outputs.last_hidden_state
            # print(f"z shape: {z.shape}", f"lz shape: {lz.shape}")
            condz = z + lz
            return condz
        else:
            return z

    def encode(self, text):
        return self(text)
    
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

    
    
    
if __name__ == "__main__":
    encoder = FrozenCLIPEmbedder()
    text = 1
    z = encoder.forward(text)
    print(f"z shape: {z.shape}")
    print(f"z: {z}")