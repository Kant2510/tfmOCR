from model.backbone.cnn import CNN
from tool.config import Cfg
from torch import nn
from torchsummary import summary
from model.seqmodel.transformer import LanguageTransformer
# class VietOCR(nn.Module):
#     def __init__(self,
#                  vocab_size=141,
#                  **transformer_args,):
        
#         super(VietOCR, self).__init__()
        
#         self.cnn = CNN(backbone, **cnn_args)
#         self.transformer = LanguageTransformer(vocab_size, **transformer_args)

#     def forward(self, img, tgt_input, tgt_key_padding_mask):
#         """
#         Shape:
#             - img: (N, C, H, W)
#             - tgt_input: (T, N)
#             - tgt_key_padding_mask: (N, T)
#             - output: b t v
#         """
#         src = self.cnn(img)
#         outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
#         return outputs
class VietOCR(nn.Module):
    def __init__(self,
                 backbone,
                 cnn_args,):
        
        super(VietOCR, self).__init__()
        
        self.cnn = CNN(backbone, **cnn_args)

    def forward(self, img):
        src = self.cnn(img)
        print(src.shape)
        return src
config = Cfg.load_config_from_file('defaults.yml')
model = VietOCR('resnet50', config['cnn'])#, input_channel=3, output_channel=256, pretrained=True)
# transformer = LanguageTransformer(vocab_size, **transformer_args)

summary(model, (3, 96, 1600))
import torch
t = torch.randn(32, 512, 3, 50)
print(t.shape)
t = t.transpose(-1, -2)
print(t.shape)
t = t.flatten(2)

print(t.shape)
t = t.permute(-1, 0, 1)
print(t.shape)
# from torch import nn
# import torch
# transformer_model = nn.Transformer(nhead=2, num_encoder_layers=12, d_model=6)
# src = torch.rand((2, 4, 6))
# tgt = torch.rand((3, 4, 6))
# out = transformer_model(src, tgt)
# print(out)