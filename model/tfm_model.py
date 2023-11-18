from tfmOCR.model.backbone.cnn import CNN
from tfmOCR.model.seqmodel.transformer import LanguageTransformer
from tfmOCR.model.vocab import Vocab
from torch import nn

class tfmOCR(nn.Module):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args, 
                 transformer_args):
        
        super(tfmOCR, self).__init__()
        
        self.cnn = CNN(backbone, **cnn_args)
        self.transformer = LanguageTransformer(vocab_size, **transformer_args)

    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        return outputs

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    model = tfmOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'])
    
    model = model.to(device)

    return model, vocab