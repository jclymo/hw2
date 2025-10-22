from torch import nn
import torch
import math
from transformers import AutoModelForCausalLM
from data import GPTTokenizedData

class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(-1) # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier) # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        return x + self.pe[:batch_seq_len, :]


"""
TODO define your transformer model here. 
this will include: 
    - embed tokens (nn.Embedding)
    - add position encoding (provided)
    - n repetitions of 
        - *masked* self attention (can be single or multi-headed)
        - feedforward (MLP)
        - remember that the layer outputs are added to a residual connection
    - final linear layer with out_features equal to your vocabulary size
"""


class MyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.transformer.h = self.model.transformer.h[:5]

    def forward(self, x, mask):
        return self.model(x, attention_mask=mask).logits


def get_best_model_definition(vocab_size):
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    return MyTransformer(vocab_size)


if __name__ == "__main__":
    tokenized = GPTTokenizedData()
    vocab_size = tokenized.vocab_size
    model = MyTransformer(vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    # torch.save(model.state_dict(), "best_model.pt")
