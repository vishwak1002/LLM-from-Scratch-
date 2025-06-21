
import torch
from tokenizer.GPTDataSetV1 import create_dataloader_v1



vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
max_length = 4
with open("verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
dataloader =create_dataloader_v1(
    raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False
)

data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("The Token ids are \n",inputs)
print("The Inputs shape are \n",inputs.shape)
