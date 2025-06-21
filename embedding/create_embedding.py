
import torch
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.GPTDataSetV1 import create_dataloader_v1

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
max_length = 4
with open("./embedding/verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
dataloader =create_dataloader_v1(
    raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False
)

data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("The Token ids are \n",inputs)
print("The Inputs shape are \n",inputs.shape)


token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)



input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


