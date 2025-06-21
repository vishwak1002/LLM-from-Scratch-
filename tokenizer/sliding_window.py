import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")   
#now Runing Sliding Window
raw_text = " "
with open("verdict.txt","r",encoding = "utf-8") as f :
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))


enc_sample = enc_text[50:]
context_size = 5
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)