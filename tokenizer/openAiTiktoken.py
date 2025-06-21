from utils.openVerdict_text import read_text_file

import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-2")


text = read_text_file("verdict.txt")
enc_txt = tokenizer.encode(text)
print(len(enc_txt))

enc_sample = enc_txt[50:] 


context_size = 4

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"Context: {context}, Desired: {desired}")

   
# text = (
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
# "of someunknownPlace."
# )


# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# strings = tokenizer.decode(integers)
# print(strings)


# unknown_text = "Akwirw ier"
# integers = tokenizer.encode(unknown_text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)
                                                           
