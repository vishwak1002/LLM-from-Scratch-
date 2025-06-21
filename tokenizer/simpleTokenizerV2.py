import re

class SimpleTokenizerV2:
    """
    A simple tokenizer that splits text into words based on whitespace and punctuation.
    """

    def __init__(self,vocab):
        vocab['<|endoftext|'] = len(vocab)  # Add end of text token
        vocab['<|unk|>'] = len(vocab)
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
    
    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        tokens = [self.int_to_str[i] for i in ids]
        text = ' '.join(tokens)
        final_text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return final_text
 

 

    