
from  utils.openVerdict_text import read_text_file,split_text_into_sentences,convert_to_token_ids,convert_to_vocab_dict
from simpleTokenizerV1 import SimpleTokenizerV1
from simpleTokenizerV2 import SimpleTokenizerV2


text_file = "verdict.txt"
raw_text = read_text_file(text_file)
split_text = split_text_into_sentences(raw_text)
vocab_size, allwords = convert_to_token_ids(split_text)
vocab = convert_to_vocab_dict(allwords)

tokenizer1 = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))


tokenizer2 = SimpleTokenizerV2(vocab)
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
print(tokenizer2.encode(text))
print(tokenizer2.decode(tokenizer2.encode(text)))


