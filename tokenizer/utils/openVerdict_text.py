import re
def read_text_file(file_path):
    with open("verdict.txt", "r",encoding= 'utf-8') as f:
        raw_text = f.read()
    # Split the text into sentences
        print("Total number of characters in the text:", len(raw_text))
        return raw_text

def split_text_into_sentences(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed


def convert_to_token_ids(arr):
    allwords = sorted(set(arr))
    vocab_size = len(allwords)
    return vocab_size, allwords
    


def convert_to_vocab_dict(allwords):    
    return {word: i for i, word in enumerate(allwords)}


