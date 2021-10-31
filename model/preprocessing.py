import re
from nltk.corpus import stopwords


#   gets text
#   preprocesses text, replace capital letters, remove unwanted chars, remove stopwords....
#   returns clean text
def text_preprocessing(text):
    regex_legal_letters = re.compile('[^ \\t\\n\\v\\f\\ra-zA-Z]')
    text = regex_legal_letters.sub('', text)
    text = text.lower()
    words_arr = text.split()
    clean_word_arr = remove_stop_words(words_arr)
    clean_text = " ".join(word for word in clean_word_arr)
    return clean_text


#   gets words array
#   removes all words that appear in stopwords list
#   returns clean words array
def remove_stop_words(words_arr):
    stop_words = set(stopwords.words('english'))
    clean_word_arr = [w for w in words_arr if not w in stop_words]
    return clean_word_arr


#   gets text and number of maximum tokens in text
#   separates text to blocks with max_text_len tokens
#   returns list with blocks, each block in size max_text_len
def separate_text_to_blocks(text, max_text_len):
    words = text.split()
    blocks = []
    current = ""
    tokens = 0
    for word in words:
        if tokens > max_text_len - 3:
            blocks.append(current)
            current = word
            tokens = 0
        else:
            current += word + " "
            tokens += 1
    blocks.append(current)
    return blocks
