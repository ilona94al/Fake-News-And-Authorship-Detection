import re
import numpy as np


# define label: fake=0,real=1
def define_expected_classification(fake_num, all_n):
    expected_classification = np.empty(all_n, int)
    expected_classification[0:fake_num] = 0
    expected_classification[fake_num:all_n] = 1
    return expected_classification


# cleaner of the text
def preProcess(string):
    string = re.sub(" +", " ", string)
    string = re.sub("'", "", string)
    string = re.sub("[^a-zA-Z ]", " ", string)
    string = string.lower()
    return string


def Text_processing(name):
    import csv
    fake_news = []
    real_news = []
    i = 0
    with open('DB/fake50.csv', 'r', encoding='utf-8') as file:
        fakeReader = csv.reader(file)
        for row in fakeReader:
            fake_news[i:] = row[1:2]
            i = i + 1
    i = 0
    with open('DB/true50.csv', 'r', encoding='utf-8') as file:
        trueReader = csv.reader(file)
        for row in trueReader:
            real_news[i:] = row[1:2]
            i = i + 1

    news = fake_news + real_news
    # Y - labels for fake/real news
    Y = define_expected_classification(len(fake_news), len(news))

    # clean text
    texts_ = []
    for text in news:
        texts_.append(preProcess(text))
    print(texts_)

    # Import tokenizer from transformers package
    from transformers import BertTokenizer

    # Load the tokenizer of the "bert-base-cased" pretrained model
    tz = BertTokenizer.from_pretrained("bert-base-cased")
    encodedText_IDs = []
    attn_maskForBertInput = []
    i = 0
    for x in texts_:
        # Encode the sentence
        encoded = tz.encode_plus(
            text=texts_[i],  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=512,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask=True,  # Generate the attention mask
            return_tensors='pt',  # ask the function to return PyTorch tensors
        )

        # Get the input IDs and attention mask in tensor format
        input_ids = encoded['input_ids']
        encodedText_IDs.append(input_ids)
        attn_mask = encoded['attention_mask']
        attn_maskForBertInput.append(attn_mask)
        i = i + 1
    print(encodedText_IDs)
    print(attn_maskForBertInput)


# The “attention mask” tells the model which tokens should be attended to and which (the [PAD] tokens) should not.
# It will be needed when we feed the input into the BERT model.
# https://huggingface.co/transformers/preprocessing.html

if __name__ == '__main__':
    Text_processing('Lets go')
