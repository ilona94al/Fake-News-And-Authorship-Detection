import re

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import numpy as np

config_128tokens = {
    'max_seq_len': 128,
    'seq_len_cnnb_4': 74,
    'seq_len_cnnb_5': 14
}

config = config_128tokens


def seperate_to_blocks(book):
    words = iter(book.split())
    lines, current = [], next(words)
    tokens = 0
    for word in words:
        if tokens > config['max_seq_len'] - 3:
            lines.append(current)
            current = word
            tokens = 0
        else:
            current += " " + word
            tokens += 1
    lines.append(current)
    return lines


def text_preprocessing(text):
    regex_legal_letters = re.compile('[^ \\t\\n\\v\\f\\ra-zA-Z]')
    text = regex_legal_letters.sub('', text)
    text = text.lower()
    words_arr = text.split()
    clean_word_arr = remove_stop_words(words_arr)
    clean_text = " ".join(word for word in clean_word_arr)
    return clean_text


from nltk.corpus import stopwords


def remove_stop_words(words_arr):
    stop_words = set(stopwords.words('english'))
    clean_word_arr = [w for w in words_arr if not w in stop_words]
    return clean_word_arr
def read_book( books_dir_path, author_name, book_name):
    with open(books_dir_path + '/' + author_name + '/' + book_name+".txt", 'r', encoding='UTF-8') as book_file:
        book_string = book_file.read()
        return book_string
import os

#os.chdir("TRAINED_MODELS/FakeNews/")

books="C:/Users/Ilona Aliyev/PycharmProjects/Fake-News-And-Authorship-Detection/DATABASE/books_not_for_train"
book=read_book(books_dir_path=books,author_name="shakespeare",book_name="Cardenio")

# true = "LONDON (Reuters) - Abdul Daoud spilt most of the cappuccino into the saucer the first time he served Princess Diana, his nerves getting the better of him. Almost 20 years on since she was killed when her car crashed in a Paris tunnel, he still works surrounded by pictures of the woman he calls  the princess of the people  in his cafe, named Diana, his very personal attempt to keep her memory alive.  My promise to her is to put this place as a tribute for her,  he said of his cafe, set up in 1989, near London s Kensington Gardens, home to the palace where Diana used to live.  For him, celebrating her life is  business as usual  at the cafe where visitors can eat Diana salads or Diana burgers and where he said she used to stop by regularly.   She is the princess of the people, always,  he said, adding that he does not believe she will ever be forgotten. But many younger Britons said that while they can understand the fascination with the princess, whose struggles to fit in to the royal household played out in the full glare of the media, they don t feel it themselves.       I think she maybe meant more to my mother,  said Stephen Butler in the west London area Diana used to live.  When she died I remember my mother shaking me awake and being quite devastated about it.  Student Shermine Grigorius was three-years-old when Diana died but after being told stories about the Princess of Wales by her mother, sees her as a  symbol of kindness .  While the royals have always been dutifully charitable, Diana was known for going beyond her in-laws, or even any celebrity at the time, in her philanthropy.  Whether in charity work or in royal life, she earned a reputation for being a rebel who defied convention: from campaigning for a worldwide ban on landmines despite opposition from the British government to flouting royal protocol to speak candidly about her experiences with bulimia and infidelity.   She bought a different side to the whole monarchy,  said Anika Wijuria, a 30-year-old project manager.  They were quite stiff and she was quite liberal.  At the Da Mario restaurant, Marco Molino remembers another side of Diana, describing a  down to earth  woman who liked to eat Italian dishes with her sons, Princes William and Harry, or friends.  Her personality was very normal, very down to earth, very friendly,  he said near an oil painting of Diana on the wall.  I think that s what she really wanted - a bit of normality ... Here was one of the places where she could achieve that.  Ronald van Bronkhorst, who has lived above Da Mario since the 70s, also said she never made flashy entrances.  Her legacy will never leave ... You think about her all the time, especially in the area we live in. "
true_text = []
preprocessed_text = text_preprocessing(book)
tweet_blocks = seperate_to_blocks(preprocessed_text)
true_text.extend((block for block in tweet_blocks))

trained_model_name = 'TrainedFakeBERTModelShakespeare2.h5'
model = tf.keras.models.load_model(trained_model_name, custom_objects={'KerasLayer': hub.KerasLayer})


Y_predicted_prob = model.predict(x=tf.constant(true_text))
Y_predicted = np.argmax(Y_predicted_prob, -1)
# count_real= len([Y_predicted == 0])

fake_numb=np.count_nonzero(Y_predicted)
real_numb=np.count_nonzero(Y_predicted == 0)
all=Y_predicted.shape

print(fake_numb)
print(real_numb)

import matplotlib.pyplot as plt


plt.plot()
plt.plot(Y_predicted_prob[0])
plt.plot(Y_predicted_prob[1])

plt.title('True/Fake prob in book parts')
plt.ylabel('Prob')
plt.xlabel('Part')
plt.legend(['Real', 'Fake'], loc='upper left')
plt.show()


print(all)
print(100.0*fake_numb/all)
print(100.0*real_numb/all)



