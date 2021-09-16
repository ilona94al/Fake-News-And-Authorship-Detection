import re
import numpy as np

def define_expected_classification(fake_num, all_n):
    expected_classification = np.empty(all_n, int)
    expected_classification[0:fake_num] = 0
    expected_classification[fake_num:all_n] = 1
    return  expected_classification

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
    with open('fake50.csv', 'r', encoding='utf-8') as file:
        fakeReader = csv.reader(file)

        for row in fakeReader:
            fake_news[i:] = row[1:2]
            i = i + 1
    i = 0
    with open('true50.csv', 'r', encoding='utf-8') as file:
        trueReader = csv.reader(file)
        for row in trueReader:
            real_news[i:] = row[1:2]
            i = i + 1
    print(fake_news)
    print(real_news)
    news = fake_news+real_news
    Y=define_expected_classification(len(fake_news), len(news))
    print(news)
if __name__ == '__main__':
    Text_processing('Lets go')
