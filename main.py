
def Text_processing(name):

    import csv
    with open('fake50.csv', 'r', encoding='utf-8') as file:
        fakeNews = csv.reader(file)
        for row in fakeNews:
            print(row[1:2])
    print("__________________________________________________________________________________________")
    with open('true50.csv', 'r', encoding='utf-8') as file:
        trueNews = csv.reader(file)
        for row in trueNews:
            print(row[1:2])

if __name__ == '__main__':
    Text_processing('Lets go')


