#The finally block gets executed no matter if the try block raises any errors or not:
x=1
y=0
try:
  print(x/y)
except:
  print("Something went wrong")
finally:
  print("The 'try except' is finished")
#
# import csv
# from pathlib import Path
#
#
# my_file = Path('detection results.csv')
#
# file_exist=my_file.exists()
#
#
# with open('detection results.csv', 'a', encoding='UTF8') as f:
#     writer = csv.writer(f)
#     if not file_exist:
#         header = ['author name', 'book name', 'percent that written by author']
#         writer.writerow(header)
#     data = ['auth', 'test', 0.8]
#     writer.writerow(data)
#     writer.writerow(data)
