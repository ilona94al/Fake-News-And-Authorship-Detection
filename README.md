# Fake-News-And-Plagiarism-Detection
This program was made as a final project in the software engineering degree. 

The project-based on a deep learning approach and performs two tasks:
1) Fake news detection
2) Plagiarism detection

### Program Illustration

https://user-images.githubusercontent.com/57364429/149624231-74ebd269-df8a-4c13-9da7-b9e4eaa46e9c.mp4

https://user-images.githubusercontent.com/57364429/149624237-d4f89f5a-ba25-4514-8cb0-c91126f2388c.mp4

### Used algorithms:
-  BERT for NLP
-  CNN

### Deep learning model architecture 
<img src="https://user-images.githubusercontent.com/57364429/149625170-2b72b8c4-a544-412c-93b7-fa9eb395ed4b.png" width="650" hight="400">

### Program architecture
<img src="https://user-images.githubusercontent.com/57364429/149635959-14ea9339-1d51-4ccf-a5f5-2a1fcb22940c.jpg"  width="650" hight="850">

## Requirements
● Python > = 3.6

● TensorFlow 2.6.0

● TensorFlow-hub 0.12.0

● PyQt5

**Run**: 

💡 pip install requirements.txt

💡 pip install -q -U tensorflow-text

💡 pip install -q tf-models-official

### BERT installation
●	Download BERT encoder from: https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3

●	Download BERT preprocessor from: https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3

●	Create in root directory a new folder "BERT" and inside create two folders "encoder" and "preprocessor".

●	Unzip and paste enсoder files into BERT/encoder directory.

● Unzip and paste preprocessor files into BERT/preprocessor directory.

## How to use
Run: mainWinController.py which is located in the "gui_controllers" package.

## Database
### Fake news
[Single csv file with column for label](https://www.kaggle.com/c/fake-news/data)

[Two csv files - one file for each label](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

<img width="350" alt="9 train_fake_news" src="https://user-images.githubusercontent.com/57364429/150197062-ae3c9a96-4e77-4ba3-85a2-efcd741ab5d3.png">
<img width="350" alt="10 train_fake_news2" src="https://user-images.githubusercontent.com/57364429/150197075-047ff546-05fb-4cba-aa4f-e86c4781fa44.png">

## Authors
* **Ilona Aliev** - [Ilona's github](https://github.com/ilona94al)
* **Vladislav Platunov** - [Vlad's github](https://github.com/Vladplat12)



