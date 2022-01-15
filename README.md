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
<img src="https://user-images.githubusercontent.com/57364429/149625170-2b72b8c4-a544-412c-93b7-fa9eb395ed4b.png" width="700" hight="500">
<img src="https://user-images.githubusercontent.com/57364429/149635335-92ccc12b-317a-4e6f-893f-303cec26b2db.png" width="500" hight="800">
<img src="https://user-images.githubusercontent.com/57364429/149625922-86692c07-2726-43a2-bf40-0ad10cab7e28.png" width="500" hight="300">

### Program architecture
<img src="https://user-images.githubusercontent.com/57364429/149635621-49915628-11f5-40e9-bdab-d4d0ac78251f.jpg" width="700" hight="1000">

## Requirements
Run: pip install requirements.txt

💡 if TensorFlow-text installation was failed try: pip install -q -U tensorflow-text

💡 if tf-models-official installation was failed try: pip install -q tf-models-official

### BERT installation
●	Download BERT encoder from: https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3

●	Download BERT preprocessor from: https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3

●	Create in root directory a new folder "BERT" and inside create two folders "encoder" and "preprocessor".

●	Unzip and paste enсoder files into BERT/encoder directory.

●   Unzip and paste preprocessor files into BERT/preprocessor directory.

## How to use
Run: mainWinController.py which is located in the "gui_controllers" package.



