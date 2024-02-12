import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv(r"C:\Users\SRUTHI\OneDrive\Desktop\Exam\test (1).csv",sep="\t",names=['label','message'])
test = pd.read_csv(r"C:\Users\SRUTHI\OneDrive\Desktop\Exam\train (3).csv",sep="\t",names=['label','message'])
df = pd.concat([train, test], ignore_index=True)


df['label'] = df['label'].map( {'positive': 2, 'neutral': 1,'negative':0} )
X= df["message"].values
y= df["label"].values

(X_train,X_test,y_train,y_test) = train_test_split(X,y,test_size=0.7,random_state=42)

tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
encoded_train = tokeniser.texts_to_sequences(X_train)
encoded_test = tokeniser.texts_to_sequences(X_test)

max_length = 10
padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')

# Model building using word embedding words are converted to vectors
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(100, embedding_vecor_length, input_length=max_length))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=3, batch_size=64)
accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (accuracy[1]*100))