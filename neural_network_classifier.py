import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from plotify import Plotify
from feature_extraction import extract_features
from import_data import load_pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

def magic(df, filename):
  #df = pd.read_pickle('word_2_grams.pkl')

  X = df['text'].tolist()
  y = df['author_one_hot'].tolist()

  plotify = Plotify()


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  # define model
  model = Sequential()
  model.add(Dense(2048, input_dim=len(X_train[0]), activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1024, input_dim=2048, activation='relu',kernel_initializer='he_uniform'))
  model.add(Dense(1024, input_dim=2048, activation='relu',kernel_initializer='he_uniform'))

  model.add(Dense(19, activation='softmax'))
  opt = SGD(lr=0.01, momentum=0.9)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

  history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

  # evaluate the model
  _, train_acc = model.evaluate(X_train, y_train, verbose=0)
  _, test_acc = model.evaluate(X_test, y_test, verbose=0)
  print(filename + ' Train: %.3f, Test: %.3f' % (train_acc, test_acc))
  # plot loss during training
  plt.subplot(211)
  plt.title('Loss')
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  # plot accuracy during training
  plt.subplot(212)
  plt.title('Accuracy')

  plt.plot(history.history['accuracy'], label='train')
  plt.plot(history.history['val_accuracy'], label='test')
  plt.legend()
  plt.savefig(('plots/' + filename), dpi=160)
  plt.tight_layout()
  plt.show()
