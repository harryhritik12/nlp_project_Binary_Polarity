from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_texts)

X_train_dl = tokenizer.texts_to_sequences(train_texts)
X_val_dl = tokenizer.texts_to_sequences(val_texts)
X_test_dl = tokenizer.texts_to_sequences(test_texts)

max_len = 100
X_train_dl = pad_sequences(X_train_dl, maxlen=max_len)
X_val_dl = pad_sequences(X_val_dl, maxlen=max_len)
X_test_dl = pad_sequences(X_test_dl, maxlen=max_len)

model = Sequential()
model.add(Dense(512, input_shape=(max_len,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_dl, train_labels, epochs=10, batch_size=128, validation_data=(X_val_dl, val_labels))

test_loss, test_acc = model.evaluate(X_test_dl, test_labels)
print('Test accuracy:', test_acc)
