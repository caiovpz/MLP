import tensorflow as tf
import pandas as pd

data = pd.read_csv('samples.csv')

X = data[['in1', 'in2']]
y = data[['out']]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

adam = tf.keras.optimizers.Adam(lr=0.3)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mae', 'acc'])

model.fit(X, y, batch_size=100, epochs=100)

predictions = model.predict(X)
print(predictions)

score = model.evaluate(X, y, verbose=0)
print(score[2])

