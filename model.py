import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, CategoricalAccuracy
from matplotlib import pyplot as plt

#img = cv2.imread(os.path.join('images','hello','Image_1725576331.5350685.jpg'))
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#plt.show(block = True)

data = tf.keras.utils.image_dataset_from_directory('images',label_mode='categorical')
data = data.map(lambda x,y: (x/255, y))

#data_iterator = data.as_numpy_iterator()
#batch = data_iterator.next()
#print(batch[0].shape)

#fig, ax = plt.subplots(ncols=4, figsize = (20,20))
#for idx, img in enumerate(batch[0][:4]):
#    ax[idx].imshow(img.astype(int))
#    ax[idx].title.set_text(batch[1][idx])
#plt.show(block = True)


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#print(len(test))


model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#print(model.summary())

logdir= 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=300, validation_data=val, callbacks=[tensorboard_callback])



#fig = plt.figure()
#plt.plot(hist.history['loss'], color='teal', label='loss')
#plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
#fig.suptitle('Loss', fontsize=20)
#plt.legend(loc="upper left")
#plt.show(block = True)

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
    
print(f'Precision: {pre.result().numpy()},Recall: {re.result().numpy()},Accuracy: {acc.result().numpy()}')


img = cv2.imread('yes.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255
img = np.expand_dims(img, axis=0)

#print(img.shape)

yhat = model.predict(img)
predicted_class = np.argmax(yhat, axis=1)
print(f'Predicted Class: {predicted_class[0]}')

model.save('Data/model.h5')
new_model = load_model('Data/model.keras')

yhat_new = new_model.predict(img)
predicted_class = np.argmax(yhat_new, axis=1)
print(f'Predicted Class2: {predicted_class[0]}')