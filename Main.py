import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization


class Cifar:


     def test(self):

          dict = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse',
                  8: 'Ship', 9: 'Truck'}

          imageIndex = 1206

          (data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
          loadData = load_model('version1.h5')

          pre = loadData.predict(data_test[imageIndex].reshape(1, 32, 32, 3))

          print("index: ", pre.argmax())



          plt.imshow(data_test[imageIndex].reshape(32, 32, 3), interpolation='bicubic')
          for x in label_test[imageIndex]:
               print("Predicted: ", dict[pre.argmax()])
               print("Actually: ", dict[x])
               plt.title("Predicted:%s  Actually:%s" % (dict[pre.argmax()], dict[x]))
          plt.show()

     def createModel(self):
          (data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()

          model = Sequential()

          model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 3), activation='relu'))
          model.add(BatchNormalization())
          model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 3), activation='relu'))
          model.add(BatchNormalization())
          model.add(MaxPool2D(pool_size=(2, 2), strides=2))

          model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
          model.add(Conv2D(64, (3, 3), activation='relu'))
          model.add(MaxPool2D(pool_size=(2, 2), strides=2))
          model.add(BatchNormalization())

          model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
          model.add(BatchNormalization())



          model.add(Flatten())

          # Last layer, classifying the images
          model.add(Dense(units=10, activation='softmax'))

          model.summary()


          # Compile the model
          model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

          return model, data_train, label_train, data_test, label_test

     def train(self):

          model, data_train, label_train, data_test, label_test = self.createModel()

         # data augmentation, rotate image so the filter can see the image in different angles and different zoom levels so it can recognize it better
          datagen = ImageDataGenerator(
               rotation_range=90,
               width_shift_range=0.2,
               height_shift_range=0.2,
               horizontal_flip=True,
               vertical_flip=True
          )

          train_generator = datagen.flow(
               data_train,
               label_train,
               batch_size=10
          )

          #Train the model
          #Batch_size, how many samples shall you use before updating the parameters, epoch, numbers you will se the whole dataset, steps_per_epoch, number that is used to define an epoch.
         # model.fit_generator(train_generator, epochs=7, steps_per_epoch=len(data_train))

          model.fit(x=data_train, y=label_train, epochs=1, batch_size=128)
         # model.summary()
          score = model.evaluate(data_test, label_test)
          print("Test loss: ", score[0])
          print("Test accuracy: ", score[1])


          #Save model
          model.save('version1.h5')


c = Cifar()
#c.train()
c.test()