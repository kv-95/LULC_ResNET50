# Deep CNN model using ResNET-50

# Import libraries

import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

class model():
    def __init__(self):
        self.batch_size = 16
        self.img_height = 180
        self.img_width = 180

# Load data
    def load_data(self):
        data_dir = "Data"
        data_dir = pathlib.Path(data_dir)

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split = 0.2,
            subset = "training",
            seed = 123,
            image_size = (self.img_height, self.img_width),
            batch_size = self.batch_size
        )

        self.validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset='validation',
            seed=123,
            image_size=(self.img_height,self.img_width),
            batch_size = self.batch_size
        )

        data_dir = None

# standardize and augmentation 
    def standard(self):
        self.autotune = tf.data.AUTOTUNE

        self.train_ds_1 = self.train_ds.cache().shuffle(1000).prefetch(buffer_size= self.autotune)
        self.validation_ds_1 = self.validation_ds.cache().prefetch(buffer_size= self.autotune)

        self.augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                            input_shape=(self.img_height, 
                                                                        self.img_width,
                                                                        3)),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            ]
            )
        
        self.autotune = None
    
    def model(self):

        lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

        num_classes = len(self.train_ds.class_names)

        self.model = tf.keras.models.Sequential()
        self.model.add(self.augmentation)
        self.model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
        self.model.add(tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling=None))
        
        self.model.add(tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.1))

        self.model.add(tf.keras.layers.Conv2D(128, 1, padding='same', activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(256, 1, padding="same", activation="relu"))
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Conv2D(512, 1, padding='same', activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        # self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Dense(num_classes))

        num_classes = None

    def model_compile(self):
        self.model.compile(optimizer='adam',
                            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

        self.epochs = 10

        self.history = self.model.fit(
            self.train_ds_1,
            validation_data = self.validation_ds_1,
            epochs = self.epochs
        )

    def classify(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        self.model.save("model/")
        acc = None  
        val_acc = None
        loss = None
        val_loss = None
        epochs_range = None

if __name__ == '__main__':
    model1 = model()

    model1.load_data()
    model1.standard()
    model1.model()
    model1.model_compile()
    model1.classify()

    model1 = None
