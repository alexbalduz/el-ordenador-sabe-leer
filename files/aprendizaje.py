#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 13 - El ordenador saber leer
#
# Modulos necesarios:
#   TENSORFLOW 1.13.1
#   KERAS 2.2.4
#   OPENCV 3.4.5.20
#   PYTTSX3 2.7.1
#   SCIKIT-LEARN 0.21.1
#   NUMPY 1.16.3
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------


from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from mnist import MNIST

class Aprendizaje():

    def __init__(self, largoImagen, anchoImagen, cantidadImagenes, cantidadEtiquetas, cantidadCaracteres, epochs, batch_size):
        self.largoImagen = largoImagen
        self.anchoImagen = anchoImagen
        self.cantidadImagenes = cantidadImagenes
        self.cantidadEtiquetas = cantidadEtiquetas
        self.cantidadCaracteres = cantidadCaracteres
        self.epochs = epochs
        self.batch_size = batch_size


    #Carga de las imágenes
    def cargaImagenes(self):
        emnist_data = MNIST(path='datas\\', return_type='numpy')
        emnist_data.select_emnist('letters')
        Imagenes, Etiquetas = emnist_data.load_training()

        return Imagenes, Etiquetas

    #Conversión de las imágenes y etiquetas en tabla numpy
    def conversionImagenes(self, Imagenes, Etiquetas):

        Imagenes = np.asarray(Imagenes)
        Etiquetas = np.asarray(Etiquetas)

        return Imagenes, Etiquetas


#Transformación de las tablas de imágenes, para que sean de 28*28
    def transformacionImagenes(self, Imagenes, Etiquetas):

        Imagenes = Imagenes.reshape(self.cantidadImagenes, self.anchoImagen, self.largoImagen)
        Etiquetas= Etiquetas.reshape(self.cantidadEtiquetas, 1)

        return Imagenes, Etiquetas

    #Visualización de la imagen N.° 70000
    def visualizacionImagen(self, Imagenes, Etiquetas):

        plt.imshow(Imagenes[70000], cmap='gray')
        plt.title('Etiqueta: '+str(Etiquetas[70000]))
        plt.show()

        Etiquetas = Etiquetas-1
        print("Etiqueta de la imagen N.° 70000...")
        print(Etiquetas[70000])


class Validar():

    def __init__(self, Imagenes, Etiquetas, epochs, batch_size):
        self.imagenes_aprendizaje, self.imagenes_validacion, self.etiquetas_aprendizaje, self.etiquetas_validacion = train_test_split(Imagenes, Etiquetas, test_size=0.25, random_state=42)
        self.epochs = epochs
        self.batch_size = batch_size

    def validacion(self, anchoimagen, largoimagen, cantidad_de_clases):
        self.imagenes_aprendizaje = self.imagenes_aprendizaje.reshape(self.imagenes_aprendizaje.shape[0], anchoimagen, largoimagen, 1)
        self.imagenes_validacion = self.imagenes_validacion.reshape(self.imagenes_validacion.shape[0], anchoimagen, largoimagen, 1)

        self.imagenes_aprendizaje = self.imagenes_aprendizaje.astype('float32')
        self.imagenes_validacion = self.imagenes_validacion.astype('float32')
        self.imagenes_aprendizaje /= 255
        self.imagenes_validacion /= 255

        self.etiquetas_aprendizaje = keras.utils.to_categorical(self.etiquetas_aprendizaje, cantidad_de_clases)
        self.etiquetas_validacion = keras.utils.to_categorical(self.etiquetas_validacion, cantidad_de_clases)

        return self.imagenes_aprendizaje, self.imagenes_validacion, self.etiquetas_aprendizaje, self.etiquetas_validacion

class red_neuronal():
    def __init__(self):
        self.redCNN = Sequential()

    def red(self, anchoimagen, largoimagen, cantidad_de_clases):
        self.redCNN.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(anchoimagen, largoimagen, 1)))
        self.redCNN.add(Conv2D(64, (3, 3), activation='relu'))
        self.redCNN.add(MaxPooling2D(pool_size=(2, 2)))
        self.redCNN.add(Dropout(0.25))
        self.redCNN.add(Flatten())
        self.redCNN.add(Dense(128, activation='relu'))
        self.redCNN.add(Dropout(0.5))
        self.redCNN.add(Dense(cantidad_de_clases, activation='softmax'))

        return self.redCNN

    def compilacion(self, redCNN):
        redCNN.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

        return redCNN

    def training(self, redCNN, imagenes_aprendizaje, etiquetas_aprendizaje, imagenes_validacion, etiquetas_validacion, epochs, batch_size):
        redCNN.fit(imagenes_aprendizaje, etiquetas_aprendizaje,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(imagenes_validacion, etiquetas_validacion))

        return redCNN

    def save(self, redCNN):
        redCNN.save('modelo/modelo.h5')


    def evaluate(self, redCNN, imagenes_validacion, etiquetas_validacion):
        score = redCNN.evaluate(imagenes_validacion, etiquetas_validacion, verbose=0)
        print('Pérdida:', score[0])
        print('Precisión:', score[1])


def main():
    aprendizaje = Aprendizaje(28, 28, 124800, 124800, 26, 12, 128)

    Imagenes, Etiquetas = aprendizaje.cargaImagenes()

    print("Cantidad de imágenes ="+str(len(Imagenes)))
    print("Cantidad de etiquetas ="+str(len(Etiquetas)))

    Imagenes, Etiquetas = aprendizaje.conversionImagenes(Imagenes, Etiquetas)

    print("Transformación de las tablas de imágenes...")
    Imagenes, Etiquetas = aprendizaje.transformacionImagenes(Imagenes, Etiquetas)

    print("Visualización de la imagen N.° 70000...")
    aprendizaje.visualizacionImagen(Imagenes, Etiquetas)

    validacion = Validar(Imagenes, Etiquetas, 12, 128)

    imagenes_aprendizaje, imagenes_validacion, etiquetas_aprendizaje, etiquetas_validacion = validacion.validacion(28, 28, 26)

    #Creación de la red neuronal
    red = red_neuronal()
    #Establecimiento de la red neuronal
    redCNN = red.red(28, 28, 26)
    #Compilación de la red neuronal
    redCNN = red.compilacion(redCNN)
    #Entrenamiento de la red neuronal
    redCNN = red.training(redCNN, imagenes_aprendizaje, etiquetas_aprendizaje, imagenes_validacion, etiquetas_validacion, 12, 128)
    red.save(redCNN)
    red.evaluate(redCNN, imagenes_validacion, etiquetas_validacion)
