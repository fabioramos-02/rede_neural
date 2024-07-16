import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

# Carregando o conjunto de dados MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizando os dados
x_train, x_test = x_train / 255.0, x_test / 255.0

# Dividindo o conjunto de treinamento em treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Construindo o modelo com camadas convolucionais
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Expandindo as dimensões dos dados de entrada para (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Treinando o modelo com dados de validação
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Salvando o modelo
model.save('mnist_cnn_model.h5')

# Avaliando o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Carregar o modelo salvo (caso queira usar o modelo salvo)
# model = load_model('mnist_cnn_model.h5')

# Função para carregar e preprocessar uma imagem
def load_and_preprocess_image(filepath):
    img = image.load_img(filepath, color_mode="grayscale", target_size=(28, 28))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão para o batch
    return img_array

# Fornecer uma imagem para classificação
img_path = "path/images.png"  # Substitua pelo caminho da sua imagem
img = load_and_preprocess_image(img_path)
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
print(f'Predicted class: {predicted_class}')

# Visualizar a imagem fornecida
plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()
