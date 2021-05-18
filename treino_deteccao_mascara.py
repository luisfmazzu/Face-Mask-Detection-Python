# Linha de comando para teste rápido:
# python treino_deteccao_mascara.py --data data

# Importa os pacotes necessarios para o projeto
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Argumentos ao executar o programa
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="Caminho para a pasta com dados")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="Caminho para a imagem com grafico dos resultados")
ap.add_argument("-m", "--model", type=str,
	default="modelo_deteccao_mascara.model",
	help="Caminho para o arquivo resultante do modelo")
args = vars(ap.parse_args())

# Inicializa a taxa de aprendizado, numero de epocas e tamanho do lote
INIT_TA = 1e-4
EPOCAS = 20
TL = 32

# Lista de imagens do dataset e inicializa variaveis
print("-- Carregando imagens...")
caminhoImagens = list(paths.list_images(args["data"]))
data = []
legendas = []

# Loop sobre o caminho das imagens
for caminhoImagem in caminhoImagens:
	# Extrai a classe da legenda da imagem
	legenda = caminhoImagem.split(os.path.sep)[-2]

	# Carrega a imagem de entrada (224x224) e preprocessa
	imagem = load_img(caminhoImagem, target_size=(224, 224))
	imagem = img_to_array(imagem)
	imagem = preprocess_input(imagem)

	# Atualiza dado da imagem e legenda
	data.append(imagem)
	legendas.append(legenda)

# Converte data e legendas para vetores numpy 
data = np.array(data, dtype="float32")
legendas = np.array(legendas)

# Aplica one-hot encoding nas legendas
lb = LabelBinarizer()
legendas = lb.fit_transform(legendas)
legendas = to_categorical(legendas)

# Particiona dados para utilizar 75% para treino e 25% para testes
(treinoX, testeX, treinoY, testeY) = train_test_split(data, legendas,
	test_size=0.20, stratify=legendas, random_state=42)

# Constroi gerador de treinamento de imagem
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Carrega rede MobileNetV2 e constroi headmodel
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Coloca headModel em cima do modelo base, que ira se tornar o modelo utilizado
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop sobre todas camadas do modelo base e as congela para nao haver atualizacao durante primeira etapa do treinamento
for camada in baseModel.layers:
	camada.trainable = False

# Compila o modelo
print("-- Compilando modelo...")
opt = Adam(lr=INIT_TA, decay=INIT_TA / EPOCAS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Treina o head model
print("-- Treinando head model..")
H = model.fit(
	aug.flow(treinoX, treinoY, batch_size=TL),
	steps_per_epoch=len(treinoX) // TL,
	validation_data=(testeX, testeY),
	validation_steps=len(testeX) // TL,
	epochs=EPOCAS)

# Realiza predicoes nas imagens de teste
print("-- Predicoes nas imagens de teste...")
predIdxs = model.predict(testeX, batch_size=TL)

# Para cada imagem de teste há necessidade de achar o index da legenda com maior probabilidade de predicao
predIdxs = np.argmax(predIdxs, axis=1)

# Mostra report
print(classification_report(testeY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Serializa modelo para o disco
print("-- Salvando modelo para o disco...")
model.save(args["model"], save_format="h5")

# Plot
N = EPOCAS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="treino_perda")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_perda")
plt.plot(np.arange(0, N), H.history["accuracy"], label="treino_prc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_prc")
plt.title("Treinamento - Perda e precisao")
plt.xlabel("Epoca #")
plt.ylabel("Perda/Precisao")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
