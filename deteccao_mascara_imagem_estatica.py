# Linha de comando para teste rápido:
# python deteccao_mascara_imagem_estatica.py --imagem exemplos/exemplo_01.png

# Importa os pacotes necessarios para o projeto
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# Argumentos ao executar o programa
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagem", required=True,
	help="Caminho para a imagem de entrada")
ap.add_argument("-f", "--face", type=str,
	default="detector_face",
	help="Caminho para o modelo de deteccao de face")
ap.add_argument("-c", "--confianca", type=float, default=0.5,
	help="Probabilidade minima (confianca) para filtrar deteccoes")
args = vars(ap.parse_args())

# Carrega o modelo de deteccao de face
print("-- Detectando modelo de deteccao de face...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carrega a imagem estatica e capta as suas dimensoes altura (a) e largura (l)
imagem = cv2.imread(args["imagem"])
orig = imagem.copy()
(a, l) = imagem.shape[:2]

# Constroi um blob 300x300 da imagem
blob = cv2.dnn.blobFromImage(imagem, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# Passa o blob através do modelo da rede
print("-- Detectando faces...")
net.setInput(blob)
deteccoes = net.forward()

# Loop entre as deteccoes obtidas
for i in range(0, deteccoes.shape[2]):
	# Extrai a confianca das deteccoes
	confianca = deteccoes[0, 0, i, 2]

	# Filtra deteccoes fracas atraves do nivel de confianca estabelecido
	if confianca > args["confianca"]:
		# Computa coordenadas (x, y) do bounding box a ser mostrado
		box = deteccoes[0, 0, i, 3:7] * np.array([l, a, l, a])
		(inicioX, inicioY, fimX, fimY) = box.astype("int")

		# Bounding box tem que estar no limite da imagem
		(inicioX, inicioY) = (max(0, inicioX), max(0, inicioY))
		(fimX, fimY) = (min(l - 1, fimX), min(a - 1, fimY))

		# Extrai ROI da face e converte para RGB e muda para tamanho 224x224
		face = imagem[inicioY:fimY, inicioX:fimX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		cor = (0, 255, 0)

		cv2.rectangle(imagem, (inicioX, inicioY), (fimX, fimY), cor, 2)

# Imagem de said
cv2.imshow("Imagem de saida", imagem)
cv2.waitKey(0)
