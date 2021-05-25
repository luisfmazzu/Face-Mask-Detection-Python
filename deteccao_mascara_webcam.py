# Linha de comando para teste rápido:
# python deteccao_mascara_webcam.py

# Importa os pacotes necessarios para o projeto
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import timeit

def deteccao_mascara_webcam(frame, faceNet, mascaraNet):
	# Pega as dimensoes do blob e redimensiona
	(a, l) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Passa o blob através do modelo da rede
	faceNet.setInput(blob)
	deteccoes = faceNet.forward()

	# Inicializa lista de faces, localizacoes e predicoes
	faces = []
	locs = []
	preds = []

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
			face = frame[inicioY:fimY, inicioX:fimX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# listas
			faces.append(face)
			locs.append((inicioX, inicioY, fimX, fimY))

	# Realiza predicoes somente se foram encontradas faces
	if len(faces) > 0:
		# Para uma inferencia mais rapida, realiza-se um lote de predicoes em todas
		# faces ao mesmo tempo
		faces = np.array(faces, dtype="float32")
		preds = mascaraNet.predict(faces, batch_size=32)

	# Retorna localizacoes e predicoes
	return (locs, preds)

# Argumentos ao executar o programa
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="detector_face",
	help="Caminho para o modelo de deteccao de face")
ap.add_argument("-c", "--confianca", type=float, default=0.5,
	help="Probabilidade minima (confianca) para filtrar deteccoes")
ap.add_argument("-m", "--model", type=str,
	default="modelo_deteccao_mascara.model",
	help="Caminho para o arquivo resultante do modelo")
args = vars(ap.parse_args())

# Carrega o modelo de deteccao de face
print("-- Detectando modelo de deteccao de face...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carrega o modelo de deteccao de mascara
print("-- Detectando modelo de deteccao de mascara...")
mascaraNet = load_model(args["model"])

# Initicializa a webcam
print("-- Inicializando webcam...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

counter = 0;
buffer = 0;

# Loop entre os frames da webcam
while True:
	start = timeit.timeit()
	# Pega o frame da webcam e redimensiona para 400x400
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Detecta as faces no frame e determina as predicoes
	(locs, preds) = deteccao_mascara_webcam(frame, faceNet, mascaraNet)

	# Loop entre as localizacoes das faces detectadas
	for (box, pred) in zip(locs, preds):
		# Pega as dimensoes obtidas da deteccao de faces
		(inicioX, inicioY, fimX, fimY) = box
		(comMascara, semMascara) = pred
		
		# Determina a classe da legenda
		legenda = "Com mascara" if comMascara > semMascara else "Sem mascara"
		cor = (0, 255, 0) if legenda == "Com mascara" else (0, 0, 255)

		# Inclui a probabilidade na legenda
		legenda = "{}: {:.2f}%".format(legenda, max(comMascara, semMascara) * 100)

		# Mostra a legenda na imagem
		cv2.putText(frame, legenda, (inicioX, inicioY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 2)

		# Forma um retangulo no frame com as dimensoes
		cv2.rectangle(frame, (inicioX, inicioY), (fimX, fimY), cor, 2)

	# Imagem de saida
	cv2.imshow("Frame", frame)
	
	end = timeit.timeit()
	buffer += end - start
	counter += 1
	if(counter == 1000):
		print("-- Tempo medio de execucao")
		print(buffer / 1000)
	
	key = cv2.waitKey(1) & 0xFF

	# Apertar 'q' para sair do loop
	if key == ord("q"):
		break

# Destruicao de objetos
cv2.destroyAllWindows()
vs.stop()
