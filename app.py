# app.py - VERSÃO OTIMIZADA PARA PRECISÃO

from flask import Flask, render_template, request, jsonify 
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import face_recognition
import boto3
import os
import requests

# --- CONFIGURAÇÕES ---
# Lembre-se de usar credenciais válidas e temporárias do AWS Learner Lab!
AWS_ACCESS_KEY_ID = 'ASIA6ODU7GZ6COM6UN42'
AWS_SECRET_ACCESS_KEY = 'oDNNue8GCdxqMB6E+FOijBR7F3zLqhJWjAs67D10'
AWS_SESSION_TOKEN = 'IQoJb3JpZ2luX2VjEJn//////////wEaCXVzLXdlc3QtMiJHMEUCIQDJL8xdN5B0YgLuQW9lDKKKNhG0BRKfIWleaPVjZA/uaAIgUwg/ePyeshAYlSxXSCptG7WJmWmQpp2V1eiKlIzijjoquAIIwf//////////ARAAGgw5OTIzODI3NjA1NzIiDOf8n667vavrrV90oiqMAnydR2BGFlmZaJKqbaHh3dJrebWmRLmsFv/u/KDg8j9Q5S5WkGYvAdTdwz2DORIpBBnEs4oMTnnOoor77B9RxYvIiUXLz9Ak+2tcKxE33BvI/SF2TpFPcOIPAOC5Q7tGamsNrZCefj/frwU5P9LysvkTvsAzq+A/8ae9GHJnZ3N3Js3hB5IeD7Ka4W12JYlsdMllcHr63Vdb1qA5duOsRo7/uQYw6WG7+fDtOm8ZpvtRpL2ZMZBL5mB5PpVWzOxRufgt+rH+09aXPAIucw2IpApyAupJ0JFxNhqm8cv89TWxC0hye2jh0Pa3xi8aQOTf8k/+mkBq8kVupWoNjVSgEHNoxn4pIughVUKGp28wiY2pxAY6nQEZA2rnwldUs9AEY+3tNnM4/cLoPzRmJoW8G/rAMu50Bcjt72mIJ1IxRrq99hahfTPvFyDXmT2J7uUFS6qUQH05Y6+oTO5IQ4iR9gfDnwM7wuCPX5ZaVtA+9y5olDxGE+Rv+VYJZREmryAeybRQGYoMq8ZINTdDZpuzkls/35FrXS9rBFxVxelXYVDk/ULFn1FRRuKa7bK0O6Ud4sdu'

AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'visaocomputacional-senai'

# Constante para o limite de distância (área mínima do rosto em pixels).
# Este valor é experimental! Você pode precisar ajustá-lo para a sua webcam.
MIN_FACE_AREA_THRESHOLD = 20000 

# --- VARIÁVEIS GLOBAIS ---
known_face_encodings = []
known_face_names = []
recognized_person_set = set()

# --- FUNÇÕES DE LÓGICA ---

def load_known_faces():
    """Carrega as assinaturas faciais do S3 para a memória."""
    global known_face_encodings, known_face_names
    print("➡️  Carregando rostos conhecidos do S3...")
    try:
        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                 aws_session_token=AWS_SESSION_TOKEN,
                                 region_name=AWS_REGION)
        main_folder_prefix = 'known_faces/'
        paginator = s3_client.get_paginator('list_objects_v2')
        person_folders = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=main_folder_prefix, Delimiter='/')
        
        for page in person_folders:
            for prefix in page.get('CommonPrefixes', []):
                person_folder_prefix = prefix.get('Prefix')
                person_name = person_folder_prefix.replace(main_folder_prefix, '').strip('/')
                person_encodings = []
                image_files = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=person_folder_prefix)
                for image_obj_summary in image_files.get('Contents', []):
                    s3_key = image_obj_summary['Key']
                    if not s3_key.endswith('/'):
                        image_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                        image_content = image_obj['Body'].read()
                        image = face_recognition.load_image_file(BytesIO(image_content))
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            person_encodings.append(encodings[0])
                if person_encodings:
                    average_encoding = np.mean(person_encodings, axis=0)
                    known_face_encodings.append(average_encoding)
                    known_face_names.append(person_name)
        
        if not known_face_names:
            print("⚠️ AVISO: Nenhuma face encontrada no S3. O reconhecimento não funcionará.")
        else:
            print(f"✅ {len(known_face_names)} pessoas carregadas: {', '.join(known_face_names)}")
            
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao carregar faces do S3: {e}")
        print("   Verifique suas credenciais, nome do bucket e permissões.")


### FUNÇÃO DE RECONHECIMENTO TOTALMENTE REFEITA PARA PRECISÃO ###
def process_frame(image_data_url):
    """Processa um frame para reconhecer o rosto mais próximo com alta precisão."""
    header, encoded = image_data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    frame = np.array(Image.open(BytesIO(image_data)))
    rgb_frame = frame[:, :, ::-1]

    # 1. Detecta todos os rostos na imagem
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    if not face_locations:
        return "Desconhecido"

    # 2. Encontra o maior rosto (o mais próximo da câmera)
    largest_face_location = None
    max_area = 0
    for (top, right, bottom, left) in face_locations:
        area = (bottom - top) * (right - left)
        if area > max_area:
            max_area = area
            largest_face_location = (top, right, bottom, left)

    # 3. Verifica se o rosto está perto o suficiente
    if max_area < MIN_FACE_AREA_THRESHOLD:
        return "Aproxime-se"

    # 4. Gera a assinatura facial APENAS para o maior rosto
    face_encodings = face_recognition.face_encodings(rgb_frame, [largest_face_location])
    name = "Desconhecido"
    
    if face_encodings:
        face_encoding = face_encodings[0]
        
        # 5. COMPARAÇÃO RIGOROSA: Usa a tolerância de 0.5 para evitar falsos positivos
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
    
    return name

# --- APLICAÇÃO WEB (FLASK) ---
# O resto do código Flask/SocketIO permanece o mesmo

app = Flask(__name__)
app.config['SECRET_KEY'] = 'um-segredo-muito-secreto!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list')
def list_page():
    current_names = sorted(list(recognized_person_set))
    return render_template('list.html', names=current_names)

@socketio.on('image')
def handle_image(image_data_url):
    recognized_name = process_frame(image_data_url)
    socketio.emit('response', {'name': recognized_name})
    
    if recognized_name not in ["Desconhecido", "Aproxime-se"] and recognized_name not in recognized_person_set:
        print(f"✔️ Nova pessoa adicionada à lista: {recognized_name}")
        recognized_person_set.add(recognized_name)
        sorted_list = sorted(list(recognized_person_set))
        socketio.emit('update_list', {'names': sorted_list})

@app.route('/get-speech', methods=['POST'])
def get_speech():
    text_to_speak = request.json.get('text')
    if not text_to_speak:
        return jsonify({'error': 'No text provided'}), 400
    try:
        url = "https://translate.google.com/translate_tts"
        params = {'ie': 'UTF-8', 'q': text_to_speak, 'tl': 'pt-BR', 'client': 'tw-ob'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        return jsonify({'audio': audio_base64})
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao buscar áudio do Google: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_known_faces()
    print("🚀 Servidor pronto!")
    print("   - Página da Webcam: http://127.0.0.1:5000")
    print("   - Página da Lista:  http://127.0.0.1:5000/list")
    socketio.run(app, host='0.0.0.0', port=5000)
