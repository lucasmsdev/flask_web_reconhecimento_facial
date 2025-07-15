# app.py MODIFICADO

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

AWS_ACCESS_KEY_ID = 'ASIA6ODU7GZ6GMNKSVMY' # Lembre-se de usar credenciais válidas
AWS_SECRET_ACCESS_KEY = '4VoAHlTV74bQ0EOQK04HC1K5zifoR7ebWYzmQyE3'
AWS_SESSION_TOKEN = 'IQoJb3JpZ2luX2VjEC0aCXVzLXdlc3QtMiJHMEUCIQDfpkjMl1eg0taKMrQMYBjlxVyKCY2HVlQ0JeF1ATApQQIgXRVXywHQNeL/2PkpTeL498/j1pVXRR6wLraA1ZyOK+0qrwIIRhAAGgw5OTIzODI3NjA1NzIiDDJGVMDKUfia4xDPPCqMAn/r5+n+8A0z/DE5X9vdVcT+dG9G0wpj83WN9FTIagzxXZfUb2Rf28J7S3euN2zffNSzQAf6T34nNc5THVWVqHITKsNyxvRwuJ9Lnc6bYHC2ZPxoReZ9ssVoQE2wxHCUDiwRGRqGw4QLKE94ev59zpqOCtkO4rdehkeWkyCjt40mM/9sUC4PiJGVgLpLXUzK4K8Kf4l1CUUjSnimcFHbptV1xTVXJ/sPiFQSh6hzmZkEVwtejWyflNz9e15dajHiVOCwFFH8CmWJ1pnw0oZ6YNgz61+7ZNzWQ/q4RMqqTC2cF315utD6w+PGSsUhb3xpxnGRpW3kkjuk1zUIQGMXNIPwNj7fOp8uHm1OVl4w1pfZwwY6nQEgFyFXDDi6fKF7c28gzHDc6rAk8lk/kAwFSEdqaSDcXdO6wdRbnRraRWaJUZZPb2FPFTSvjYI8hmq7Z4I4L+gnQ5hn4AxMuhHXOTMWWhMa+YmQrrxrE/9CQ/W0UGZ1QQBL33k6O81shnYDQV0QqWehHzRgRchQ4B90WI+Cej00O6WxhWpmtBQbtnHINhzeGrJBn5LoEgBc2xJ66Xw1'
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'visaocomputacional-senai'

known_face_encodings = []
known_face_names = []

### NOVO: Conjunto para armazenar nomes únicos de pessoas já reconhecidas ###
recognized_person_set = set()

def load_known_faces():
    # ... (esta função continua exatamente a mesma, não precisa mudar)
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
                print(f"Processando pasta de: {person_name}")
                
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
                    print(f"Assinatura média para {person_name} criada.")
        
        if not known_face_names:
            print("Nenhuma face encontrada. O reconhecimento não funcionará.")
        else:
            print(f"{len(known_face_names)} pessoas carregadas: {', '.join(known_face_names)}")
    except Exception as e:
        print(f"ERRO ao carregar faces do S3: {e}")


def process_frame(image_data_url):
    # ... (esta função continua exatamente a mesma, não precisa mudar)
    header, encoded = image_data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data))
    frame = np.array(image)
    rgb_frame = frame
    
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    name = "Desconhecido"
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                break
    return name

# --- APLICAÇÃO WEB (FLASK) ---

app = Flask(__name__)
app.config['SECRET_KEY'] = 'um-segredo-muito-secreto!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

### NOVO: Rota para a página que exibirá a lista de nomes ###
@app.route('/list')
def list_page():
    # Passa a lista atual de nomes (já ordenada) para o template
    # Assim, a página já carrega com os nomes que foram reconhecidos até o momento.
    current_names = sorted(list(recognized_person_set))
    return render_template('list.html', names=current_names)

@socketio.on('image')
def handle_image(image_data_url):
    recognized_name = process_frame(image_data_url)
    socketio.emit('response', {'name': recognized_name})
    
    ### NOVO: Lógica para atualizar e transmitir a lista de nomes únicos ###
    # Verifica se o nome é válido e se ainda não foi adicionado à lista
    if recognized_name != "Desconhecido" and recognized_name not in recognized_person_set:
        print(f"✔️  Nova pessoa adicionada à lista: {recognized_name}")
        recognized_person_set.add(recognized_name)
        
        # Converte o set para uma lista ordenada
        sorted_list = sorted(list(recognized_person_set))
        
        # Emite um evento para TODOS os clientes com a lista atualizada
        # Qualquer página (como a list.html) que estiver escutando este evento, irá se atualizar.
        socketio.emit('update_list', {'names': sorted_list})


@app.route('/get-speech', methods=['POST'])
def get_speech():
    # ... (esta função continua exatamente a mesma, não precisa mudar)
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
        print(f"Erro ao buscar áudio do Google: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_known_faces()
    print("🚀 Servidor pronto!")
    print("   - Página da Webcam: http://127.0.0.1:5000")
    print("   - Página da Lista:  http://127.0.0.1:5000/list")
    socketio.run(app, host='0.0.0.0', port=5000)