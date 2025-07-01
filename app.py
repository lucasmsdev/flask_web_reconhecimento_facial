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

AWS_ACCESS_KEY_ID = 'ASIA6ODU7GZ6G7LNWEHD'
AWS_SECRET_ACCESS_KEY = 'JOWT1oKq5xrwVQkYoR/PCwQL8j2/7MvFQC9GB0EN       '
AWS_SESSION_TOKEN = 'IQoJb3JpZ2luX2VjEMP//////////wEaCXVzLXdlc3QtMiJHMEUCIQCwBzfKTS0jeT5P+JiwhBzT2NvdKEtQkRcXzmYFmseClAIgE3z6VEj/ASkWtCCBBoSGXejrHLGvF36ih7nx/8x6tC8quAIIvP//////////ARAAGgw5OTIzODI3NjA1NzIiDGaVmezjMXTAQDLkLCqMAjdswJFaEthfo3t/lWK57VO/mCr/iyNGxAtxgNwfJEnEfPHkjpCln/jAhH53WJiISLxCXgPpkxNje09Ul0wi8+5FnGofBwPrcPXiVB/V/JNb7h+d2tONnbd9Oa6G/J15tt3Ddj9r7SvDZxWgu5KF0nly7CwDdSD7ahrpWdA26Pb3AV26MWFR5W80LqjJ/rW+wNJuC9kT3Q/ujc7VUtXh+dRycT14z2vfENxR1rCl/OG5yfA1HrTo+0CleIdPWlMB77qbL9GgVyYERCC83CTqQWwCs50tvj4Z3/Yga+pTB2YuA047Ha0t/dhbPF+QcDDmErCVEzUJEYTDwuNB2A7gVWQ5rlb/eD/xFHypYVkwn9aJwwY6nQEa7C2Dudf10+cbV3moCo72XBfF5+SHsx6EsOKJzCJ/XTNipSliP/KIkt8uAsxt4mR7sBmb1f0jt8SVHMXvV60bB+T7N/GBwZXMHgguIoKCCIMwZLZaekLKvAhUUyM83Z+wNbfGM2upM3pzQqJI0Le0CTXrY6YofBqfCVtvZLQMd8aHbDTD2to5Di63PFM8NU7lwAOcHY4g3OAMwbAd'
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'visaocomputacional-senai'

known_face_encodings = []
known_face_names = []

def load_known_faces():
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
    # ... (o código da função process_frame continua exatamente o mesmo)
    header, encoded = image_data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data))
    frame = np.array(image)
    rgb_frame = frame
    
    face_locations = face_recognition.face_locations(rgb_frame, model="hog") # 'hog' é mais rápido
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

@socketio.on('image')
def handle_image(image_data_url):
    recognized_name = process_frame(image_data_url)
    # Agora, em vez de apenas enviar o nome, preparamos para a narração
    socketio.emit('response', {'name': recognized_name})

### --- NOVO ENDPOINT PARA GERAR A FALA --- ###
@app.route('/get-speech', methods=['POST'])
def get_speech():
    """Recebe um texto, busca o áudio no Google TTS e retorna como base64."""
    text_to_speak = request.json.get('text')
    if not text_to_speak:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # URL da API não-oficial do Google Translate TTS
        url = "https://translate.google.com/translate_tts"
        params = {
            'ie': 'UTF-8',
            'q': text_to_speak,
            'tl': 'pt-BR',
            'client': 'tw-ob' # Parâmetro necessário para a API funcionar
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status() # Lança um erro se a requisição falhar
        
        # Codifica o conteúdo do áudio (bytes) em base64 (string)
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        
        return jsonify({'audio': audio_base64})
        
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar áudio do Google: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_known_faces()
    print("🚀 Servidor pronto! Acesse http://127.0.0.1:5000 no seu navegador.")
    socketio.run(app, host='0.0.0.0', port=5000)