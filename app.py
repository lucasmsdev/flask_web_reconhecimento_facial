# app.py - VERSÃO COM ÁUDIO DO S3

from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition
import boto3
from botocore.exceptions import ClientError # ### NOVO ### Para tratar erros do S3

# --- CONFIGURAÇÕES ---
# ⚠️ Lembre-se de substituir com suas credenciais válidas e temporárias do AWS Learner Lab!
AWS_ACCESS_KEY_ID = 'ASIA6ODU7GZ6EUS7IZEE'
AWS_SECRET_ACCESS_KEY = 'WeJAaV6J5yJT9vBMkD6K4H9DN8Oz4tCEUW1W5krM'
AWS_SESSION_TOKEN = 'IQoJb3JpZ2luX2VjEMT//////////wEaCXVzLXdlc3QtMiJHMEUCIQCDfG7ceKMWfSZ9cAZrMfv1wrKADM6wVHhiWXYJQB2pxwIgUNrD0wGtaCz8ak1w085p0jKwopsaGZtkQlXtVNAc0MQquAII7f//////////ARAAGgw5OTIzODI3NjA1NzIiDFHPyeENJmdK9nja8iqMAioM5iaQhNRUTd05fJrcf04DTvTkqJ0Q7x/MdA1OtoGMi8oJOYTZcDsdpXOu78JhTkMz9HavpMdUboYO/QE/Zqw/0QaHouk8DATgXxKJ/bTzpe00LQfpmd4nOaXDEUWRZ7Lst64OgQr9PJF8itD0gYeBte3OUFAbv8EWtDXvw3gYwFwrc5o5GJfUsIQ1HV12A3d8cyZTMu4EjZTGUSl6MHXpWAxBZsBbhjUX0Obd5e+rm4+W0U/SYCk/ghjwb5fFbVZWnkqgX/LcTo8cdthvF6CA/pG3otspwylDh0Y0Max2OBOUXzkPU0zCmvhP0/HQfSO1D+cTA2Yc/5fm9gdaQKjj79MQraiU91holnswndWyxAY6nQFAtuOH0XwRV5wju/xjVc9ccFXkVdfHJDJiX2OUrUfWi4S//tr5E/0VblLjfRU4ZvzV4SimIJHNEMTWKRVVpe8ZNWtWr/CtL9UUJWYh6yQ9jXwNywxJminnJHAzdgnR3aXGiwzUFIaMNa1uVP2LKq6MZRkL8Z2ClP+mgD4GcBqkCaOC6oJGJW/+kWuJ5StwR3/c3qpfdGbpwP2B8IGz'
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'visaocomputacional-senai'
MIN_FACE_AREA_THRESHOLD = 20000 

# --- VARIÁVEIS GLOBAIS E CLIENTE S3 ---
known_face_encodings = []
known_face_names = []
recognized_person_set = set()

# ### NOVO ### Instancia o cliente S3 fora das funções para reutilização
s3_client = boto3.client('s3', 
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                         aws_session_token=AWS_SESSION_TOKEN,
                         region_name=AWS_REGION)

# --- FUNÇÕES DE LÓGICA ---

def load_known_faces():
    """Carrega as assinaturas faciais do S3 para a memória."""
    global known_face_encodings, known_face_names
    print("➡️  Carregando rostos conhecidos do S3...")
    try:
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
                    # ### MODIFICADO ### Processa apenas arquivos de imagem, ignorando o áudio
                    if s3_key.lower().endswith(('.png', '.jpg', '.jpeg')):
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

# ### VERSÃO 3: FUNÇÃO AINDA MAIS ROBUSTA CONTRA O TypeError ###
def process_frame(image_data_url):
    """Processa um frame para reconhecer o rosto mais próximo com alta precisão."""
    header, encoded = image_data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    
    try:
        pil_image = Image.open(BytesIO(image_data))
        # Converte a imagem para o formato que a biblioteca face_recognition espera (RGB)
        frame = np.array(pil_image.convert('RGB'))
    except Exception as e:
        print(f"Erro ao converter a imagem: {e}")
        return "Erro de Imagem"

    # 1. Detecta todos os rostos na imagem
    # Usando o modelo 'cnn' pode ser mais lento, mas é mais preciso. Se ficar lento, volte para "hog".
    face_locations = face_recognition.face_locations(frame, model="hog")
    if not face_locations:
        return "Desconhecido"

    # 2. Gera as assinaturas faciais para todos os rostos encontrados.
    # Esta é a chamada que está causando o erro, vamos mantê-la simples.
    # Se esta linha continuar a falhar, o problema está no ambiente/dependências.
    try:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    except TypeError as e:
        # Se o erro acontecer aqui, saberemos que é um problema fundamental de chamada.
        print(f"!!! ERRO CRÍTICO no face_encodings: {e}")
        print("!!! O problema provavelmente está nas versões das bibliotecas dlib/face_recognition.")
        return "Erro Interno"
        
    if not face_encodings:
        # Não foi possível gerar uma assinatura para os rostos encontrados
        return "Desconhecido"

    # 3. Calcula a área de cada rosto para encontrar o maior (mais próximo)
    face_areas = [(bottom - top) * (right - left) for top, right, bottom, left in face_locations]
    largest_face_index = np.argmax(face_areas)
    max_area = face_areas[largest_face_index]
    
    # 4. Verifica se o rosto está perto o suficiente
    if max_area < MIN_FACE_AREA_THRESHOLD:
        return "Aproxime-se"

    # 5. Seleciona a assinatura facial que corresponde ao maior rosto
    face_encoding_to_check = face_encodings[largest_face_index]
    
    name = "Desconhecido"
    
    # 6. Compara a assinatura do maior rosto com as conhecidas
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.5)
    
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
    
    return name

# ### NOVO ### Função para gerar URL pré-assinada para o áudio no S3
def generate_presigned_audio_url(person_name):
    """Gera uma URL segura e temporária para o arquivo audio.mp3 de uma pessoa."""
    object_key = f"known_faces/{person_name}/audio.mp3"
    try:
        url = s3_client.generate_presigned_url('get_object',
                                               Params={'Bucket': S3_BUCKET_NAME, 'Key': object_key},
                                               ExpiresIn=300) # URL válida por 5 minutos
        print(f"🔑 URL de áudio gerada para {person_name}")
        return url
    except ClientError as e:
        # Ocorre se o arquivo 'audio.mp3' não for encontrado para essa pessoa
        print(f"⚠️ AVISO: Não foi possível gerar URL para '{object_key}'. O arquivo existe? Erro: {e}")
        return None

# --- APLICAÇÃO WEB (FLASK) ---
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
    # Envia o nome para ser exibido na tela em tempo real
    socketio.emit('response', {'name': recognized_name})
    
    # ### MODIFICADO ### Lógica de reconhecimento e áudio
    # Verifica se a pessoa é conhecida e se é a PRIMEIRA vez que é reconhecida nesta sessão
    if recognized_name not in ["Desconhecido", "Aproxime-se"] and recognized_name not in recognized_person_set:
        print(f"✔️ Nova pessoa adicionada à lista: {recognized_name}")
        
        # 1. Adiciona à lista de presença
        recognized_person_set.add(recognized_name)
        sorted_list = sorted(list(recognized_person_set))
        socketio.emit('update_list', {'names': sorted_list})

        # 2. Tenta gerar e enviar a URL do áudio para o cliente
        audio_url = generate_presigned_audio_url(recognized_name)
        if audio_url:
            # Emite um evento específico para tocar o áudio no frontend
            socketio.emit('play_audio', {'url': audio_url})

# ### REMOVIDO ### A rota /get-speech não é mais necessária.
# @app.route('/get-speech', methods=['POST'])
# def get_speech():
#     ...

if __name__ == '__main__':
    load_known_faces()
    print("🚀 Servidor pronto!")
    print("   - Página da Webcam: http://127.0.0.1:5000")
    print("   - Página da Lista:  http://127.0.0.1:5000/list")
    socketio.run(app, host='0.0.0.0', port=5000)

