"""
Projeto para detecção de poses.


Autores:
- Brunna Dalzini
- Edmilho dos Anjos
- Matheus Costa
- Maressa Karen

Última atualização: 27/11/2024
"""


import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog
import subprocess

videos = {}

frame_cam = None
frame_file = None
output_cam = None
output_file = None
my_landmarks = None
video_landmarks = None



def escolheMidia():
    # Criando uma imagem de menu para exibir no OpenCV
    menu = np.zeros((800, 900, 3), dtype=np.uint8)
    menu[:] = (50, 50, 50)  # Fundo cinza escuro
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(menu, "Escolha um video para reproduzir:", (30, 50), font, 1, (255, 255, 255), 2)

    for idx, (key, video) in enumerate(videos.items(), start=1):
        cv2.putText(menu, f"[{key}] {video}", (30, 100 + idx * 50), font, 0.7, (255, 255, 255), 2)

    cv2.putText(menu, "[0] Inserir link", (30, 100 + len(videos) * 50 + 50), font, 0.7, (255, 255, 255), 2)

    cv2.imshow("Menu de Videos", menu)

    selected_video = None

    while True:
        key = cv2.waitKey(0) & 0xFF  # Aguarda o pressionamento de uma tecla
        if chr(key) in videos:
            selected_video = videos[chr(key)]
            cv2.destroyAllWindows()
            print(f"Reproduzindo: {selected_video}")
            # Aqui você pode adicionar a lógica para reproduzir o vídeo
            break
        elif chr(key) == '0':  # Pressione '0' para inserir um link
            cv2.destroyAllWindows()
            root = tk.Tk()
            root.withdraw()  # Esconde a janela principal do Tkinter
            link = simpledialog.askstring("Input", "Digite o link do vídeo:")
            if link:
                teste_play(link)
                selected_video = "youtube_video_full.mp4"
                # Aqui você pode adicionar a lógica para reproduzir o link
            else:
                print("Nenhum link foi inserido.")
            break
        elif key == ord('q'):  # Pressione 'q' para sair
            cv2.destroyAllWindows()
            print("Saindo...")
            break
    return selected_video
# Função para calcular a porcentagem de acerto
def calculate_similarity(landmarks1, landmarks2, threshold=0.1):
    """
    Calcula a porcentagem de similaridade entre dois conjuntos de landmarks.
    :param landmarks1: Lista de pontos do esqueleto 1 (ex: você).
    :param landmarks2: Lista de pontos do esqueleto 2 (ex: vídeo).
    :param threshold: Limiar para considerar um movimento como "acerto".
    :return: Porcentagem de acerto.
    """
    if not landmarks1 or not landmarks2:
        return 0

    total_points = len(landmarks1)
    similar_points = 0

    for p1, p2 in zip(landmarks1, landmarks2):
        # Distância euclidiana
        dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        if dist < threshold:
            similar_points += 1

    return (similar_points / total_points) * 100

temp_full_path="youtube_video_full.mp4"
temp_video_path="youtube_video.mp4"
temp_audio_path="youtube_audio.mp4"

def teste_play(videoLink):
    global temp_full_path 

    print("Baixando vídeo com yt-dlp...")
    subprocess.run(
        ["yt-dlp", "-f", "best[ext=mp4]", "-o", temp_full_path, videoLink],
        check=True
    )
    print("Download concluído.")

def fazTudo():
    global frame_cam, frame_file, output_cam, output_file, my_landmarks, video_landmarks,video_path,temp_video_path,temp_audio_path,temp_full_path

    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_cam = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    

    selected_video = escolheMidia()
    temp_full_path = selected_video
    # split_video_and_audio(temp_full_path,temp_video_path,temp_audio_path)


    video_camera = cv2.VideoCapture(0)  # Webcam
    video_file = cv2.VideoCapture(selected_video)  # Vídeo


    # stop_event = threading.Event()
    # audio_thread = threading.Thread(target=play_audio, args=(temp_audio_path, stop_event))

    if selected_video:
        try:
            cv2.namedWindow('Meu Esqueleto', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Esqueleto do Vídeo', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Vídeo Original', cv2.WINDOW_NORMAL)
            # Evento para controle de parada

            # Iniciar thread para reprodução de áudio
            # audio_thread.start()

            while True:
                # Processar frame da webcam
                if video_camera.isOpened():
                    ok_cam, frame_cam = video_camera.read()
                    if ok_cam:
                        # Processar landmarks da câmera
                        frame_cam = cv2.resize(frame_cam, (1280, 720))
                        imageRGB_cam = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
                        # frame_cam = cv2.flip(frame_cam, 1)
                        results_cam = pose_cam.process(imageRGB_cam)
                        if results_cam.pose_landmarks:
                            my_landmarks = results_cam.pose_landmarks.landmark

                        # Desenhar esqueleto
                        output_cam = frame_cam.copy()
                        # output_cam = np.zeros_like(frame_cam)
                        if results_cam.pose_landmarks:
                            mp_drawing.draw_landmarks(output_cam, results_cam.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Processar frame do vídeo
                if video_file.isOpened():
                    ok_video, frame_file = video_file.read()
                    if ok_video:
                        frame_file = cv2.resize(frame_file, (1280, 720))

                        # Processar landmarks do vídeo
                        imageRGB_video = cv2.cvtColor(frame_file, cv2.COLOR_BGR2RGB)
                        results_video = pose_video.process(imageRGB_video)
                        if results_video.pose_landmarks:
                            video_landmarks = results_video.pose_landmarks.landmark

                        # Desenhar esqueleto
                        output_file = frame_file.copy()
                        if results_video.pose_landmarks:
                            mp_drawing.draw_landmarks(output_file, results_video.pose_landmarks, mp_pose.POSE_CONNECTIONS)


                
                output_cam=cv2.flip(output_cam, 1)
                # Calcular similaridade e exibir
                if my_landmarks and video_landmarks:
                    similarity = calculate_similarity(my_landmarks, video_landmarks)
                    cv2.putText(output_cam, f'Acerto: {similarity:.2f}%', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)



                # Exibir frames
                if output_cam is not None:
                    cv2.imshow('Meu Esqueleto', output_cam)
                    cv2.imshow('Meu vídeo', frame_cam)
                    
                if output_file is not None:
                    cv2.imshow('Esqueleto do Vídeo', output_file)
                    cv2.imshow('Vídeo Original', frame_file)

                # Pressionar 'q' para sair
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # stop_event.set()
                    break
        finally:
            # Liberar recursos
            video_camera.release()
            video_file.release()
            cv2.destroyAllWindows()
            # stop_event.set()
            # audio_thread.join()
            cv2.destroyAllWindows()


def listar_videos_mp4():
    global videos
    """
    Lista todos os arquivos .mp4 do diretório 'media/' e os adiciona no dicionário 'videos'.
    """
    media_dir = "media"  # Diretório onde estão os arquivos de mídia
    
    # Verifica se o diretório existe
    if not os.path.exists(media_dir):
        print(f"Diretório '{media_dir}' não encontrado.")
        return videos

    # Lista apenas os arquivos .mp4 no diretório
    arquivos = [f for f in os.listdir(media_dir) if f.endswith(".mp4") and os.path.isfile(os.path.join(media_dir, f))]

    # Adiciona os arquivos no dicionário 'videos'
    for idx, arquivo in enumerate(arquivos, start=1):
        videos[str(idx)] = os.path.join(media_dir, arquivo)
    
    return videos

listar_videos_mp4()

fazTudo()



