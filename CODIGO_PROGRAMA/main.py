import cv2
from ultralytics import YOLO
import numpy as np
import pygame  # Importa o pygame para tocar sons
import threading
# from signal import Signal
from gpiozero import OutputDevice

# from signal import Signal  # Importa a biblioteca para criação de threads

MODEL_PATH = "weights/best.pt"
ALERT_SOUND = "sound/danger3.mp3"

def load_model(MODEL_PATH):
    """Carrega o YOLO"""
    return YOLO(MODEL_PATH)

def open_video_source():
    """Abre o fonte de vídeo (webCam ou arquivo)"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
         print("Sem fonte em 0, tentando em 1")
         cap = cv2.VideoCapture(0)
    elif not cap.isOpened():
         raise ValueError("Erro: sem fonte de vídeo")
    return cap

def process_frame(frame, model, conf_threshold):
    """Processa um único quadro e retorna o frame com a segmentação aplicada"""
    results = model(frame, conf=conf_threshold)

    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

    mask_overlay = np.zeros_like(frame, dtype=np.uint8)

    for mask in masks:
        mask_binary = (mask * 255).astype(np.uint8)
        mask_binary_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0]))

        if len(mask_binary_resized.shape) == 2:
            mask_binary_resized = cv2.cvtColor(mask_binary_resized, cv2.COLOR_GRAY2BGR)

        # A cor verde para a luva (modificar conforme necessário)
        color_mask = mask_binary_resized * np.array([255, 55, 0], dtype=np.uint8)

        mask_overlay = cv2.add(mask_overlay, color_mask)

    return cv2.addWeighted(frame, 0.6, mask_overlay, 0.4, 0), masks  # Retorna as máscaras

def draw_center_rectangle(frame, change_color=False):
    """Desenha um retângulo no centro do frame. Muda de cor para vermelho se 'change_color' for True."""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    rect_width, rect_height = 100, 200

    top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
    bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)

    # Cor do retângulo: muda para vermelho se 'change_color' for True
    color = (0, 0, 255) if change_color else (0, 255, 255)

    # Desenha o texto dentro do retângulo
    if change_color:
        cv2.putText(frame, "PERIGO!", (center_x - 40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return cv2.rectangle(frame, top_left, bottom_right, color, 3)

def check_mask_inside_rectangle(masks, frame):
    """Verifica se qualquer parte da máscara da luva está dentro do retângulo central."""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    rect_width, rect_height = 150, 150

    # Definir a área do retângulo
    top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
    bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)

    # Verifica se alguma máscara verde está dentro do retângulo
    for mask in masks:
        mask_binary = (mask * 255).astype(np.uint8)
        mask_binary_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0]))

        # Encontrar os contornos da máscara
        contours, _ = cv2.findContours(mask_binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Verifica se qualquer ponto do contorno da máscara está dentro do retângulo
            for point in contour:
                x, y = point[0]
                # Verifica se o ponto está dentro do retângulo (também na parte superior)
                if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                    return True  # Retorna True assim que encontrar qualquer ponto dentro

    return False  # Nenhuma parte da máscara está dentro do retângulo

def play_sound():
    """Função para tocar o som em uma thread separada."""
    # Inicializa o pygame mixer para reprodução de áudio
    pygame.mixer.init()
    pygame.mixer.music.load(ALERT_SOUND)  # Carrega o arquivo de som
    pygame.mixer.music.play(-1)  # Reproduz o som em loop

def stop_sound():
    """Função para parar o som."""
    pygame.mixer.music.stop()

def run_segmentation(conf_threshold=0.879):
    """
    Executa a segmentação em tempo real usando o modelo YOLOv8-Seg com filtro de confiança.
    
    Args:
        model_path (str): Caminho para o modelo YOLOv8-Seg treinado (ex: 'best.pt').
        source (int or str): 0 para webcam, ou caminho para arquivo de vídeo.
        conf_threshold (float): Limiar de confiança para filtrar as detecções.
    """
    try:
        model = load_model(MODEL_PATH)
    
        cap = open_video_source()

        sound_played = False  # Variável para controlar o estado do som
        

        #signal = Signal(relay_pin=17)
        relay = OutputDevice(4)  # Substitui GPIO.setup
        

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erro: não foi possível capturar o quadro")
                break

            frame, masks = process_frame(frame, model, conf_threshold)

            # Verifica se a máscara da luva está dentro do retângulo
            is_mask_inside = check_mask_inside_rectangle(masks, frame)

            if is_mask_inside:

                # Inicia a thread para tocar o som continuamente
                if not sound_played:
                    sound_thread = threading.Thread(target=play_sound)
                    sound_thread.start()
                    sound_played = True  # Marca que o som foi tocado

                if not relay_activated:
                    
                    relay_activated = True
                    relay.on()  # Liga o relé

                     

            else:
                if sound_played:
                    stop_sound()  # Para o som se a luva sair da área
                    sound_played = False  # Reseta o controle de som
                if relay_activated:
                    relay.off()  # Desliga o relé
                    relay_activated = False

            # Desenha o retângulo (fica vermelho se a luva estiver dentro)
            frame = draw_center_rectangle(frame, change_color=is_mask_inside)

            cv2.imshow("Reconhecimento de Luva", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except ValueError as e:
        print(e)
    finally:
        # signal.cleanup_gpio()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_segmentation()



