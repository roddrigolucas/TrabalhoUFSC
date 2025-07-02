import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
import numpy as np
import threading
from PIL import Image, ImageTk

# from src.signalanalogic import UniversalGPIO

MODEL_PATH = "weights/yolo11.pt"
LOGO_PATH = "image/logo.jpg"
ICON_PATH = os.path.abspath("image/favicon.ico")
CORRECT_PASSWORD = "tower25"  # Defina a senha desejada 

# Dimensões da Área Crítica (menor)
rect_critical_width = 100
rect_critical_height = 200

# Dimensões da Área Perigosa (envolve a Área Crítica)
rect_danger_width = 200
rect_danger_height = 300

conf_threshold = 0.879

class VisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI COMPUTER VISION SAETOWERS")
        self.root.configure(bg="#3B3B3B")
        
        # self.objectSignal = UniversalGPIO() #GPIO

        # Definir ícone da janela
        self.set_window_icon()

        # Criar um frame principal
        self.main_frame = tk.Frame(root, bg="#3B3B3B")
        self.main_frame.pack(fill="both", expand=True)

        # Criar um painel para exibir o vídeo
        self.video_label = tk.Label(self.main_frame, bg="black")
        self.video_label.pack(side="left", padx=10, pady=10)

        # Criar um frame lateral para os controles
        self.control_frame = tk.Frame(self.main_frame, bg="#3B3B3B", padx=10, pady=10)
        self.control_frame.pack(side="right", fill="y")

        # self.load_logo()

        title = tk.Label(self.control_frame, text="SAFE AREA CONFIGURATION", font=("Arial", 10, "bold"), fg="white", bg="#3B3B3B")
        title.pack(pady=10)

        # Criar seções de controle
        self.create_password_field()
        self.create_controls("CRITICAL AREA", "#00FF00", self.update_critical_width, self.update_critical_height)
        self.create_controls("DANGER AREA", "#5191f8", self.update_danger_width, self.update_danger_height)
        self.disable_controls()
        

        # Carregar modelo YOLO
        self.model = YOLO(MODEL_PATH)

        # Inicializar a captura de vídeo
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Sem fonte em 1, tentando em 0")
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Erro: sem fonte de vídeo")

        self.load_logo()
        self.update_frame()
    
    def set_window_icon(self):
        """Define o ícone da janela"""
        try:
            if os.path.exists(ICON_PATH):
                self.root.iconbitmap(ICON_PATH)  # Usa o método correto para .ico
            else:
                print(f"⚠️ Erro: O ícone não foi encontrado em '{ICON_PATH}'")
        except Exception as e:
            print(f"⚠️ Erro ao carregar ícone: {e}")

    def create_controls(self, title, color, update_width_func, update_height_func):
        """Cria uma seção de controles"""
        frame = tk.LabelFrame(self.control_frame, text=title, fg=color, bg="#3B3B3B", font=("Arial", 12, "bold"))
        frame.pack(fill="x", padx=5, pady=5)

        tk.Label(frame, text="Width:", bg="#3B3B3B", fg="white").pack(anchor="w", padx=5)
        width_slider = ttk.Scale(frame, from_=50, to=500, orient="horizontal", command=update_width_func)
        width_slider.set(150 if title == "CRITICAL AREA" else 250)
        width_slider.pack(fill="x", padx=5)

        tk.Label(frame, text="Height:", bg="#3B3B3B", fg="white").pack(anchor="w", padx=5)
        height_slider = ttk.Scale(frame, from_=50, to=500, orient="horizontal", command=update_height_func)
        height_slider.set(200 if title == "CRITICAL AREA" else 300)
        height_slider.pack(fill="x", padx=5)

    def update_critical_width(self, value):
        global rect_critical_width
        rect_critical_width = int(float(value))

    def update_critical_height(self, value):
        global rect_critical_height
        rect_critical_height = int(float(value))

    def update_danger_width(self, value):
        global rect_danger_width
        rect_danger_width = int(float(value))

    def update_danger_height(self, value):
        global rect_danger_height
        rect_danger_height = int(float(value))

    def process_frame(self, frame):
        """Processa um único quadro e retorna a segmentação"""
        results = self.model(frame, conf=conf_threshold)
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        for mask in masks:
            mask_binary = (mask * 255).astype(np.uint8)
            mask_binary_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0]))

            if len(mask_binary_resized.shape) == 2:
                mask_binary_resized = cv2.cvtColor(mask_binary_resized, cv2.COLOR_GRAY2BGR)

            color_mask = mask_binary_resized * np.array([255, 55, 0], dtype=np.uint8)
            mask_overlay = cv2.add(mask_overlay, color_mask)

        return cv2.addWeighted(frame, 0.6, mask_overlay, 0.4, 0), masks

    def draw_areas(self, frame, is_danger, is_critical):
        """Desenha as Áreas e muda a cor caso ultrapassadas"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Mudar a cor da área conforme violação
        danger_color = (200, 15, 0)  # Azul 🟦 (padrão)
        critical_color = (0, 255, 0)  # Verde 🟩 (padrão)

        if is_danger:
            danger_color = (0, 255, 255)  # Amarelo 🟡
        
        if is_critical:
            critical_color = (0, 0, 255)  # Vermelho 🔴

        # Desenhar retângulos
        cv2.rectangle(frame, (center_x - rect_danger_width // 2, center_y - rect_danger_height // 2),
                      (center_x + rect_danger_width // 2, center_y + rect_danger_height // 2), danger_color, 2)

        cv2.rectangle(frame, (center_x - rect_critical_width // 2, center_y - rect_critical_height // 2),
                      (center_x + rect_critical_width // 2, center_y + rect_critical_height // 2), critical_color, 3)

        
        # Exibir alertas
        if is_danger:
            cv2.putText(frame, "DANGER", (center_x - 80, center_y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
            # threading.Thread(target=self.objectSignal.set_danger, daemon=True).start()
        else:
            pass
            # threading.Thread(target=self.objectSignal.reset_danger, daemon=True).start()

        if is_critical:
            cv2.putText(frame, "CRITICAL", (center_x - 50, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            # threading.Thread(target=self.objectSignal.set_critical, daemon=True).start()
        else:
            pass
            # threading.Thread(target=self.objectSignal.reset_critical, daemon=True).start()

        return frame

    def update_frame(self):
        """Atualiza a interface gráfica com os frames"""
        ret, frame = self.cap.read()
        if ret:
            frame, masks = self.process_frame(frame)

            is_danger = self.check_mask_inside_rectangle(masks, rect_danger_width, rect_danger_height)
            is_critical = self.check_mask_inside_rectangle(masks, rect_critical_width, rect_critical_height)

            frame = self.draw_areas(frame, is_danger, is_critical)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def check_mask_inside_rectangle(self, masks, rect_width, rect_height):
        """Verifica se qualquer parte da máscara está dentro da área"""
        height, width = int(self.cap.get(4)), int(self.cap.get(3))
        center_x, center_y = width // 2, height // 2

        for mask in masks:
            mask_binary = (mask * 255).astype(np.uint8)
            mask_binary_resized = cv2.resize(mask_binary, (width, height))

            if np.any(mask_binary_resized[center_y - rect_height // 2:center_y + rect_height // 2,
                                          center_x - rect_width // 2:center_x + rect_width // 2]):
                return True
        return False
    
    def load_logo(self):
        """Carrega e exibe o logo no topo da barra lateral."""
        try:
            logo = Image.open(LOGO_PATH)
            logo = logo.resize((150, 80), Image.LANCZOS)  # Redimensionar para caber no painel
            self.logo_img = ImageTk.PhotoImage(logo)
            logo_label = tk.Label(self.control_frame, image=self.logo_img, bg="#3B3B3B")
            logo_label.pack(pady=10)
        except Exception as e:
            print(f"Erro ao carregar o logo: {e}")

    def create_password_field(self):
        """Adiciona um campo para digitar a senha."""
        password_frame = tk.LabelFrame(self.control_frame, text="ACCESS CONTROL", fg="red", bg="#3B3B3B", font=("Arial", 12, "bold"))
        password_frame.pack(fill="x", padx=5, pady=10)
        
        self.password_entry = ttk.Entry(password_frame, show="*")
        self.password_entry.pack(pady=5, padx=5, fill="x")
        
        check_button = ttk.Button(password_frame, text="ENABLE", command=self.check_password)
        check_button.pack(pady=5)

        check_button = ttk.Button(password_frame, text="DISABLE", command=self.disable_controls)
        check_button.pack(pady=5)
    
    def check_password(self):
        """Verifica se a senha digitada está correta."""
        entered_password = self.password_entry.get()
        
        if entered_password == CORRECT_PASSWORD:
            messagebox.showinfo("Access released", "Password correct!")
            self.password_entry.delete(0, tk.END)
            self.enable_controls()
        else:
            messagebox.showerror("Access Denied", "Incorrect password!")

    def disable_controls(self):
        """Desabilita todos os ttk.Entry, ttk.Scale e ttk.Button que estejam em self.control_frame ou seus descendentes."""
        for child in self.control_frame.winfo_children():
            # Se o child for outro container (p. ex., LabelFrame), percorra subwidgets
            for subchild in child.winfo_children():
                if isinstance(subchild, (ttk.Scale)):
                    subchild.config(state='disabled')
            # Se o próprio child for controle, desabilite
            if isinstance(child, (ttk.Scale)):
                child.config(state='disabled')

    def enable_controls(self):
        """Habilita todos os ttk.Entry, ttk.Scale e ttk.Button que estejam em self.control_frame ou seus descendentes."""
        for child in self.control_frame.winfo_children():
            # Se o child for outro container, percorra subwidgets
            for subchild in child.winfo_children():
                if isinstance(subchild, (ttk.Scale)):
                    subchild.config(state='normal')
            # Se o próprio child for controle, habilite
            if isinstance(child, (ttk.Scale)):
                child.config(state='normal')


if __name__ == "__main__":
    root = tk.Tk()
    app = VisionApp(root)
    root.mainloop()


