import cv2
import numpy as np
import sqlite3
import json
import joblib
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import datetime
import os
import pywt
from scipy.fftpack import fft2, fftshift
import threading
import ctypes
from ctypes import *

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Hikrobot SDK Path Setup
import sys
mv_sdk_path = os.getenv('MVCAM_COMMON_RUNENV', r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64") 
if os.path.exists(mv_sdk_path):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(mv_sdk_path)
    else:
        os.environ['PATH'] = mv_sdk_path + os.pathsep + os.environ['PATH']

# Try to find MvImport, typically in the installation Samples
sdk_import_path = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
if os.path.exists(sdk_import_path):
    sys.path.append(sdk_import_path)

try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

class HikrobotCamera:
    """Hikrobot (MVS) Camera Wrapper"""
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cam = None
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.is_opened = False
        self.is_grabbing = False
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        self.st_out_frame = MV_FRAME_OUT()
        self.t_grab = None
        self.running = False

    def open(self):
        if not SDK_AVAILABLE:
            return False, "Hikrobot SDK not found"
        
        # Enumerate devices
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != 0 or self.device_list.nDeviceNum == 0:
            return False, f"No device found (ret={ret})"

        if self.device_index >= self.device_list.nDeviceNum:
            return False, "Selected device index out of range"

        # Create handle
        self.cam = MvCamera()
        st_device_info = cast(self.device_list.pDeviceInfo[self.device_index], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(st_device_info)
        if ret != 0:
            return False, f"Create handle failed: {ret}"

        # Open device
        ret = self.cam.MV_CC_OpenDevice()
        if ret != 0:
            return False, f"Open device failed: {ret}"
        
        # Set trigger mode to OFF (continuous)
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)
        self.is_opened = True
        return True, "Success"

    def start(self):
        if not self.is_opened or self.is_grabbing:
            return False
        
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            return False
            
        self.is_grabbing = True
        self.running = True
        self.t_grab = threading.Thread(target=self._grab_thread, daemon=True)
        self.t_grab.start()
        return True

    def _grab_thread(self):
        while self.running:
            try:
                ret = self.cam.MV_CC_GetImageBuffer(self.st_out_frame, 1000)
                if ret == 0:
                    # Process the frame
                    st_info = self.st_out_frame.stFrameInfo
                    p_data = self.st_out_frame.pBufAddr

                    # Copy to local buffer (convert pixel type inside)
                    with self.frame_lock:
                        self.frame_buffer = self._process_frame(p_data, st_info)

                    self.cam.MV_CC_FreeImageBuffer(self.st_out_frame)
                else:
                    # Timeout/empty buffer -> küçük bekleme CPU kullanımını düşürür
                    threading.Event().wait(0.01)
            except Exception as e:
                print(f"Hikrobot grab thread error: {e}")
                threading.Event().wait(0.5)

    def _process_frame(self, p_data, st_info):
        """Convert Hikrobot frame to OpenCV (numpy) BGR format"""
        # Create a buffer for conversion
        n_data_len = st_info.nWidth * st_info.nHeight * 3
        p_bgr_buf = (c_ubyte * n_data_len)()
        
        st_convert_param = MV_CC_PIXEL_CONVERT_PARAM()
        st_convert_param.nWidth = st_info.nWidth
        st_convert_param.nHeight = st_info.nHeight
        st_convert_param.pSrcData = p_data
        st_convert_param.nSrcDataLen = st_info.nFrameLen
        st_convert_param.enSrcPixelType = st_info.enPixelType
        st_convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        st_convert_param.pDstBuffer = p_bgr_buf
        st_convert_param.nDstBufferSize = n_data_len
        
        ret = self.cam.MV_CC_ConvertPixelType(st_convert_param)
        if ret != 0:
            return None
            
        # Convert c_ubyte buffer to numpy array
        img_np = np.frombuffer(p_bgr_buf, dtype=np.uint8)
        img_np = img_np.reshape((st_info.nHeight, st_info.nWidth, 3))
        return img_np

    def get_frame(self):
        if not self.is_grabbing:
            # Try to grab a single frame if not continuous
            st_out_frame = MV_FRAME_OUT()
            ret = self.cam.MV_CC_GetImageBuffer(st_out_frame, 1000)
            if ret == 0:
                frame = self._process_frame(st_out_frame.pBufAddr, st_out_frame.stFrameInfo)
                self.cam.MV_CC_FreeImageBuffer(st_out_frame)
                return frame
            return None
            
        with self.frame_lock:
            return self.frame_buffer.copy() if self.frame_buffer is not None else None

    def stop(self):
        self.running = False
        if self.t_grab:
            self.t_grab.join(timeout=1)
        if self.is_grabbing:
            self.cam.MV_CC_StopGrabbing()
            self.is_grabbing = False

    def close(self):
        self.stop()
        if self.is_opened:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.is_opened = False

    def __del__(self):
        self.close()



class ProductionMonitoringSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Üretim Kamera İzleme Sistemi")
        # Tam ekran başlat
        self.root.state('zoomed')
        self.setup_styles()
        
        # Veritabanı başlat
        self.init_database()
        
        # SDK Başlat
        if SDK_AVAILABLE:
            MvCamera.MV_CC_Initialize()
        
        # Değişkenler
        self.camera = None
        self.hik_camera = None
        self.reference_image = None
        self.roi_coords = None        # geriye uyumluluk için
        self.roi_list = []            # [{"name", "coords":(x,y,w,h), "canvas_id", "color"}]
        self.current_project = None
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
        self.monitoring_active = False
        self.frame = None
        # Canlı önizleme
        self._preview_active = False
        self._preview_thread = None
        self._preview_source = None
        # Arka plan kamera okuma thread'i (ayarlar + kontrol ekranı için)
        self._cam_reader_active = False
        self._cam_reader_thread = None
        self._cam_reader_frame = None   # son okunan ham frame (BGR)
        self._cam_reader_lock = threading.Lock()
        # ML modelleri cache (project_id, roi_name) -> model
        self.ml_models = {}
        # ROI renk seti
        self._roi_colors = ['red','blue','orange','purple','cyan','magenta','yellow','lime']
        
        # Ana menüyü göster
        self.show_main_menu()

    def setup_styles(self):
        """TTK tema ve modern stil ayarları"""
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except tk.TclError:
            # Varsayılan temayı kullan
            pass

        base_bg = "#0f172a"
        header_bg = "#020617"
        card_bg = "#020617"
        primary = "#2563eb"
        primary_hover = "#1d4ed8"
        success = "#16a34a"
        danger = "#dc2626"
        accent = "#f97316"
        text_primary = "#e5e7eb"
        text_muted = "#9ca3af"

        # Pencere arka planı
        self.root.configure(bg=base_bg)

        # Genel frame stilleri
        style.configure("App.TFrame", background=base_bg)
        style.configure("Header.TFrame", background=header_bg)
        style.configure("Card.TFrame", background=card_bg)

        # Label stilleri
        style.configure("Title.TLabel",
                        font=('Segoe UI', 22, 'bold'),
                        background=base_bg,
                        foreground=text_primary)
        style.configure("Subtitle.TLabel",
                        font=('Segoe UI', 11),
                        background=base_bg,
                        foreground=text_muted)
        style.configure("HeaderTitle.TLabel",
                        font=('Segoe UI', 16, 'bold'),
                        background=header_bg,
                        foreground=text_primary)
        style.configure("Section.TLabel",
                        font=('Segoe UI', 11, 'bold'),
                        background=card_bg,
                        foreground=text_primary)
        style.configure("Muted.TLabel",
                        font=('Segoe UI', 9),
                        background=card_bg,
                        foreground=text_muted)

        # Buton baz stil
        style.configure("Primary.TButton",
                        font=('Segoe UI', 12, 'bold'),
                        padding=(18, 10),
                        background=primary,
                        foreground="white",
                        borderwidth=0)
        style.map("Primary.TButton",
                  background=[("active", primary_hover),
                              ("disabled", "#1e293b")])

        style.configure("Success.TButton",
                        font=('Segoe UI', 11, 'bold'),
                        padding=(14, 8),
                        background=success,
                        foreground="white",
                        borderwidth=0)
        style.map("Success.TButton",
                  background=[("active", "#15803d")])

        style.configure("Danger.TButton",
                        font=('Segoe UI', 11, 'bold'),
                        padding=(12, 8),
                        background=danger,
                        foreground="white",
                        borderwidth=0)
        style.map("Danger.TButton",
                  background=[("active", "#b91c1c")])

        style.configure("Accent.TButton",
                        font=('Segoe UI', 11, 'bold'),
                        padding=(14, 8),
                        background=accent,
                        foreground="white",
                        borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", "#ea580c")])

        style.configure("Ghost.TButton",
                        font=('Segoe UI', 10),
                        padding=(10, 6),
                        background=header_bg,
                        foreground=text_muted,
                        borderwidth=0)
        style.map("Ghost.TButton",
                  background=[("active", "#111827")],
                  foreground=[("active", text_primary)])
    
    def init_database(self):
        """Veritabanını başlat"""
        self.conn = sqlite3.connect('production_monitoring.db')
        self.cursor = self.conn.cursor()
        
        # Projeler tablosu
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                camera_id INTEGER,
                reference_image BLOB,
                roi_x INTEGER,
                roi_y INTEGER,
                roi_width INTEGER,
                roi_height INTEGER,
                created_date TEXT,
                updated_date TEXT
            )
        ''')
        
        # Yeni sütunları ekle (mevcut DB'ye geriye uyumlu)
        for col, typedef in [
            ("roi_list", "TEXT"),
            ("algorithm", "TEXT DEFAULT 'SSIM'"),
            ("algo_threshold", "REAL DEFAULT 0.75"),
        ]:
            try:
                self.cursor.execute(f"ALTER TABLE projects ADD COLUMN {col} {typedef}")
            except Exception:
                pass  # Sütun zaten varsa atla
        
        # İzleme kayıtları tablosu
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                timestamp TEXT,
                status TEXT,
                similarity_score REAL,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # ML model bilgileri tablosu (her ROI için)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS roi_ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                roi_name TEXT NOT NULL,
                model_type TEXT,
                model_path TEXT,
                params_json TEXT,
                created_date TEXT,
                updated_date TEXT,
                UNIQUE(project_id, roi_name),
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')

        self.conn.commit()

    def _safe_name(self, s: str) -> str:
        """Dosya/klasör isimleri için güvenli ad üret."""
        s = (s or '').strip()
        if not s:
            return 'ROI'
        bad = '<>:"/\\|?*'
        for ch in bad:
            s = s.replace(ch, '_')
        return s
    
    def get_camera_source(self, camera_id):
        """Kamera ID'sine göre kaynak döndür (IP Kamera desteği için)"""
        # Kullanıcı Kamera 100 (IP) seçtiğinde 192.168.1.37 IP'si kullanılacak
        if camera_id == 100 or camera_id == 3: # ID 3 is legacy
            return "http://192.168.1.37:4747/video"
        elif camera_id == 101 or camera_id == 4: # ID 4 and 101 are Hikrobot
            return "HIKROBOT"
        return camera_id

    # ──────────────────────────────────────────────────────────────
    # Arka Plan Kamera Okuma Thread'i
    # ──────────────────────────────────────────────────────────────
    def _start_cam_reader(self, source):
        """Verilen kaynak için arka planda frame okuyan thread başlat."""
        self._stop_cam_reader()
        self._cam_reader_active = True
        self._cam_reader_frame = None
        self._cam_reader_thread = threading.Thread(
            target=self._cam_reader_loop, args=(source,), daemon=True
        )
        self._cam_reader_thread.start()

    def _stop_cam_reader(self):
        """Arka plan okuma thread'ini durdur."""
        self._cam_reader_active = False
        if self._cam_reader_thread:
            self._cam_reader_thread.join(timeout=2)
            self._cam_reader_thread = None
        self._cam_reader_frame = None

    def _cam_reader_loop(self, source):
        """Arka planda sürekli frame okuyan döngü (thread içinde çalışır)."""
        import time
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)          # IP/URL → FFmpeg
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # USB

        while self._cam_reader_active:
            if not cap.isOpened():
                # Bağlantı kesildiyse yeniden dene
                time.sleep(1)
                if isinstance(source, str):
                    cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                continue

            ret, frame = cap.read()
            if ret:
                with self._cam_reader_lock:
                    self._cam_reader_frame = frame
            else:
                time.sleep(0.05)  # başarısız okumada kısa bekle

        cap.release()

    def _get_latest_frame(self):
        """Arka plan thread'inden en son frame'i al (BGR). Yoksa None döner."""
        with self._cam_reader_lock:
            return self._cam_reader_frame.copy() if self._cam_reader_frame is not None else None
    
    def show_main_menu(self):
        """Ana menü ekranı"""
        # Mevcut widget'ları temizle
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Ana frame (tam ekran arka plan)
        main_frame = ttk.Frame(self.root, style="App.TFrame")
        main_frame.pack(expand=True, fill='both')
        
        # Ortalanmış içerik kartı
        content = ttk.Frame(main_frame, style="Card.TFrame")
        content.place(relx=0.5, rely=0.5, anchor='center')

        title = ttk.Label(
            content,
            text="Üretim Kamera İzleme Sistemi",
            style="Title.TLabel"
        )
        title.pack(pady=(32, 8), padx=40)

        subtitle = ttk.Label(
            content,
            text="Projelerinizi oluşturun, izleme parametrelerini ayarlayın ve üretimi gerçek zamanlı takip edin.",
            style="Subtitle.TLabel",
            wraplength=520,
            justify="center"
        )
        subtitle.pack(pady=(0, 28), padx=40)
        
        # Butonlar
        button_frame = ttk.Frame(content, style="Card.TFrame")
        button_frame.pack(padx=40, pady=(0, 32), fill='x')
        
        new_project_btn = ttk.Button(
            button_frame,
            text="Yeni Proje Oluştur",
            style="Primary.TButton",
            command=self.new_project_screen
        )
        new_project_btn.pack(pady=(0, 12), fill='x')
        
        start_control_btn = ttk.Button(
            button_frame,
            text="Kontrole Başla",
            style="Success.TButton",
            command=self.control_screen
        )
        start_control_btn.pack(pady=6, fill='x')

        settings_btn = ttk.Button(
            button_frame,
            text="⚙  Ayarlar",
            style="Accent.TButton",
            command=self.settings_screen
        )
        settings_btn.pack(pady=(12, 0), fill='x')
    
    def scan_cameras(self):
        """Mevcut kameraları tara (USB/Dahili 0-9 + IP Kamera + Hikrobot)"""
        found = []
        found.append((0, "Kamera Seç"))
        # USB/Dahili kameraları tara (0-9)
        for idx in range(1,10):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                found.append((idx, f"Kamera {idx} (USB/Dahili)"))
                cap.release()
        
        # IP kamera sabiti (ID 100)
        found.append((100, "Kamera 100 (IP - 192.168.1.37:4747)"))
        
        if SDK_AVAILABLE:
            # Hikrobot Kamera (ID 101)
            found.append((101, "Kamera 101 (Hikrobot)"))
        return found

    def start_live_preview(self, source):
        """Canlı önizlemeyi başlat"""
        self.stop_live_preview()
        self._preview_active = True
        self._preview_source = source
        import threading as _t
        self._preview_thread = _t.Thread(target=self._live_preview_loop, daemon=True)
        self._preview_thread.start()

    def stop_live_preview(self):
        """Canlı önizlemeyi durdur"""
        self._preview_active = False
        if self._preview_thread:
            self._preview_thread.join(timeout=1)
            self._preview_thread = None

    def _live_preview_loop(self):
        """Önizleme thread fonksiyonu"""
        source = self._preview_source
        if source == "HIKROBOT":
            if not self.hik_camera:
                self.hik_camera = HikrobotCamera()
                ok, msg = self.hik_camera.open()
                if not ok:
                    return
            self.hik_camera.start()
            while self._preview_active:
                frame = self.hik_camera.get_frame()
                if frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.root.after(0, lambda f=rgb: self._update_preview_canvas(f))
                import time; time.sleep(0.033)
        else:
            # IP kamera (str URL) için CAP_DSHOW kullanma; USB için kullan
            if isinstance(source, str):
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            # IP kamera için bağlantı zaman aşımı ayarla
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            while self._preview_active and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.root.after(0, lambda f=rgb: self._update_preview_canvas(f))
                import time; time.sleep(0.033)
            cap.release()

    def _update_preview_canvas(self, rgb_frame):
        """Preview frame'ini canvas'a çiz (ana thread)"""
        if not self._preview_active:
            return
        self.reference_image = rgb_frame
        h, w = rgb_frame.shape[:2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        scale = min(cw / w, ch / h) * 0.95
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(rgb_frame, (nw, nh))
        self.display_scale = scale
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.canvas_offset = (ox, oy)
        pil = Image.fromarray(img_r)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete('bg_img')
        self.canvas.create_image(ox, oy, anchor='nw', image=self.photo, tags='bg_img')
        # ROI dikdörtgenlerini yeniden çiz
        for roi_item in self.roi_list:
            coords = roi_item['coords']
            x1c = int(coords[0] * scale) + ox
            y1c = int(coords[1] * scale) + oy
            x2c = int((coords[0] + coords[2]) * scale) + ox
            y2c = int((coords[1] + coords[3]) * scale) + oy
            self.canvas.create_rectangle(x1c, y1c, x2c, y2c,
                                         outline=roi_item['color'], width=2,
                                         tags=f"roi_{roi_item['name']}")
            self.canvas.create_text(x1c + 4, y1c + 4, anchor='nw',
                                    text=roi_item['name'],
                                    fill=roi_item['color'], font=('Arial', 9, 'bold'),
                                    tags=f"roi_{roi_item['name']}")

    def new_project_screen(self):
        """Yeni proje oluşturma ekranı"""
        self.stop_live_preview()
        self.roi_list = []
        for widget in self.root.winfo_children():
            widget.destroy()

        # Ana frame
        project_frame = ttk.Frame(self.root, style="App.TFrame")
        project_frame.pack(expand=True, fill='both')

        # Üst panel
        top_panel = ttk.Frame(project_frame, style="Header.TFrame", height=60)
        top_panel.pack(fill='x')
        top_panel.pack_propagate(False)
        ttk.Label(
            top_panel,
            text="Yeni Proje Oluştur",
            style="HeaderTitle.TLabel"
        ).pack(side='left', padx=80)
        ttk.Button(
            top_panel,
            text="← Ana Menü",
            command=lambda: [self.stop_live_preview(), self.show_main_menu()],
            style="Ghost.TButton"
        ).place(x=10, y=15)

        # ── Sol Panel ──────────────────────────────────
        left_panel = ttk.Frame(project_frame, style="Card.TFrame", width=320)
        left_panel.pack(side='left', fill='y', padx=(10,5), pady=10)
        left_panel.pack_propagate(False)

        # Kamera Seçimi
        ttk.Label(left_panel, text="Kamera Seçimi", style="Section.TLabel").pack(pady=(14,4))

        cam_row = ttk.Frame(left_panel, style="Card.TFrame")
        cam_row.pack(fill='x', padx=10)

        self._cam_options = []   # (source_val, label) tuples
        self.camera_combo = ttk.Combobox(cam_row, state='readonly', width=22,
                                         font=('Arial', 10))
        self.camera_combo.pack(side='left', padx=(0,5))

        def _scan_and_fill():
            cams = self.scan_cameras()
            self._cam_options = cams
            print(cams)
            self.camera_combo['values'] = [c[1] for c in cams]
            if cams:
                self.camera_combo.current(0)
                _on_cam_select(None)

        ttk.Button(cam_row, text="🔍 Tara", command=_scan_and_fill,
                   style="Ghost.TButton").pack(side='left')

        def _on_cam_select(event):
            idx = self.camera_combo.current()
            if idx < 0 or idx >= len(self._cam_options):
                return
            src_val, _ = self._cam_options[idx]
            source = self.get_camera_source(src_val)
            self.camera_var.set(src_val)
            self.start_live_preview(source)

        self.camera_var = tk.IntVar(value=0)
        self.camera_combo.bind('<<ComboboxSelected>>', _on_cam_select)

        # Resim Çek (anlık snapshot)
        ttk.Button(left_panel, text="📷 Anlık Görüntü Al",
                   style="Primary.TButton",
                   command=self.capture_reference).pack(
                       pady=(10,4), padx=14, fill='x')

        # Ayırıcı
        tk.Frame(left_panel, bg='#1f2937', height=1).pack(fill='x', padx=10, pady=8)

        # Algoritma Seçimi
        ttk.Label(left_panel, text="Algoritma", style="Section.TLabel").pack(anchor='w', padx=14)

        ALGO_OPTIONS = [
            "Basit SSIM (Varsayılan)",
            "Template Matching",
            "Histogram Karşılaştırma",
            "Fourier Dönüşümü",
            "Wavelet Dönüşümü",
            "Brute-Force ORB"
        ]
        self.algo_combo = ttk.Combobox(left_panel, values=ALGO_OPTIONS,
                                       state='readonly', font=('Arial', 10))
        self.algo_combo.current(0)
        self.algo_combo.pack(fill='x', padx=14, pady=4)

        thresh_row = ttk.Frame(left_panel, style="Card.TFrame")
        thresh_row.pack(fill='x', padx=14, pady=(0,8))
        ttk.Label(thresh_row, text="Eşik:", style="Muted.TLabel").pack(side='left')
        self.threshold_var = tk.DoubleVar(value=0.75)
        tk.Entry(thresh_row, textvariable=self.threshold_var, width=7,
                 font=('Arial', 10)).pack(side='left', padx=6)

        # Ayırıcı
        tk.Frame(left_panel, bg='#1f2937', height=1).pack(fill='x', padx=10, pady=4)

        # ROI Listesi
        ttk.Label(left_panel, text="ROI Bölgeleri", style="Section.TLabel").pack(anchor='w', padx=14, pady=(6,2))
        ttk.Label(left_panel,
                  text="Görüntü üzerinde sürükle-bırak ile ROI çizin.\nHer bölgeye anlamlı bir isim verin.",
                  style="Muted.TLabel",
                  justify='left').pack(anchor='w', padx=14)

        # Scrollable ROI listesi
        roi_frame_outer = ttk.Frame(left_panel, style="Card.TFrame")
        roi_frame_outer.pack(fill='both', expand=True, padx=10, pady=4)

        roi_scrollbar = tk.Scrollbar(roi_frame_outer)
        roi_scrollbar.pack(side='right', fill='y')
        self.roi_listbox_frame = tk.Frame(roi_frame_outer, bg='#020617')
        self.roi_listbox_frame.pack(fill='both', expand=True)

        # Kaydet (bottom)
        self.save_btn = ttk.Button(left_panel, text="💾 Projeyi Kaydet",
                                   style="Success.TButton",
                                   command=self.save_project,
                                   state='disabled')
        self.save_btn.pack(pady=10, padx=14, fill='x', side='bottom')

        # ── Sağ Panel – Canvas ──────────────────────────
        right_panel = ttk.Frame(project_frame, style="App.TFrame")
        right_panel.pack(side='right', expand=True, fill='both', padx=(5,10), pady=10)

        self.canvas = tk.Canvas(right_panel, bg='#2c3e50', cursor='cross')
        self.canvas.pack(expand=True, fill='both')

        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.temp_rect = None

        # İlk açılışta kameraları otomatik tara
        self.root.after(200, _scan_and_fill)

    def _refresh_roi_listbox(self):
        """Sol paneldeki ROI listesini güncelle"""
        for w in self.roi_listbox_frame.winfo_children():
            w.destroy()
        for idx, roi_item in enumerate(self.roi_list):
            row = tk.Frame(self.roi_listbox_frame, bg='#020617')
            row.pack(fill='x', pady=2, padx=2)
            c = roi_item['color']
            x, y, ww, hh = roi_item['coords']
            tk.Label(row, text="●", fg=c, bg='#020617', font=('Arial', 12)).pack(side='left')
            tk.Label(row, text=f"{roi_item['name']}  ({ww}×{hh})",
                     bg='#020617', fg='#e5e7eb', font=('Segoe UI', 9), anchor='w').pack(side='left', expand=True, fill='x')
            tk.Button(row, text="✕", fg='white', bg='#e74c3c',
                      font=('Arial', 8, 'bold'), width=2, cursor='hand2',
                      command=lambda i=idx: self._delete_roi(i)).pack(side='right')
        if self.roi_list:
            self.save_btn.config(state='normal')
        else:
            self.save_btn.config(state='disabled')

    def _delete_roi(self, index):
        """ROI'yi listeden ve canvas'tan sil"""
        if 0 <= index < len(self.roi_list):
            name = self.roi_list[index]['name']
            self.canvas.delete(f"roi_{name}")
            self.roi_list.pop(index)
            self._refresh_roi_listbox()

    def _ask_roi_name(self):
        """ROI ismi sor, None dönerse iptal"""
        suggested = f"ROI_{len(self.roi_list)+1}"
        name = simpledialog.askstring("ROI İsmi",
                                      "Bu bölge için bir isim girin:",
                                      initialvalue=suggested,
                                      parent=self.root)
        return name

    
    def capture_reference(self):
        """Referans görüntü yakala"""
        # Kaynak açılmadan önce önizlemeyi durdur (resource conflict önlemek için)
        self.stop_live_preview()
        
        camera_id = self.camera_var.get()
        self.roi_list = [] # Yeni resim çekilince ROI'ler sıfırlanır
        if hasattr(self, 'reference_rois'):
            self.reference_rois = []
        
        try:
            source = self.get_camera_source(camera_id)
            
            if source == "HIKROBOT":
                if not self.hik_camera:
                    self.hik_camera = HikrobotCamera()
                    success, msg = self.hik_camera.open()
                    if not success:
                        messagebox.showerror("Hata", f"Hikrobot Kamera açılamadı: {msg}")
                        self.hik_camera = None
                        return

                frame = self.hik_camera.get_frame()
                ret = frame is not None
                if ret:
                    self.reference_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.display_image(self.reference_image)
                    messagebox.showinfo("Başarılı", "Görüntü yakalandı! ROI bölgesini işaretleyiniz.")
                else:
                    messagebox.showerror("Hata", "Görüntü yakalanamadı!")
            else:
                if isinstance(source, int):
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(source)
                    
                if not cap.isOpened():
                    messagebox.showerror("Hata", f"Kamera kaynağına erişilemedi: {source}")
                    return
                
                ret, frame = cap.read()
                if ret:
                    self.reference_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.display_image(self.reference_image)
                    messagebox.showinfo("Başarılı", "Görüntü yakalandı! ROI bölgesini işaretleyiniz.")
                else:
                    messagebox.showerror("Hata", "Kamera görüntüsü alınamadı!")
                
                cap.release()
                
        except Exception as e:
            messagebox.showerror("Hata", f"Kamera hatası: {str(e)}")
    
    def display_image(self, image):
        """Görüntüyü canvas'a göster"""
        # Görüntüyü yeniden boyutlandır
        h, w = image.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            scale = min(canvas_w/w, canvas_h/h) * 0.95
            new_w, new_h = int(w*scale), int(h*scale)
            
            img_resized = cv2.resize(image, (new_w, new_h))
            self.display_scale = scale
            
            # PIL Image'e çevir
            img_pil = Image.fromarray(img_resized)
            self.photo = ImageTk.PhotoImage(img_pil)
            
            # Canvas'ı temizle ve göster
            self.canvas.delete('all')
            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            self.canvas_offset = (x, y)
            self.canvas.create_image(x, y, anchor='nw', image=self.photo)
    
    def on_mouse_down(self, event):
        """Mouse basıldığında"""
        if self.reference_image is not None:
            self.roi_start = (event.x, event.y)
            self.drawing = True
    
    def on_mouse_move(self, event):
        """Mouse hareket ederken"""
        if self.drawing and self.roi_start:
            if self.temp_rect:
                self.canvas.delete(self.temp_rect)
            
            x1, y1 = self.roi_start
            x2, y2 = event.x, event.y
            self.temp_rect = self.canvas.create_rectangle(x1, y1, x2, y2,
                                                          outline='red', width=3,
                                                          dash=(5, 5))
    
    def on_mouse_up(self, event):
        """Mouse bırakıldığında – çoklu ROI"""
        if self.drawing and self.roi_start:
            self.roi_end = (event.x, event.y)
            self.drawing = False

            if self.temp_rect:
                self.canvas.delete(self.temp_rect)
                self.temp_rect = None

            if not hasattr(self, 'canvas_offset') or not hasattr(self, 'display_scale'):
                return

            x1 = min(self.roi_start[0], self.roi_end[0]) - self.canvas_offset[0]
            y1 = min(self.roi_start[1], self.roi_end[1]) - self.canvas_offset[1]
            x2 = max(self.roi_start[0], self.roi_end[0]) - self.canvas_offset[0]
            y2 = max(self.roi_start[1], self.roi_end[1]) - self.canvas_offset[1]

            x1 = max(0, int(x1 / self.display_scale))
            y1 = max(0, int(y1 / self.display_scale))
            x2 = max(0, int(x2 / self.display_scale))
            y2 = max(0, int(y2 / self.display_scale))

            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 10:
                messagebox.showwarning("Uyarı", "ROI bölgesi çok küçük!")
                return

            name = self._ask_roi_name()
            if not name:
                return

            color = self._roi_colors[len(self.roi_list) % len(self._roi_colors)]
            cx1 = int(x1 * self.display_scale) + self.canvas_offset[0]
            cy1 = int(y1 * self.display_scale) + self.canvas_offset[1]
            cx2 = int(x2 * self.display_scale) + self.canvas_offset[0]
            cy2 = int(y2 * self.display_scale) + self.canvas_offset[1]
            tag = f"roi_{name}"
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, tags=tag)
            self.canvas.create_text(cx1+4, cy1+4, anchor='nw', text=name,
                                    fill=color, font=('Arial', 9, 'bold'), tags=tag)

            self.roi_list.append({'name': name, 'coords': (x1, y1, w, h), 'color': color})
            if len(self.roi_list) == 1:
                self.roi_coords = (x1, y1, w, h)
            self._refresh_roi_listbox()



    def save_project(self):
        """Projeyi kaydet (roi_list + algoritma + eşik)"""
        if self.reference_image is None:
            messagebox.showwarning("Uyarı", "Lütfen önce kameradan görüntü alın!")
            return
        if not self.roi_list:
            messagebox.showwarning("Uyarı", "Lütfen en az bir ROI bölgesi seçin!")
            return

        project_name = simpledialog.askstring("Proje Adı", "Proje adını girin:")
        if not project_name:
            return

        try:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR))
            img_binary = buffer.tobytes()
            now = datetime.datetime.now().isoformat()

            roi_json = json.dumps([
                {'name': r['name'], 'x': r['coords'][0], 'y': r['coords'][1],
                 'w': r['coords'][2], 'h': r['coords'][3]}
                for r in self.roi_list
            ], ensure_ascii=False)

            algo_map = {
                "Basit SSIM (Varsayılan)": "SSIM",
                "Template Matching": "TEMPLATE",
                "Histogram Karşılaştırma": "HISTOGRAM",
                "Fourier Dönüşümü": "FOURIER",
                "Wavelet Dönüşümü": "WAVELET",
                "Brute-Force ORB": "ORB"
            }
            algo_key = algo_map.get(
                self.algo_combo.get() if hasattr(self, 'algo_combo') else "", "SSIM")
            try:
                thresh = float(self.threshold_var.get())
            except Exception:
                thresh = 0.75

            first = self.roi_list[0]['coords']

            self.cursor.execute('''
                INSERT INTO projects (name, camera_id, reference_image,
                                     roi_x, roi_y, roi_width, roi_height,
                                     roi_list, algorithm, algo_threshold,
                                     created_date, updated_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (project_name, self.camera_var.get(), img_binary,
                  first[0], first[1], first[2], first[3],
                  roi_json, algo_key, thresh, now, now))

            self.conn.commit()
            self.stop_live_preview()
            messagebox.showinfo("Başarılı", f"Proje '{project_name}' kaydedildi!")
            self.show_main_menu()

        except sqlite3.IntegrityError:
            messagebox.showerror("Hata", "Bu isimde bir proje zaten var!")
        except Exception as e:
            messagebox.showerror("Hata", f"Kayıt hatası: {str(e)}")


    def settings_screen(self):
        """Ayarlar ekranı - Proje Seçimi"""
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Proje seç
        self.cursor.execute('SELECT id, name FROM projects ORDER BY created_date DESC')
        projects = self.cursor.fetchall()
        
        if not projects:
            messagebox.showwarning("Uyarı", "Henüz kayıtlı proje yok!")
            self.show_main_menu()
            return
        
        # Proje seçim dialogu
        project_names = [p[1] for p in projects]
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Validation - Proje Seç")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Test edilecek projeyi seçin:",
                font=('Arial', 12, 'bold')).pack(pady=20)
        
        selected_project = tk.StringVar()
        
        listbox = tk.Listbox(dialog, font=('Arial', 11), height=8)
        listbox.pack(pady=10, padx=20, fill='both', expand=True)
        
        for name in project_names:
            listbox.insert(tk.END, name)
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_project.set(project_names[selection[0]])
                dialog.destroy()
        
        tk.Button(dialog, text="Seç ve Devam Et", command=on_select,
                 bg='#e67e22', fg='white', font=('Arial', 12)).pack(pady=10)
        
        dialog.wait_window()
        
        if not selected_project.get():
            self.show_main_menu()
            return
        
        # Seçilen projeyi yükle
        project_id = [p[0] for p in projects if p[1] == selected_project.get()][0]
        self.load_project(project_id)
        self.start_settings()

    def start_settings(self):
        """Ayarlar ekranını başlat"""
        # Ekranı temizle
        for widget in self.root.winfo_children():
            widget.destroy()

        # Ana container
        main_container = ttk.Frame(self.root, style="App.TFrame")
        main_container.pack(fill='both', expand=True)
        
        # Üst Panel (Header)
        header_frame = ttk.Frame(main_container, style="Header.TFrame", height=60)
        header_frame.pack(fill='x', side='top')
        header_frame.pack_propagate(False)
        
        ttk.Button(header_frame, text="← Ana Menü", command=self.stop_monitoring,
                   style="Danger.TButton").pack(side='left', padx=10, pady=10)
        
        ttk.Label(header_frame, text=f"Ayarlar: {self.current_project['name']}",
                  style="HeaderTitle.TLabel").pack(side='left', padx=20)

        # Sol Panel (Kontroller)
        left_panel = ttk.Frame(main_container, style="Card.TFrame", width=300)
        left_panel.pack(side='left', fill='y', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Yöntem Seçimi (Combobox)
        ttk.Label(left_panel, text="Algoritma Seçimi", style="Section.TLabel").pack(pady=(10, 5))
        
        self.algo_options_map = {
            "Basit SSIM (Standart)": "SSIM",
            "Industrial IV3 AI": "INDUSTRIAL_AI",
            "Advanced IV3 Engine": "ADVANCED_ENGINE",
            "SIFT Hizalama + SSIM": "ALIGN_SSIM",
            "Template Matching": "TEMPLATE",
            "Histogram Karşılaştırma": "HISTOGRAM",
            "Fourier Dönüşümü": "FOURIER",
            "Wavelet Dönüşümü": "WAVELET",
            "Brute-Force ORB": "ORB",
            "ML model (HOG-SVM)": "ML_MODEL",
            "ML model (SIFT-SVM)": "ML_MODEL_SIFT",
            "DAISY -SVM": "ML_MODEL_DAISY",
            "Haar-like -SVM": "ML_MODEL_HAAR",
            "CENSURE -SVM": "ML_MODEL_CENSURE",
            "Multi-Block LBP -SVM": "ML_MODEL_MBLBP",
            "GLCM -SVM": "ML_MODEL_GLCM",
            "LBP -SVM": "ML_MODEL_LBP",
            "Gabor -SVM": "ML_MODEL_GABOR",
            "Fisher Vector -SVM": "ML_MODEL_FISHER",
        }
        self.algo_combo_settings = ttk.Combobox(left_panel, values=list(self.algo_options_map.keys()), state='readonly', font=('Arial', 10))
        
        # Mevcut algoritmayı seç
        current_algo = self.current_project.get('algorithm', 'SSIM')
        for display_name, internal_name in self.algo_options_map.items():
            if internal_name == current_algo:
                self.algo_combo_settings.set(display_name)
                break
        else:
            self.algo_combo_settings.current(0)
            
        self.algo_combo_settings.pack(fill='x', padx=20, pady=5)
        
        # Eşik Değer Girişi
        ttk.Label(left_panel, text="Eşik Değeri / Parametre", style="Section.TLabel").pack(pady=(20, 5))
        
        self.threshold_var = tk.DoubleVar(value=0.75)
        entry_frame = ttk.Frame(left_panel, style="Card.TFrame")
        entry_frame.pack(pady=5)
        
        tk.Entry(entry_frame, textvariable=self.threshold_var, width=10, font=('Arial', 12)).pack(side='left')
        
        # Ayarları Kaydet Butonu
        def save_settings():
            try:
                display_name = self.algo_combo_settings.get()
                algo_internal = self.algo_options_map.get(display_name, "SSIM")
                thresh = float(self.threshold_var.get())
                
                self.cursor.execute('''
                    UPDATE projects 
                    SET algorithm = ?, algo_threshold = ?, updated_date = ?
                    WHERE id = ?
                ''', (algo_internal, thresh, datetime.datetime.now().isoformat(), self.current_project['id']))
                self.conn.commit()
                
                # Bellekteki projeyi güncelle
                self.current_project['algorithm'] = algo_internal
                self.current_project['algo_threshold'] = thresh
                
                messagebox.showinfo("Başarılı", "Ayarlar kaydedildi!")
            except Exception as e:
                messagebox.showerror("Hata", f"Kaydetme hatası: {e}")

        ttk.Button(left_panel, text="💾 Ayarları Kaydet", command=save_settings,
                   style="Success.TButton").pack(pady=15, padx=20, fill='x')

        # ML model eğitimi butonu (sadece ML_MODEL seçiliyken aktif)
        def on_algo_change(event=None):
            display = self.algo_combo_settings.get()
            internal = self.algo_options_map.get(display, "SSIM")
            state = 'normal' if internal in (
                "ML_MODEL", "ML_MODEL_SIFT", "ML_MODEL_DAISY",
                "ML_MODEL_HAAR", "ML_MODEL_CENSURE", "ML_MODEL_MBLBP",
                "ML_MODEL_GLCM", "ML_MODEL_LBP", "ML_MODEL_GABOR",
                "ML_MODEL_FISHER",
            ) else 'disabled'
            self.train_ml_btn.config(state=state)

        self.algo_combo_settings.bind("<<ComboboxSelected>>", on_algo_change)

        self.train_ml_btn = ttk.Button(
            left_panel,
            text="🧠 ML Model Eğit",
            style="Primary.TButton",
            command=self.train_ml_models,
            state='disabled'
        )
        self.train_ml_btn.pack(pady=5, padx=20, fill='x')

        # İlk açılışta doğru state ayarı
        on_algo_change()

        # Anlık Sonuç Göstergeleri
        ttk.Label(left_panel, text="Test Sonuçları (Anlık)", style="Section.TLabel").pack(pady=(20, 5))
        
        self.score_label = tk.Label(left_panel, text="Skor: -", font=('Segoe UI', 11), bg='#020617', fg='#e5e7eb')
        self.score_label.pack(pady=5)
        
        self.result_label = tk.Label(left_panel, text="DURUM: -", font=('Segoe UI', 14, 'bold'),
                                     bg='#020617', fg='#e5e7eb')
        self.result_label.pack(pady=10)
        
        self.detail_label = tk.Label(left_panel, text="", font=('Segoe UI', 9),
                                     bg='#020617', fg='#9ca3af', wraplength=280)
        self.detail_label.pack(pady=5)

        # ROI label (OK / Not OK) seçimi
        label_frame = ttk.LabelFrame(left_panel, text="ROI Etiketi", padding=(10, 5))
        label_frame.pack(pady=(15, 5), padx=20, fill='x')

        self.roi_label_var = tk.StringVar(value="")
        tk.Radiobutton(label_frame, text="OK", variable=self.roi_label_var,
                       value="OK", bg='#020617', fg='#e5e7eb',
                       selectcolor='#020617', anchor='w').pack(fill='x')
        tk.Radiobutton(label_frame, text="Not OK", variable=self.roi_label_var,
                       value="NOK", bg='#020617', fg='#e5e7eb',
                       selectcolor='#020617', anchor='w').pack(fill='x')

        # ROI Kaydet butonu
        preview_btn = ttk.Button(left_panel, text="👁 ROI kaydet",
                                 style="Accent.TButton",
                                 command=self.save_roi_image)
        preview_btn.pack(pady=10, padx=20, fill='x')
        self.preview_btn = preview_btn


        # Sağ Panel (Kamera)
        right_panel = ttk.Frame(main_container, style="App.TFrame")
        right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.monitor_canvas = tk.Canvas(right_panel, bg='black')
        self.monitor_canvas.pack(fill='both', expand=True)
        
        # Başlat
        self.monitoring_active = True
        source = self.get_camera_source(self.current_project['camera_id'])
        if source == "HIKROBOT":
            if not self.hik_camera:
                self.hik_camera = HikrobotCamera()
                success, msg = self.hik_camera.open()
                if not success:
                    messagebox.showerror("Hata", f"Hikrobot Kamera açılamadı: {msg}")
                    return
            self.hik_camera.start()
        else:
            # Arka plan thread'i başlat (IP ve USB kamera için ortak)
            self._start_cam_reader(source)
        self.update_settings_loop()

    def update_settings_loop(self):
        """Ayarlar ekranı için izleme döngüsü"""
        self.update_validation()

    def update_validation(self):
        """Ayarlar ekranı validation döngüsü – sabit 1 sn aralık"""
        if not self.monitoring_active:
            return

        frame = None
        source = self.get_camera_source(self.current_project['camera_id'])

        if source == "HIKROBOT" and self.hik_camera:
            frame = self.hik_camera.get_frame()   # zaten thread'li
        else:
            frame = self._get_latest_frame()       # arka plan reader'dan al

        if frame is not None:
            self.frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ayarlar sayfasındaki dropdown'dan algoritmayı al
            display_name = self.algo_combo_settings.get()
            algorithm = self.algo_options_map.get(display_name, 'SSIM')
            
            try:
                threshold = float(self.threshold_var.get())
            except ValueError:
                threshold = 0.75
            
            roi_list = self.current_project.get('roi_list', [])
            ref_rois = getattr(self, 'reference_rois', [])
            
            if not roi_list:
                roi_list = [{'name': 'ROI_1', 'coords': self.current_project['roi']}]
                ref_rois = [self.reference_roi]
                
            # Tüm ROI'leri işle
            display_frame = frame_rgb.copy()
            overlay = display_frame.copy()
            
            total_score = 0
            passed_count = 0
            
            for idx, rc in enumerate(roi_list):
                x, y, w, h = rc['coords']
                current_roi = frame_rgb[y:y+h, x:x+w]
                ref_roi = ref_rois[idx] if idx < len(ref_rois) else self.reference_roi

                if algorithm in (
                    "ML_MODEL", "ML_MODEL_SIFT", "ML_MODEL_DAISY",
                    "ML_MODEL_HAAR", "ML_MODEL_CENSURE", "ML_MODEL_MBLBP",
                    "ML_MODEL_GLCM", "ML_MODEL_LBP", "ML_MODEL_GABOR",
                    "ML_MODEL_FISHER",
                ):
                    # ML modeli ile sınıflandır
                    roi_name = rc.get('name', f"ROI_{idx+1}")
                    model_key = (self.current_project.get('id'), roi_name, algorithm)
                    model = self.ml_models.get(model_key)

                    if model is None:
                        # DB'den model yolunu bul ve yükle
                        expected_type = (
                            "SIFT_SVM" if algorithm == "ML_MODEL_SIFT" else
                            "DAISY_SVM" if algorithm == "ML_MODEL_DAISY" else
                            "HAAR_SVM" if algorithm == "ML_MODEL_HAAR" else
                            "CENSURE_SVM" if algorithm == "ML_MODEL_CENSURE" else
                            "MBLBP_SVM" if algorithm == "ML_MODEL_MBLBP" else
                            "GLCM_SVM" if algorithm == "ML_MODEL_GLCM" else
                            "LBP_SVM" if algorithm == "ML_MODEL_LBP" else
                            "GABOR_SVM" if algorithm == "ML_MODEL_GABOR" else
                            "FISHER_SVM" if algorithm == "ML_MODEL_FISHER" else
                            "HOG_SVM"
                        )
                        self.cursor.execute(
                            "SELECT model_path, model_type FROM roi_ml_models WHERE project_id = ? AND roi_name = ?",
                            (self.current_project.get('id'), roi_name),
                        )
                        row = self.cursor.fetchone()
                        if row and row[1] == expected_type and os.path.exists(row[0]):
                            try:
                                model = joblib.load(row[0])
                                self.ml_models[model_key] = model
                            except Exception:
                                model = None

                    if model is not None:
                        roi_gray = cv2.cvtColor(current_roi, cv2.COLOR_RGB2GRAY)
                        roi_resized = cv2.resize(roi_gray, (64, 64))
                        if algorithm == "ML_MODEL_SIFT":
                            # SIFT öznitelik çıkarma (en iyi 50 keypoint)
                            feat = self._ml_extract_sift_features([roi_resized], n_keypoints=50)
                        elif algorithm == "ML_MODEL_DAISY":
                            feat = self._ml_extract_daisy_features([roi_resized], step=32, radius=16, rings=2, histograms=6, orientations=8)
                        elif algorithm == "ML_MODEL_HAAR":
                            feat = self._ml_extract_haar_features([roi_resized])
                        elif algorithm == "ML_MODEL_CENSURE":
                            feat = self._ml_extract_censure_features([roi_resized])
                        elif algorithm == "ML_MODEL_MBLBP":
                            feat = self._ml_extract_multiblock_lbp_features([roi_resized])
                        elif algorithm == "ML_MODEL_GLCM":
                            feat = self._ml_extract_glcm_features([roi_resized])
                        elif algorithm == "ML_MODEL_LBP":
                            feat = self._ml_extract_lbp_features([roi_resized], radius=3)
                        elif algorithm == "ML_MODEL_GABOR":
                            feat = self._ml_extract_gabor_features([roi_resized])
                        elif algorithm == "ML_MODEL_FISHER":
                            # Fisher için GMM yolunu params_json içinden alıp yükle
                            gmm_obj = None
                            try:
                                self.cursor.execute(
                                    "SELECT params_json FROM roi_ml_models WHERE project_id = ? AND roi_name = ?",
                                    (self.current_project.get('id'), roi_name),
                                )
                                prow = self.cursor.fetchone()
                                if prow and prow[0]:
                                    p = json.loads(prow[0])
                                    gmm_path = p.get("gmm_path")
                                    if gmm_path and os.path.exists(gmm_path):
                                        gmm_obj = joblib.load(gmm_path)
                            except Exception:
                                gmm_obj = None
                            if gmm_obj is None:
                                feat = np.zeros((1, 1), dtype=np.float32)
                            else:
                                feat = self._ml_extract_fisher_features([roi_resized], gmm=gmm_obj, n_components=16, n_keypoints=50)
                        else:
                            from skimage.feature import hog
                            feat = hog(
                                roi_resized,
                                orientations=9,
                                pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2),
                                block_norm='L2-Hys',
                                visualize=False,
                            ).reshape(1, -1)
                        pred = model.predict(feat)[0]
                        status = (str(pred).upper() == "OK")
                        score = 1.0 if status else 0.0
                    else:
                        status = False
                        score = 0.0
                else:
                    # Klasik benzerlik algoritmaları
                    score = self._score_roi(current_roi, ref_roi, algorithm)
                    status = score >= threshold
                
                if status: passed_count += 1
                total_score += score
                
                color = (0, 255, 0) if status else (255, 0, 0)
                
                # Saydam dolgu
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
                # Çerçeve
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Skor / sınıf metni
                if algorithm in (
                    "ML_MODEL", "ML_MODEL_SIFT", "ML_MODEL_DAISY",
                    "ML_MODEL_HAAR", "ML_MODEL_CENSURE", "ML_MODEL_MBLBP",
                    "ML_MODEL_GLCM", "ML_MODEL_LBP", "ML_MODEL_GABOR",
                    "ML_MODEL_FISHER",
                ):
                    label = f"{rc.get('name', f'ROI_{idx+1}')}: {'OK' if status else 'NOK'}"
                else:
                    label = f"{rc.get('name', f'ROI_{idx+1}')}: {score:.0%}"
                cv2.putText(display_frame, label, (x, max(y-5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            
            # Genel Durum Geri Bildirimi
            if algorithm in (
                "ML_MODEL", "ML_MODEL_SIFT", "ML_MODEL_DAISY",
                "ML_MODEL_HAAR", "ML_MODEL_CENSURE", "ML_MODEL_MBLBP",
                "ML_MODEL_GLCM", "ML_MODEL_LBP", "ML_MODEL_GABOR",
                "ML_MODEL_FISHER",
            ):
                algo_lbl = {
                    "ML_MODEL": "HOG-SVM",
                    "ML_MODEL_SIFT": "SIFT-SVM",
                    "ML_MODEL_DAISY": "DAISY-SVM",
                    "ML_MODEL_HAAR": "HAAR-SVM",
                    "ML_MODEL_CENSURE": "CENSURE-SVM",
                    "ML_MODEL_MBLBP": "MBLBP-SVM",
                    "ML_MODEL_GLCM": "GLCM-SVM",
                    "ML_MODEL_LBP": "LBP-SVM",
                    "ML_MODEL_GABOR": "GABOR-SVM",
                    "ML_MODEL_FISHER": "FISHER-SVM",
                }.get(algorithm, algorithm)
                self.score_label.config(text=f"ML({algo_lbl}): {passed_count}/{len(roi_list)} ROI OK")
            else:
                avg_score = total_score / len(roi_list) if roi_list else 0
                self.score_label.config(text=f"Ort. Skor: {avg_score:.2%}")
            
            all_ok = passed_count == len(roi_list)
            self.result_label.config(
                text=f"DURUM: {'OK' if all_ok else 'NG'} ({passed_count}/{len(roi_list)})",
                fg='#27ae60' if all_ok else '#e74c3c'
            )
            
            # Canvas'a göster
            self.display_monitoring_frame(display_frame)

        # Sabit 1 saniyelik döngü – kamera türünden bağımsız
        self.root.after(1000, self.update_validation)
    
    def control_screen(self):
        """Kontrol ekranı"""
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Proje seç
        self.cursor.execute('SELECT id, name FROM projects ORDER BY created_date DESC')
        projects = self.cursor.fetchall()
        
        if not projects:
            messagebox.showwarning("Uyarı", "Henüz kayıtlı proje yok!")
            self.show_main_menu()
            return
        
        # Proje seçim dialogu
        project_names = [p[1] for p in projects]
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Proje Seç")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Kontrol edilecek projeyi seçin:",
                font=('Arial', 12, 'bold')).pack(pady=20)
        
        selected_project = tk.StringVar()
        
        listbox = tk.Listbox(dialog, font=('Arial', 11), height=8)
        listbox.pack(pady=10, padx=20, fill='both', expand=True)
        
        for name in project_names:
            listbox.insert(tk.END, name)
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_project.set(project_names[selection[0]])
                dialog.destroy()
        
        tk.Button(dialog, text="Seç", command=on_select,
                 bg='#27ae60', fg='white', font=('Arial', 12)).pack(pady=10)
        
        dialog.wait_window()
        
        if not selected_project.get():
            self.show_main_menu()
            return
        
        # Seçilen projeyi yükle
        project_id = [p[0] for p in projects if p[1] == selected_project.get()][0]
        self.load_project(project_id)
        self.start_monitoring()
    
    def load_project(self, project_id):
        """Projeyi yükle (geriye uyumlu)"""
        self.cursor.execute('''
            SELECT name, camera_id, reference_image, roi_x, roi_y, roi_width, roi_height,
                   roi_list, algorithm, algo_threshold
            FROM projects WHERE id = ?
        ''', (project_id,))

        row = self.cursor.fetchone()

        # roi_list JSON varsa parse et, yoksa eski tek-ROI sütunlarından al
        roi_list_raw = row[7]
        if roi_list_raw:
            try:
                rois = json.loads(roi_list_raw)
                roi_tuples = [(r['x'], r['y'], r['w'], r['h']) for r in rois]
                roi_names  = [r.get('name', f'ROI_{i+1}') for i, r in enumerate(rois)]
            except Exception:
                rois = []
                roi_tuples = [(row[3], row[4], row[5], row[6])]
                roi_names  = ['ROI_1']
        else:
            roi_tuples = [(row[3], row[4], row[5], row[6])]
            roi_names  = ['ROI_1']

        self.current_project = {
            'id': project_id,
            'name': row[0],
            'camera_id': row[1],
            'roi': roi_tuples[0],         # geriye uyumluluk
            'roi_list': [{'name': roi_names[i], 'coords': roi_tuples[i]}
                         for i in range(len(roi_tuples))],
            'algorithm': row[8] or 'SSIM',
            'algo_threshold': row[9] if row[9] is not None else 0.75,
        }

        # Referans görüntüyü yükle
        img_array = np.frombuffer(row[2], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        self.reference_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # İlk ROI'nin referans kesimini oluştur (geriye uyumluluk)
        x, y, w, h = self.current_project['roi']
        self.reference_roi = self.reference_image[y:y+h, x:x+w]

        # Tüm ROI'lerin referans kesimlerini de oluştur
        self.reference_rois = []
        for rc in self.current_project['roi_list']:
            rx, ry, rw, rh = rc['coords']
            self.reference_rois.append(self.reference_image[ry:ry+rh, rx:rx+rw])


    
    def start_monitoring(self):
        """İzlemeyi başlat"""
        # Kontrol ekranı oluştur
        for widget in self.root.winfo_children():
            widget.destroy()

        monitor_frame = ttk.Frame(self.root, style="App.TFrame")
        monitor_frame.pack(expand=True, fill='both')
        
        # Üst panel
        top_panel = ttk.Frame(monitor_frame, style="Header.TFrame", height=100)
        top_panel.pack(fill='x')
        top_panel.pack_propagate(False)
        
        # Proje adı
        ttk.Label(top_panel, text=f"Proje: {self.current_project['name']}",
                  style="HeaderTitle.TLabel").pack(pady=10)
        
        # Durum
        self.status_label = tk.Label(top_panel, text="Başlatılıyor...", 
                                     font=('Segoe UI', 14),
                                     bg='#020617', fg='#e5e7eb')
        self.status_label.pack()
        
        # Durdur butonu
        stop_btn = ttk.Button(top_panel, text="⏹ Durdur ve Ana Menü", 
                              command=self.stop_monitoring,
                              style="Danger.TButton")
        stop_btn.place(x=10, y=35)
        
        # Canvas
        self.monitor_canvas = tk.Canvas(monitor_frame, bg='black')
        self.monitor_canvas.pack(expand=True, fill='both', padx=10, pady=10)
        
        # İzlemeyi başlat
        self.monitoring_active = True
        source = self.get_camera_source(self.current_project['camera_id'])
        if source == "HIKROBOT":
            if not self.hik_camera:
                self.hik_camera = HikrobotCamera()
                success, msg = self.hik_camera.open()
                if not success:
                    messagebox.showerror("Hata", f"Hikrobot Kamera açılamadı: {msg}")
                    return
            ok = self.hik_camera.start()
            if not ok:
                messagebox.showerror("Hata", "Hikrobot Kamera başlatılamadı (StartGrabbing başarısız).")
                return
        else:
            # Arka plan thread'i başlat (IP ve USB kamera için ortak)
            self._start_cam_reader(source)
                
        self.update_monitoring()
    
    def show_roi_preview(self):
        """ROI bölgesini yeni pencerede göster"""
        if self.frame is None or self.current_project is None:
            messagebox.showwarning("Uyarı", "Lütfen önce görüntü çekin ve ROI seçin!")
            return
        
        # ROI bölgesini çıkar
        x, y, w, h = self.current_project['roi']
        cropped_image = self.frame[y:y+h, x:x+w]
        
        # BGR formatına çevir (OpenCV için)
        cropped_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        
        # Yeni pencerede göster
        window_name = 'ROI Önizleme - ESC veya herhangi bir tuşa basın'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 300)
        cv2.imshow(window_name, cropped_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()

    def save_roi_image(self):
        """ROI görüntüsünü dosyaya kaydet"""
        if self.frame is None or self.current_project is None:
            messagebox.showwarning("Uyarı", "Lütfen önce görüntü çekin ve ROI seçin!")
            return

        # OK / Not OK seçimi zorunlu
        label_value = getattr(self, "roi_label_var", None).get() if hasattr(self, "roi_label_var") else ""
        if label_value not in ("OK", "NOK"):
            messagebox.showwarning("Uyarı", "Lütfen önce ROI için OK / Not OK seçiniz!")
            return
        
        # Proje adı: ayarlar sayfasında seçilen mevcut proje adı
        project_name = None
        if self.current_project:
            project_name = self.current_project.get('name')

        if not project_name:
            return

        # Kaydedilecek ROI listesi (çoklu ROI varsa hepsi)
        roi_items = []
        if getattr(self, 'roi_list', None):
            roi_items = list(self.roi_list)
        if not roi_items:
            roi_items = list(self.current_project.get('roi_list', []) or [])
        if not roi_items:
            # Geriye uyumluluk: projede tek ROI varsa onu kaydet
            roi_items = [{'name': 'ROI_1', 'coords': tuple(self.current_project.get('roi', (0, 0, 0, 0)))}]

        if not roi_items:
            messagebox.showwarning("Uyarı", "Kaydedilecek ROI bulunamadı!")
            return

        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        # Dosya adı: ilk ROI ismi + benzersiz sayı (her ROI için farklı)
        unique_base = int(datetime.datetime.now().timestamp() * 1000)

        saved_files = []
        try:
            h_img, w_img = frame_rgb.shape[:2]
            for idx, roi_item in enumerate(roi_items):
                x, y, w, h = roi_item.get('coords', (0, 0, 0, 0))
                if w <= 0 or h <= 0:
                    continue

                # Sınırlar dışına taşmayı engelle
                x = max(0, min(int(x), w_img - 1))
                y = max(0, min(int(y), h_img - 1))
                w = max(1, min(int(w), w_img - x))
                h = max(1, min(int(h), h_img - y))

                current_roi = frame_rgb[y:y+h, x:x+w]
                if current_roi.size == 0:
                    continue

                cropped_bgr = cv2.cvtColor(current_roi, cv2.COLOR_RGB2BGR)
                unique_num = unique_base + idx
                roi_name = self._safe_name(roi_item.get('name') or f"ROI_{idx+1}")
                # Klasör yapısı: roi_images/ProjeAdı/ROIAdı/OK|NOK
                project_folder = os.path.join('roi_images', self._safe_name(project_name))
                roi_folder = os.path.join(project_folder, roi_name)
                sub_folder = os.path.join(roi_folder, label_value)

                for folder in (project_folder, roi_folder, sub_folder):
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)

                filename = os.path.join(sub_folder, f"{roi_name}_{unique_num}.jpg")
                cv2.imwrite(filename, cropped_bgr)
                saved_files.append(filename)

            if not saved_files:
                messagebox.showwarning("Uyarı", "ROI kaydedilemedi (geçerli ROI bulunamadı).")
                return

            messagebox.showinfo(
                "Başarılı",
                f"{len(saved_files)} ROI kaydedildi.\nKlasör kökü:\nroi_images/{self._safe_name(project_name)}/"
            )

        except Exception as e:
            messagebox.showerror("Hata", f"Kaydetme hatası: {str(e)}")
    
    def _ml_load_images(self, data_dir, image_size=(128, 128)):
        images = []
        labels = []
        if not os.path.isdir(data_dir):
            return np.array([]), np.array([])
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if not os.path.isdir(label_path):
                continue
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label_dir)
        return np.array(images), np.array(labels)

    def _ml_extract_hog_features(self, images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
        from skimage.feature import hog
        feats = []
        for img in images:
            f = hog(
                img,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys',
                visualize=False,
            )
            feats.append(f)
        return np.array(feats)

    def _ml_extract_sift_features(self, images, n_keypoints=50):
        """Her görüntüden en iyi n_keypoints adet SIFT descriptoru çikar (128 boyut her biri).
        Eksik anahtar noktalar sıfırla doldurulur; böylece sabit boyutlu bir vektör elde edilir."""
        sift = cv2.SIFT_create()
        feat_size = n_keypoints * 128  # SIFT descriptor = 128 boyut
        feats = []
        for img in images:
            kps, des = sift.detectAndCompute(img, None)
            if des is None or len(des) == 0:
                feats.append(np.zeros(feat_size, dtype=np.float32))
                continue
            # Yanıt değerine göre en iyi n_keypoints noktayı seç
            kps_sorted = sorted(zip(kps, des), key=lambda x: -x[0].response)
            des_sorted = np.array([d for _, d in kps_sorted], dtype=np.float32)
            if len(des_sorted) >= n_keypoints:
                des_top = des_sorted[:n_keypoints]
            else:
                pad = np.zeros((n_keypoints - len(des_sorted), 128), dtype=np.float32)
                des_top = np.vstack([des_sorted, pad])
            feats.append(des_top.flatten())
        return np.array(feats, dtype=np.float32)

    def _ml_extract_daisy_features(self, images, step=32, radius=16, rings=2, histograms=6, orientations=8):
        """DAISY özniteliklerini çıkar (sabit boyutlu, flatten)."""
        from skimage.feature import daisy
        feats = []
        for img in images:
            f = daisy(
                img,
                step=step,
                radius=radius,
                rings=rings,
                histograms=histograms,
                orientations=orientations,
            )
            feats.append(f.flatten())
        return np.array(feats)

    def _ml_extract_haar_features(self, images):
        """Haar-like öznitelikleri (sabit pencere)"""
        from skimage.transform import integral_image
        from skimage.feature import haar_like_feature
        feats = []
        for img in images:
            ii = integral_image(img)
            feature_types = ['type-2-x', 'type-2-y']
            f = haar_like_feature(ii, 0, 0, 16, 16, feature_types)
            feats.append(f)
        return np.array(feats)

    def _ml_extract_censure_features(self, images):
        """CENSURE -> ilk 50 keypoint (r,c,scale) => 150 boyut"""
        from skimage.feature import CENSURE
        feats = []
        for img in images:
            detector = CENSURE()
            detector.detect(img)
            kp = detector.keypoints
            scales = detector.scales
            f = np.zeros(150, dtype=np.float32)
            n = min(len(kp), 50)
            for i in range(n):
                f[i*3] = kp[i, 0]
                f[i*3+1] = kp[i, 1]
                f[i*3+2] = scales[i]
            feats.append(f)
        return np.array(feats, dtype=np.float32)

    def _ml_extract_multiblock_lbp_features(self, images):
        """Multi-block LBP -> 64x64 üzerinde sabit grid"""
        from skimage.feature import multiblock_lbp
        feats = []
        for img in images:
            f = []
            h, w = img.shape
            for r in range(0, h-10, 20):
                for c in range(0, w-10, 20):
                    f.append(multiblock_lbp(img, r, c, 10, 10))
            feats.append(np.array(f, dtype=np.float32))
        return np.array(feats, dtype=np.float32)

    def _ml_extract_glcm_features(self, images):
        """GLCM props (6) x angles(4) => 24"""
        from skimage.feature import graycomatrix, graycoprops
        feats = []
        for img in images:
            glcm = graycomatrix(
                img,
                distances=[5],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True,
            )
            f = []
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                f.extend(graycoprops(glcm, prop).ravel().tolist())
            feats.append(np.array(f, dtype=np.float32))
        return np.array(feats, dtype=np.float32)

    def _ml_extract_lbp_features(self, images, radius=3):
        """LBP histogram (uniform)"""
        from skimage.feature import local_binary_pattern
        feats = []
        n_points = 8 * radius
        for img in images:
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            feats.append(hist.astype(np.float32))
        return np.array(feats, dtype=np.float32)

    def _ml_extract_gabor_features(self, images):
        """Gabor kernel bank: mean+var"""
        from skimage.filters import gabor_kernel
        from scipy import ndimage as ndi
        kernels = []
        for theta in range(4):
            t = theta / 4.0 * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernels.append(np.real(gabor_kernel(frequency, theta=t, sigma_x=sigma, sigma_y=sigma)))
        feats = []
        for img in images:
            f = []
            for k in kernels:
                filtered = ndi.convolve(img, k, mode='wrap')
                f.append(float(filtered.mean()))
                f.append(float(filtered.var()))
            feats.append(np.array(f, dtype=np.float32))
        return np.array(feats, dtype=np.float32)

    def _ml_extract_fisher_features(self, images, gmm=None, n_components=16, n_keypoints=50):
        """ORB descriptors -> Fisher Vector (GMM gerekir). Eğer gmm None ise sadece descriptor listesi döndürür."""
        from skimage.feature import ORB, fisher_vector
        all_desc = []
        for img in images:
            orb = ORB(n_keypoints=n_keypoints)
            try:
                orb.detect_and_extract(img)
                des = orb.descriptors
            except Exception:
                des = None
            if des is None or len(des) == 0:
                all_desc.append(None)
            else:
                all_desc.append(des.astype(np.float32))

        if gmm is None:
            return all_desc

        feats = []
        feat_len = None
        for des in all_desc:
            if des is None:
                if feat_len is None:
                    feats.append(None)
                else:
                    feats.append(np.zeros(feat_len, dtype=np.float32))
                continue
            fv = fisher_vector(des, gmm).astype(np.float32)
            if feat_len is None:
                feat_len = fv.shape[0]
            feats.append(fv)
        if feat_len is None:
            return np.zeros((len(images), 1), dtype=np.float32)
        # doldur
        out = []
        for f in feats:
            if f is None:
                out.append(np.zeros(feat_len, dtype=np.float32))
            else:
                out.append(f)
        return np.vstack(out)

    def train_ml_models(self):
        """Seçili proje için her ROI klasöründen ML modeli eğit (HOG/SIFT/DAISY + SVM)."""
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.mixture import GaussianMixture

        if not self.current_project:
            messagebox.showwarning("Uyarı", "Önce bir proje seçiniz!")
            return

        project_name = self.current_project.get('name')
        project_id = self.current_project.get('id')
        project_root = os.path.join("roi_images", self._safe_name(project_name) if hasattr(self, "_safe_name") else project_name)

        if not os.path.isdir(project_root):
            messagebox.showwarning("Uyarı", f"Veri kümesi klasörü bulunamadı:\n{project_root}")
            return

        roi_list = self.current_project.get('roi_list', [])
        if not roi_list:
            messagebox.showwarning("Uyarı", "Projede tanımlı ROI bulunamadı!")
            return

        # Hangi algo seçili?
        display_name = self.algo_combo_settings.get() if hasattr(self, 'algo_combo_settings') else ''
        algo_internal = self.algo_options_map.get(display_name, 'ML_MODEL') if hasattr(self, 'algo_options_map') else 'ML_MODEL'
        use_sift = (algo_internal == 'ML_MODEL_SIFT')
        use_daisy = (algo_internal == 'ML_MODEL_DAISY')
        use_haar = (algo_internal == 'ML_MODEL_HAAR')
        use_censure = (algo_internal == 'ML_MODEL_CENSURE')
        use_mblbp = (algo_internal == 'ML_MODEL_MBLBP')
        use_glcm = (algo_internal == 'ML_MODEL_GLCM')
        use_lbp = (algo_internal == 'ML_MODEL_LBP')
        use_gabor = (algo_internal == 'ML_MODEL_GABOR')
        use_fisher = (algo_internal == 'ML_MODEL_FISHER')

        summary_lines = []

        for idx, rc in enumerate(roi_list):
            roi_name = rc.get('name', f"ROI_{idx+1}")
            roi_folder = os.path.join(project_root, roi_name)

            # compare_features_SVM.py ile uyumlu ve hızlı eğitim için 64x64
            images, labels = self._ml_load_images(roi_folder, image_size=(64, 64))
            if images.size == 0 or labels.size == 0:
                summary_lines.append(f"{roi_name}: veri yok, atlandı.")
                continue

            try:
                if use_sift:
                    X = self._ml_extract_sift_features(images, n_keypoints=50)
                    model_type_str = "SIFT_SVM"
                    model_suffix = "SIFT_SVM"
                    params = {
                        "kernel": "rbf",
                        "probability": True,
                        "image_size": (64, 64),
                        "sift_n_keypoints": 50,
                        "feature_size": 50 * 128,
                    }
                elif use_daisy:
                    X = self._ml_extract_daisy_features(images, step=32, radius=16, rings=2, histograms=6, orientations=8)
                    model_type_str = "DAISY_SVM"
                    model_suffix = "DAISY_SVM"
                    params = {
                        "kernel": "linear",
                        "probability": True,
                        "image_size": (64, 64),
                        "daisy_step": 32,
                        "daisy_radius": 16,
                        "daisy_rings": 2,
                        "daisy_histograms": 6,
                        "daisy_orientations": 8,
                    }
                else:
                    if use_haar:
                        X = self._ml_extract_haar_features(images)
                        model_type_str = "HAAR_SVM"
                        model_suffix = "HAAR_SVM"
                        params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
                    elif use_censure:
                        X = self._ml_extract_censure_features(images)
                        model_type_str = "CENSURE_SVM"
                        model_suffix = "CENSURE_SVM"
                        params = {"kernel": "linear", "probability": True, "image_size": (64, 64), "censure_keypoints": 50}
                    elif use_mblbp:
                        X = self._ml_extract_multiblock_lbp_features(images)
                        model_type_str = "MBLBP_SVM"
                        model_suffix = "MBLBP_SVM"
                        params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
                    elif use_glcm:
                        X = self._ml_extract_glcm_features(images)
                        model_type_str = "GLCM_SVM"
                        model_suffix = "GLCM_SVM"
                        params = {"kernel": "linear", "probability": True, "image_size": (64, 64), "glcm_distance": 5}
                    elif use_lbp:
                        X = self._ml_extract_lbp_features(images, radius=3)
                        model_type_str = "LBP_SVM"
                        model_suffix = "LBP_SVM"
                        params = {"kernel": "linear", "probability": True, "image_size": (64, 64), "lbp_radius": 3}
                    elif use_gabor:
                        X = self._ml_extract_gabor_features(images)
                        model_type_str = "GABOR_SVM"
                        model_suffix = "GABOR_SVM"
                        params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
                    elif use_fisher:
                        # Fisher Vector: önce tüm descriptorları topla -> GMM eğit -> FV çıkar
                        desc_list = self._ml_extract_fisher_features(images, gmm=None, n_components=16, n_keypoints=50)
                        desc_all = [d for d in desc_list if d is not None]
                        if not desc_all:
                            summary_lines.append(f"{roi_name}: Fisher için ORB descriptor bulunamadı.")
                            continue
                        desc_stack = np.vstack(desc_all)
                        gmm = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
                        gmm.fit(desc_stack)
                        X = self._ml_extract_fisher_features(images, gmm=gmm, n_components=16, n_keypoints=50)
                        model_type_str = "FISHER_SVM"
                        model_suffix = "FISHER_SVM"
                        params = {"kernel": "linear", "probability": True, "image_size": (64, 64), "fisher_gmm_components": 16, "orb_keypoints": 50}
                    else:
                        X = self._ml_extract_hog_features(images)
                        model_type_str = "HOG_SVM"
                        model_suffix = ""
                        params = {
                            "kernel": "linear",
                            "probability": True,
                            "image_size": (64, 64),
                            "hog_pixels_per_cell": (8, 8),
                            "hog_cells_per_block": (2, 2),
                            "hog_orientations": 9,
                        }

                y = labels

                if len(np.unique(y)) < 2:
                    summary_lines.append(f"{roi_name}: Yeterli sınıf yok (en az 2 sınıf gerekli).")
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                kernel = 'rbf' if use_sift else 'linear'
                svm_model = SVC(kernel=kernel, probability=True)
                svm_model.fit(X_train, y_train)

                y_pred = svm_model.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred))
                params["accuracy"] = acc

                safe_proj = self._safe_name(project_name)
                safe_roi  = self._safe_name(roi_name)
                if model_suffix:
                    model_filename = f"{safe_proj}_{safe_roi}_{model_suffix}.pkl"
                else:
                    model_filename = f"{safe_proj}_{safe_roi}.pkl"
                model_path = os.path.join(roi_folder, model_filename)
                joblib.dump(svm_model, model_path)

                # Fisher için GMM de kaydet
                if use_fisher:
                    gmm_filename = f"{safe_proj}_{safe_roi}_FISHER_GMM.pkl"
                    gmm_path = os.path.join(roi_folder, gmm_filename)
                    joblib.dump(gmm, gmm_path)
                    params["gmm_path"] = gmm_path

                self.cursor.execute(
                    '''
                    INSERT INTO roi_ml_models (project_id, roi_name, model_type, model_path, params_json, created_date, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, roi_name) DO UPDATE SET
                        model_type=excluded.model_type,
                        model_path=excluded.model_path,
                        params_json=excluded.params_json,
                        updated_date=excluded.updated_date
                    ''',
                    (
                        project_id,
                        roi_name,
                        model_type_str,
                        model_path,
                        json.dumps(params),
                        datetime.datetime.now().isoformat(),
                        datetime.datetime.now().isoformat(),
                    ),
                )
                self.conn.commit()

                summary_lines.append(f"{roi_name} [{model_type_str}]: eğitim OK, doğruluk={acc:.2%}")
            except Exception as e:
                summary_lines.append(f"{roi_name}: eğitim hatası: {e}")

        if summary_lines:
            messagebox.showinfo("ML Eğitim Sonucu", "\n".join(summary_lines))
        else:
            messagebox.showwarning("ML Eğitim", "Hiç ROI için eğitim yapılamadı.")

    def _score_roi(self, current_roi, ref_roi, algorithm):
        """Seçilen algoritmaya göre benzerlik skoru hesapla (0-1)"""
        algo = (algorithm or 'SSIM').upper()
        try:
            if current_roi.size == 0 or ref_roi.size == 0:
                return 0.0
            if current_roi.shape != ref_roi.shape:
                ref_roi = cv2.resize(ref_roi, (current_roi.shape[1], current_roi.shape[0]))

            if algo == 'SSIM':
                if SKIMAGE_AVAILABLE:
                    g1 = cv2.cvtColor(current_roi, cv2.COLOR_RGB2GRAY)
                    g2 = cv2.cvtColor(ref_roi, cv2.COLOR_RGB2GRAY)
                    score = ssim(g1, g2, data_range=255)
                    return max(0.0, float(score))
                else:
                    diff = cv2.absdiff(current_roi, ref_roi)
                    mse = np.mean(diff ** 2)
                    return max(0.0, 1.0 - (mse / 65025.0))

            elif algo == 'INDUSTRIAL_AI':
                # test_similarity_formulasyon.py -> Industrial_IV3_AI
                c_norm = (current_roi - np.min(current_roi)) / (np.max(current_roi) - np.min(current_roi) + 1e-6)
                r_norm = (ref_roi - np.min(ref_roi)) / (np.max(ref_roi) - np.min(ref_roi) + 1e-6)
                mse = np.mean(np.square(c_norm - r_norm))
                score = np.exp(-mse * 10)
                return float(score)

            elif algo == 'ADVANCED_ENGINE':
                # test_similarity_formulasyon.py -> Advanced_IV3_Engine2
                def _feat(img):
                    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    h, w = img_g.shape
                    gh, gw = h // 8, w // 8
                    f = []
                    for i in range(8):
                        for j in range(8):
                            sec = img_g[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                            f.append(np.std(sec) + np.mean(np.abs(np.gradient(sec))))
                    return np.array(f)
                f1 = _feat(current_roi)
                f2 = _feat(ref_roi)
                err = np.mean(np.abs(f1 - f2) / (f2 + 1e-6))
                return float(max(0, 1 - err))

            elif algo == 'ALIGN_SSIM':
                # shif_rotate/image_alignment_compare_ROI.py logic
                try:
                    gray1 = cv2.cvtColor(current_roi, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(ref_roi, cv2.COLOR_RGB2GRAY)
                    sift = cv2.SIFT_create()
                    kp1, des1 = sift.detectAndCompute(gray1, None)
                    kp2, des2 = sift.detectAndCompute(gray2, None)
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)
                    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                    if len(good) > 4:
                        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
                        aligned = cv2.warpPerspective(current_roi, M, (ref_roi.shape[1], ref_roi.shape[0]))
                    else:
                        aligned = current_roi
                    if SKIMAGE_AVAILABLE:
                        g1 = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
                        g2 = cv2.cvtColor(ref_roi, cv2.COLOR_RGB2GRAY)
                        score = ssim(g1, g2, data_range=255)
                        return float(score)
                    return 0.5 # fallback if alignment fails or skimage missing
                except: return 0.0

            elif algo == 'TEMPLATE':
                return self.method_template_matching(current_roi, ref_roi)

            elif algo == 'HISTOGRAM':
                return self.method_histogram(current_roi, ref_roi)

            elif algo == 'FOURIER':
                return self.method_fourier(current_roi, ref_roi)

            elif algo == 'WAVELET':
                return self.method_wavelet(current_roi, ref_roi)

            elif algo == 'ORB':
                return self.method_brute_force_matching(current_roi, ref_roi)

            else:
                return self.check_similarity(current_roi, ref_roi)
        except Exception as e:
            print(f"Score error ({algo}): {e}")
            return 0.0

    def update_monitoring(self):
        """İzleme döngüsü – çoklu ROI + proje algoritması – sabit 1 sn aralık"""
        if not self.monitoring_active:
            return

        frame = None
        source = self.get_camera_source(self.current_project['camera_id'])
        if source == "HIKROBOT" and self.hik_camera:
            frame = self.hik_camera.get_frame()
        else:
            frame = self._get_latest_frame()  # arka plan reader'dan al

        if frame is not None:
            self.frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame_rgb.copy()

            algorithm  = self.current_project.get('algorithm', 'SSIM')
            threshold  = float(self.current_project.get('algo_threshold', 0.75))
            roi_list   = self.current_project.get('roi_list', [])
            ref_rois   = getattr(self, 'reference_rois', [])

            # Geriye uyumluluk: roi_list yoksa tek ROI kullan
            if not roi_list:
                roi_list = [{'name': 'ROI_1', 'coords': self.current_project['roi']}]
                ref_rois = [self.reference_roi]

            results = []   # (passed: bool, score: float, coords, name)

            for idx, rc in enumerate(roi_list):
                x, y, w, h = rc['coords']
                name = rc.get('name', f'ROI_{idx+1}')

                # Görüntüden mevcut ROI kesimi
                fh, fw = frame_rgb.shape[:2]
                x  = max(0, min(x, fw-1))
                y  = max(0, min(y, fh-1))
                x2 = max(0, min(x+w, fw))
                y2 = max(0, min(y+h, fh))
                current_roi = frame_rgb[y:y2, x:x2]

                # Referans ROI
                ref_roi = ref_rois[idx] if idx < len(ref_rois) else self.reference_roi

                if algorithm in (
                    'ML_MODEL', 'ML_MODEL_SIFT', 'ML_MODEL_DAISY',
                    'ML_MODEL_HAAR', 'ML_MODEL_CENSURE', 'ML_MODEL_MBLBP',
                    'ML_MODEL_GLCM', 'ML_MODEL_LBP', 'ML_MODEL_GABOR',
                    'ML_MODEL_FISHER',
                ):
                    # ML modeli ile sınıflandır
                    model_key = (self.current_project.get('id'), name, algorithm)
                    model = self.ml_models.get(model_key)
                    if model is None:
                        expected_type = (
                            "SIFT_SVM" if algorithm == "ML_MODEL_SIFT" else
                            "DAISY_SVM" if algorithm == "ML_MODEL_DAISY" else
                            "HAAR_SVM" if algorithm == "ML_MODEL_HAAR" else
                            "CENSURE_SVM" if algorithm == "ML_MODEL_CENSURE" else
                            "MBLBP_SVM" if algorithm == "ML_MODEL_MBLBP" else
                            "GLCM_SVM" if algorithm == "ML_MODEL_GLCM" else
                            "LBP_SVM" if algorithm == "ML_MODEL_LBP" else
                            "GABOR_SVM" if algorithm == "ML_MODEL_GABOR" else
                            "FISHER_SVM" if algorithm == "ML_MODEL_FISHER" else
                            "HOG_SVM"
                        )
                        self.cursor.execute(
                            "SELECT model_path, model_type FROM roi_ml_models WHERE project_id = ? AND roi_name = ?",
                            (self.current_project.get('id'), name),
                        )
                        row = self.cursor.fetchone()
                        if row and row[1] == expected_type and os.path.exists(row[0]):
                            try:
                                model = joblib.load(row[0])
                                self.ml_models[model_key] = model
                            except Exception:
                                model = None
                    if model is not None:
                        roi_gray = cv2.cvtColor(current_roi, cv2.COLOR_RGB2GRAY)
                        roi_resized = cv2.resize(roi_gray, (64, 64))
                        if algorithm == 'ML_MODEL_SIFT':
                            feat = self._ml_extract_sift_features([roi_resized], n_keypoints=50)
                        elif algorithm == 'ML_MODEL_DAISY':
                            feat = self._ml_extract_daisy_features([roi_resized], step=32, radius=16, rings=2, histograms=6, orientations=8)
                        elif algorithm == 'ML_MODEL_HAAR':
                            feat = self._ml_extract_haar_features([roi_resized])
                        elif algorithm == 'ML_MODEL_CENSURE':
                            feat = self._ml_extract_censure_features([roi_resized])
                        elif algorithm == 'ML_MODEL_MBLBP':
                            feat = self._ml_extract_multiblock_lbp_features([roi_resized])
                        elif algorithm == 'ML_MODEL_GLCM':
                            feat = self._ml_extract_glcm_features([roi_resized])
                        elif algorithm == 'ML_MODEL_LBP':
                            feat = self._ml_extract_lbp_features([roi_resized], radius=3)
                        elif algorithm == 'ML_MODEL_GABOR':
                            feat = self._ml_extract_gabor_features([roi_resized])
                        elif algorithm == 'ML_MODEL_FISHER':
                            gmm_obj = None
                            try:
                                self.cursor.execute(
                                    "SELECT params_json FROM roi_ml_models WHERE project_id = ? AND roi_name = ?",
                                    (self.current_project.get('id'), name),
                                )
                                prow = self.cursor.fetchone()
                                if prow and prow[0]:
                                    p = json.loads(prow[0])
                                    gmm_path = p.get("gmm_path")
                                    if gmm_path and os.path.exists(gmm_path):
                                        gmm_obj = joblib.load(gmm_path)
                            except Exception:
                                gmm_obj = None
                            if gmm_obj is None:
                                feat = np.zeros((1, 1), dtype=np.float32)
                            else:
                                feat = self._ml_extract_fisher_features([roi_resized], gmm=gmm_obj, n_components=16, n_keypoints=50)
                        else:
                            from skimage.feature import hog
                            feat = hog(
                                roi_resized,
                                orientations=9,
                                pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2),
                                block_norm='L2-Hys',
                                visualize=False,
                            ).reshape(1, -1)
                        pred = model.predict(feat)[0]
                        passed = (str(pred).upper() == 'OK')
                        score = 1.0 if passed else 0.0
                    else:
                        passed = False
                        score = 0.0
                else:
                    score   = self._score_roi(current_roi, ref_roi, algorithm)
                    passed  = score >= threshold

                results.append((passed, score, (x, y, x2-x, y2-y), name))

            # ── Her ROI için saydam overlay çiz ──────────────
            overlay = display_frame.copy()
            for passed, score, (x, y, w, h), name in results:
                color = (0, 220, 80) if passed else (220, 40, 40)

                # Saydam dolgu
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)

            cv2.addWeighted(overlay, 0.30, display_frame, 0.70, 0, display_frame)

            for passed, score, (x, y, w, h), name in results:
                color = (0, 220, 80) if passed else (220, 40, 40)

                # Kenar çizgisi
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)

                # ROI ismi + skor / sınıf
                if algorithm in (
                    'ML_MODEL', 'ML_MODEL_SIFT', 'ML_MODEL_DAISY',
                    'ML_MODEL_HAAR', 'ML_MODEL_CENSURE', 'ML_MODEL_MBLBP',
                    'ML_MODEL_GLCM', 'ML_MODEL_LBP', 'ML_MODEL_GABOR',
                    'ML_MODEL_FISHER',
                ):
                    label = f"{name}: {'OK' if passed else 'NOK'}"
                else:
                    label = f"{name}: {score:.0%}"
                cv2.putText(display_frame, label, (x, max(y-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                if passed:
                    # ✓ işareti
                    cx, cy = x + w//2, y + h//2
                    cv2.drawMarker(display_frame, (cx, cy),
                                   color, cv2.MARKER_TILTED_CROSS, min(w, h)//2, 4)
                else:
                    # X çarpı işareti
                    m = 12
                    cv2.line(display_frame, (x+m, y+m), (x+w-m, y+h-m), color, 5)
                    cv2.line(display_frame, (x+w-m, y+m), (x+m, y+h-m), color, 5)

            # ── Tüm ROI'ler geçti mi? ─────────────────────────
            all_ok   = all(r[0] for r in results)
            ng_count = sum(1 for r in results if not r[0])

            if all_ok:
                # Tam ekran yeşil çerçeve
                fh, fw = display_frame.shape[:2]
                border = 8
                cv2.rectangle(display_frame, (border, border),
                              (fw-border, fh-border), (0, 220, 80), border*2)
                # Büyük OK yazısı
                text = "OK"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 5, 8)
                tx = (fw - tw) // 2
                ty = (fh + th) // 2
                cv2.putText(display_frame, text, (tx, ty),
                            cv2.FONT_HERSHEY_DUPLEX, 5, (0, 220, 80), 8, cv2.LINE_AA)
                status_text  = f"✓ TÜM ROI'LER TAMAM ({len(results)}/{len(results)})"
                status_color = '#27ae60'
            else:
                status_text  = f"✗ {ng_count} ROI HATALI  ({len(results)-ng_count}/{len(results)} geçti)"
                status_color = '#e74c3c'

            self.display_monitoring_frame(display_frame)
            self.status_label.config(text=status_text, fg=status_color)

            # DB kaydı (ilk ROI skoru veya ML için 1/0)
            first_score = results[0][1] if results else 0.0
            self.log_monitoring(first_score, "OK" if all_ok else "FAIL")

        # Sabit 1 saniyelik döngü – kamera türünden bağımsız
        self.root.after(1000, self.update_monitoring)


    
    def check_similarity(self, img1, img2):
        """İki görüntünün benzerliğini kontrol et (Template Matching + Histogram)"""
        try:
            # Boyutları eşitle
            if img1.shape != img2.shape:
                img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
            
            # Gri tonlamaya çevir
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            

            # 1. Yöntem: Template Matching
            result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
            _, match_score, _, _ = cv2.minMaxLoc(result)
            
            # 2. Yöntem: Histogram Karşılaştırma
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            
            # Ortalama skor
            final_score = (match_score + hist_score) / 2
            
            return final_score
            
        except Exception as e:
            print(f"Similarity error: {e}")
            return 0.0
    
    def display_monitoring_frame(self, frame):
        """İzleme frame'ini göster"""
        h, w = frame.shape[:2]
        canvas_w = self.monitor_canvas.winfo_width()
        canvas_h = self.monitor_canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            scale = min(canvas_w/w, canvas_h/h) * 0.95
            new_w, new_h = int(w*scale), int(h*scale)
            
            img_resized = cv2.resize(frame, (new_w, new_h))
            img_pil = Image.fromarray(img_resized)
            self.monitor_photo = ImageTk.PhotoImage(img_pil)
            
            self.monitor_canvas.delete('all')
            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            self.monitor_canvas.create_image(x, y, anchor='nw', image=self.monitor_photo)
    
    def log_monitoring(self, similarity, status):
        """İzleme kaydı tut"""
        try:
            now = datetime.datetime.now().isoformat()
            self.cursor.execute('''
                INSERT INTO monitoring_logs (project_id, timestamp, status, similarity_score)
                VALUES (?, ?, ?, ?)
            ''', (self.current_project['id'], now, status, similarity))
            self.conn.commit()
        except Exception as e:
            print(f"Log error: {e}")
    
    def stop_monitoring(self):
        """İzlemeyi durdur"""
        self.monitoring_active = False
        # Arka plan kamera okuma thread'ini durdur
        self._stop_cam_reader()
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.hik_camera:
            self.hik_camera.stop()
        self.show_main_menu()
    
    def method_template_matching(self, img_roi, img_ref):
        """1. Template Matching Yöntemi"""
        try:
            # Boyut eşitle
            if img_roi.shape != img_ref.shape:
                img_roi = cv2.resize(img_roi, (img_ref.shape[1], img_ref.shape[0]))
                
            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
            gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
            
            result = cv2.matchTemplate(gray_roi, gray_ref, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val
        except Exception:
            return 0.0

    def method_histogram(self, img_roi, img_ref):
        """2. Histogram Karşılaştırma Yöntemi"""
        try:
            # Boyut eşitle
            if img_roi.shape != img_ref.shape:
                img_roi = cv2.resize(img_roi, (img_ref.shape[1], img_ref.shape[0]))
                
            hsv_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2HSV)
            hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2HSV)
            
            # H-S histogram hesapla
            hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
            hist_ref = cv2.calcHist([hsv_ref], [0, 1], None, [180, 256], [0, 180, 0, 256])
            
            cv2.normalize(hist_roi, hist_roi, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_ref, hist_ref, 0, 1, cv2.NORM_MINMAX)
            
            # Correlation metodu (1.0 = tam eşleşme)
            score = cv2.compareHist(hist_roi, hist_ref, cv2.HISTCMP_CORREL)
            return max(0.0, score)
        except Exception:
            return 0.0

    def method_fourier(self, img_roi, img_ref):
        """3. Fourier Transform Yöntemi"""
        try:
            # Griye çevir ve boyut eşitle
            if img_roi.shape != img_ref.shape:
                img_roi = cv2.resize(img_roi, (img_ref.shape[1], img_ref.shape[0]))
                
            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
            gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
            
            # FFT hesapla
            f_roi = np.fft.fft2(gray_roi)
            fshift_roi = np.fft.fftshift(f_roi)
            magnitude_roi = 20 * np.log(np.abs(fshift_roi) + 1)
            
            f_ref = np.fft.fft2(gray_ref)
            fshift_ref = np.fft.fftshift(f_ref)
            magnitude_ref = 20 * np.log(np.abs(fshift_ref) + 1)
            
            # Magnitude spektrumları arasındaki korelasyon
            # Basitçe matrisleri düzleştirip korelasyon katsayısına bakabiliriz
            score = np.corrcoef(magnitude_roi.flatten(), magnitude_ref.flatten())[0, 1]
            return max(0.0, score)
        except Exception:
            return 0.0

    def method_wavelet(self, img_roi, img_ref):
        """4. Wavelet Transform Yöntemi"""
        try:
            if img_roi.shape != img_ref.shape:
                img_roi = cv2.resize(img_roi, (img_ref.shape[1], img_ref.shape[0]))
                
            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
            gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
            
            # 2D Discrete Wavelet Transform
            coeffs_roi = pywt.dwt2(gray_roi, 'haar')
            cA_roi, (cH_roi, cV_roi, cD_roi) = coeffs_roi
            
            coeffs_ref = pywt.dwt2(gray_ref, 'haar')
            cA_ref, (cH_ref, cV_ref, cD_ref) = coeffs_ref
            
            # Approximation katsayıları (cA) karşılaştır
            score = np.corrcoef(cA_roi.flatten(), cA_ref.flatten())[0, 1]
            return max(0.0, score)
        except Exception:
            return 0.0

    def method_features(self, img_roi, threshold_area):
        """5. Özellik Çıkarımı (Kontur, Canny vb.)"""
        try:
            # Gürültü azaltma
            blurred = cv2.GaussianBlur(img_roi, (5, 5), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            
            # Canny Edge Detection (Otomatik eşikleme veya sabit)
            edges = cv2.Canny(gray, 50, 150)
            
            # Kontur bulma
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            total_area = 0
            valid_contours = 0
            details = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Küçük gürültüleri ele
                if area > 10: 
                    total_area += area
                    valid_contours += 1
            
            # Kriterler:
            # 1. Toplam Alan > threshold (user input)
            # 2. Kontur sayısı analizi (basitçe varlığı kontrol ediliyor şimdilik)
            
            is_ok = total_area >= threshold_area
            
            details_str = f"Kontur Sayısı: {valid_contours}\nToplam Alan: {total_area:.1f}"
            
            return is_ok, total_area, details_str
            
        except Exception as e:
            return False, 0.0, f"Hata: {str(e)}"

    def method_brute_force_matching(self, img_roi, img_ref):
        """6. Brute-Force Matcher Yöntemi (ORB ile)"""
        try:
            # Griye çevir
            if len(img_roi.shape) == 3:
                gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray_roi = img_roi
                
            if len(img_ref.shape) == 3:
                gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
            else:
                gray_ref = img_ref
            
            # ORB dedektörü oluştur
            orb = cv2.ORB_create()
            
            # Keypoint ve descriptor'ları bul
            kp1, des1 = orb.detectAndCompute(gray_roi, None)
            kp2, des2 = orb.detectAndCompute(gray_ref, None)
            
            if des1 is None or des2 is None:
                return 0.0
                
            # BFMatcher oluştur (Hamming mesafesi ile)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Eşleşmeleri bul
            matches = bf.match(des1, des2)
            
            # Mesafeye göre sırala
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Skor hesapla (Eşleşme sayısı / Maksimum olası eşleşme)
            # Veya en iyi N eşleşmenin ortalama mesafesi üzerinden bir skor türetilebilir
            # Burada basitçe eşleşme oranını kullanıyoruz
            
            max_keypoints = max(len(kp1), len(kp2))
            if max_keypoints == 0:
                return 0.0
                
            match_score = len(matches) / max_keypoints
            
            # Skoru 0-1 arasına normalize etmeye çalışalım (deneysel)
            # Genelde 0.2-0.3 üzeri iyi bir eşleşme sayılabilir karmaşık görüntülerde
            
            final_score = min(1.0, match_score * 2.0) # Skoru biraz boost edelim
            
            return final_score
            
        except Exception as e:
            print(f"BF Matcher error: {e}")
            return 0.0

    def __del__(self):
        """Temizlik"""
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()

# Uygulamayı başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = ProductionMonitoringSystem(root)
    root.mainloop()