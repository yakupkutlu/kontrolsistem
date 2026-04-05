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

try:
    import snap7
    from snap7.util import set_bool, get_bool
    SNAP7_AVAILABLE = True
except (ImportError, OSError):
    SNAP7_AVAILABLE = False

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


# ─────────────────────────────────────────────────────────────────────────────
# PLC Manager – Siemens S7 (snap7)
# ─────────────────────────────────────────────────────────────────────────────
class PLCManager:
    """Siemens S7 PLC haberleşme yöneticisi (snap7 tabanlı)."""

    DEFAULT_CONFIG = {
        "ip": "192.168.0.1",
        "rack": 0,
        "slot": 2,
        "ok_db": 600,
        "ok_byte": 0,
        "ok_bit": 0,
        "trigger_db": 600,
        "trigger_byte": 0,
        "trigger_bit": 1,
        "poll_interval": 0.10,
        "enabled": False,
    }
    CONFIG_FILE = "plc_config.json"

    def __init__(self):
        self.client = None
        self.connected = False
        self.config = dict(self.DEFAULT_CONFIG)
        self._lock = threading.Lock()   # snap7 thread-safe değil
        self._load_config()

    # ── Config ────────────────────────────────────────────────────────────────
    def _load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.config.update(saved)
            except Exception:
                pass

    def save_config(self):
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"PLC config kayıt hatası: {e}")

    # ── Bağlantı ──────────────────────────────────────────────────────────────
    def connect(self) -> tuple:
        """PLC'ye bağlan. (success: bool, message: str)"""
        if not SNAP7_AVAILABLE:
            return False, "snap7 kütüphanesi yüklü değil.\npip install python-snap7"
        try:
            if self.client and self.connected:
                self.disconnect()
            self.client = snap7.Client()
        except OSError as e:
            self.connected = False
            return False, f"snap7 DLL yüklenemedi: {e}\nsnap7.dll eksik olabilir."
        try:
            self.client.connect(
                self.config["ip"],
                self.config["rack"],
                self.config["slot"],
            )
            self.connected = True
            return True, f"Bağlandı: {self.config['ip']}"
        except Exception as e:
            self.connected = False
            return False, f"Bağlantı hatası: {e}"

    def disconnect(self):
        try:
            if self.client:
                self.client.disconnect()
                self.client.destroy()
        except Exception:
            pass
        finally:
            self.client = None
            self.connected = False

    def ensure_connected(self) -> bool:
        """Bağlı değilse yeniden bağlan."""
        if self.connected and self.client:
            try:
                self.client.get_connected()
                return True
            except Exception:
                self.connected = False
        ok, _ = self.connect()
        return ok

    # ── Bit Yaz ───────────────────────────────────────────────────────────────
    def write_bit(self, db: int, byte_offset: int, bit_offset: int, value: bool) -> tuple:
        """DB bloğuna tek bit yaz. (success: bool, message: str)"""
        if not SNAP7_AVAILABLE:
            return False, "snap7 yüklü değil"
        with self._lock:
            if not self.ensure_connected():
                return False, "PLC bağlantısı kurulamadı"
            try:
                data = self.client.db_read(db, byte_offset, 1)
                set_bool(data, 0, bit_offset, value)
                self.client.db_write(db, byte_offset, data)
                return True, f"DB{db}.DBX{byte_offset}.{bit_offset} = {int(value)}"
            except Exception as e:
                self.connected = False
                return False, f"Yazma hatası: {e}"

    def write_ok_signal(self, value: bool) -> tuple:
        """OK sinyalini yaz."""
        return self.write_bit(
            self.config["ok_db"],
            self.config["ok_byte"],
            self.config["ok_bit"],
            value,
        )

    # ── Bit Oku ───────────────────────────────────────────────────────────────
    def read_bit(self, db: int, byte_offset: int, bit_offset: int) -> tuple:
        """DB bloğundan tek bit oku. (success: bool, value: bool, message: str)"""
        if not SNAP7_AVAILABLE:
            return False, False, "snap7 yüklü değil"
        with self._lock:
            if not self.ensure_connected():
                return False, False, "PLC bağlantısı kurulamadı"
            try:
                data = self.client.db_read(db, byte_offset, 1)
                value = get_bool(data, 0, bit_offset)
                return True, bool(value), f"DB{db}.DBX{byte_offset}.{bit_offset} = {int(value)}"
            except Exception as e:
                self.connected = False
                return False, False, f"Okuma hatası: {e}"

    def read_trigger(self) -> tuple:
        """Trigger bitini oku. (success: bool, value: bool, message: str)"""
        return self.read_bit(
            self.config["trigger_db"],
            self.config["trigger_byte"],
            self.config["trigger_bit"],
        )




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
        self._zoom_level = 1.0  # Proje oluşturma canvas zoom seviyesi
        self._zoom_ox = None   # Fare merkezli zoom x offset (None = ortala)
        self._zoom_oy = None   # Fare merkezli zoom y offset (None = ortala)
        # PLC yöneticisi
        self.plc = PLCManager()
        # İzleme döngü aralığı (ms) – ayarlar ekranından değiştirilebilir
        self.monitor_interval = 1000
        # PLC etkinse uygulama açılışında otomatik bağlan
        if self.plc.config.get("enabled", False):
            def _auto_plc():
                import time; time.sleep(0.8)
                self.plc.connect()
            threading.Thread(target=_auto_plc, daemon=True).start()
        # Algoritma seçenekleri – tüm ekranlardan erişilebilir
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
        _retry_count = 0

        if isinstance(source, str):
            cap = cv2.VideoCapture(source)          # IP/URL → FFmpeg
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # USB

        if not cap.isOpened():
            print(f"[UYARI] Kamera açılamadı: {source}")

        while self._cam_reader_active:
            if not cap.isOpened():
                # Bağlantı kesildiyse yeniden dene (en fazla 60 saniye bekle)
                _retry_count += 1
                wait = min(5.0, 1.0 + _retry_count * 0.5)
                time.sleep(wait)
                print(f"[BİLGİ] Kamera yeniden deneniyor ({_retry_count}): {source}")
                if isinstance(source, str):
                    cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                continue

            _retry_count = 0
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

        manage_btn = ttk.Button(
            button_frame,
            text="📋  Projeleri Yönet",
            style="Ghost.TButton",
            command=self.project_management_screen
        )
        manage_btn.pack(pady=(8, 0), fill='x')

        plc_btn = ttk.Button(
            button_frame,
            text="🔌  PLC Ayarları",
            style="Ghost.TButton",
            command=self.plc_settings_screen
        )
        plc_btn.pack(pady=(8, 0), fill='x')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Proje Yönetimi Ekranı
    # ─────────────────────────────────────────────────────────────────────────
    def project_management_screen(self):
        """Projeleri listele, sil veya yeniden adlandır."""
        win = tk.Toplevel(self.root)
        win.title("Proje Yönetimi")
        win.geometry("560x480")
        win.configure(bg="#0f172a")
        win.grab_set()

        tk.Label(win, text="Proje Yönetimi",
                 bg="#0f172a", fg="#e5e7eb",
                 font=("Segoe UI", 14, "bold")).pack(pady=(18, 8))

        # Liste kutusu
        frame = tk.Frame(win, bg="#0f172a")
        frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        lb = tk.Listbox(frame,
                        bg="#1e293b", fg="#e5e7eb",
                        font=("Segoe UI", 11),
                        selectbackground="#2563eb",
                        activestyle="none",
                        yscrollcommand=scrollbar.set)
        lb.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=lb.yview)

        # Proje verisi
        projects = []

        def _load_projects():
            lb.delete(0, tk.END)
            projects.clear()
            self.cursor.execute(
                "SELECT id, name, created_date FROM projects ORDER BY created_date DESC"
            )
            for row in self.cursor.fetchall():
                projects.append({'id': row[0], 'name': row[1], 'created': row[2]})
                date_str = (row[2] or '')[:10]
                lb.insert(tk.END, f"  {row[1]}   [{date_str}]")

        _load_projects()

        # Çift tıklama ile seç (hızlı erişim için)
        lb.bind("<Double-Button-1>", lambda _: _rename())

        # ── Buton satırı ─────────────────────────────────────────────────────
        btn_row = tk.Frame(win, bg="#0f172a")
        btn_row.pack(fill="x", padx=20, pady=(0, 16))

        def _get_selected():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Uyarı", "Lütfen bir proje seçin!", parent=win)
                return None
            return projects[sel[0]]

        def _rename():
            proj = _get_selected()
            if proj is None:
                return
            new_name = simpledialog.askstring(
                "Yeniden Adlandır",
                f"'{proj['name']}' için yeni isim:",
                initialvalue=proj['name'],
                parent=win
            )
            if not new_name or new_name.strip() == proj['name']:
                return
            new_name = new_name.strip()
            try:
                self.cursor.execute(
                    "UPDATE projects SET name=?, updated_date=? WHERE id=?",
                    (new_name, datetime.datetime.now().isoformat(), proj['id'])
                )
                self.conn.commit()
                _load_projects()
            except sqlite3.IntegrityError:
                messagebox.showerror("Hata", f"'{new_name}' adında bir proje zaten var!", parent=win)
            except Exception as e:
                messagebox.showerror("Hata", str(e), parent=win)

        def _delete():
            proj = _get_selected()
            if proj is None:
                return
            if not messagebox.askyesno(
                "Sil",
                f"'{proj['name']}' projesi kalıcı olarak silinsin mi?\n"
                "(ROI görüntüleri ve ML modelleri silinmez.)",
                parent=win
            ):
                return
            try:
                self.cursor.execute("DELETE FROM monitoring_logs WHERE project_id=?", (proj['id'],))
                self.cursor.execute("DELETE FROM roi_ml_models WHERE project_id=?", (proj['id'],))
                self.cursor.execute("DELETE FROM projects WHERE id=?", (proj['id'],))
                self.conn.commit()
                _load_projects()
            except Exception as e:
                messagebox.showerror("Hata", str(e), parent=win)

        tk.Button(btn_row, text="✏  Yeniden Adlandır",
                  bg="#2563eb", fg="white",
                  font=("Segoe UI", 10, "bold"),
                  bd=0, padx=10, pady=8,
                  command=_rename).pack(side="left", fill="x", expand=True, padx=(0, 4))

        tk.Button(btn_row, text="🗑  Sil",
                  bg="#dc2626", fg="white",
                  font=("Segoe UI", 10, "bold"),
                  bd=0, padx=10, pady=8,
                  command=_delete).pack(side="left", fill="x", expand=True, padx=(4, 0))

        tk.Button(win, text="Kapat",
                  bg="#374151", fg="white",
                  font=("Segoe UI", 10),
                  bd=0, padx=10, pady=8,
                  command=win.destroy).pack(pady=(0, 12), padx=20, fill="x")

    # ─────────────────────────────────────────────────────────────────────────
    # PLC Ayarları Ekranı
    # ─────────────────────────────────────────────────────────────────────────
    def plc_settings_screen(self):
        """PLC bağlantı ve sinyal ayarları penceresi."""
        win = tk.Toplevel(self.root)
        win.title("PLC Ayarları – Siemens S7")
        win.geometry("480x820")
        win.resizable(False, False)
        win.configure(bg="#0f172a")
        win.grab_set()

        cfg = self.plc.config  # canlı referans

        # ── Yardımcılar ──────────────────────────────────────────────────────
        def section(parent, title):
            frm = tk.LabelFrame(parent, text=f"  {title}  ",
                                bg="#020617", fg="#7c3aed",
                                font=("Segoe UI", 10, "bold"),
                                bd=1, relief="groove",
                                padx=10, pady=8)
            frm.pack(fill="x", padx=16, pady=(8, 0))
            return frm

        def row(parent, label, widget_factory, **kw):
            r = tk.Frame(parent, bg="#020617")
            r.pack(fill="x", pady=3)
            tk.Label(r, text=label, width=16, anchor="w",
                     bg="#020617", fg="#e5e7eb",
                     font=("Segoe UI", 10)).pack(side="left")
            w = widget_factory(r, **kw)
            w.pack(side="left", fill="x", expand=True)
            return w

        entry_cfg = dict(bg="#1e293b", fg="#e5e7eb",
                         insertbackground="white",
                         relief="flat", bd=4,
                         font=("Segoe UI", 10))

        spin_cfg = dict(bg="#1e293b", fg="#e5e7eb",
                        buttonbackground="#334155",
                        relief="flat", bd=2,
                        font=("Segoe UI", 10), width=6)

        # ── PLC Etkin ────────────────────────────────────────────────────────
        enabled_var = tk.BooleanVar(value=cfg.get("enabled", False))
        top_frm = tk.Frame(win, bg="#0f172a")
        top_frm.pack(fill="x", padx=16, pady=(12, 4))
        tk.Checkbutton(top_frm, text="PLC Entegrasyonunu Etkinleştir",
                       variable=enabled_var,
                       bg="#0f172a", fg="#e5e7eb",
                       selectcolor="#1e293b",
                       activebackground="#0f172a",
                       activeforeground="#e5e7eb",
                       font=("Segoe UI", 10, "bold")).pack(side="left")

        # ── Bağlantı ─────────────────────────────────────────────────────────
        s_conn = section(win, "Bağlantı")

        ip_var = tk.StringVar(value=cfg.get("ip", "192.168.0.1"))
        row(s_conn, "PLC IP:", lambda p: tk.Entry(p, textvariable=ip_var, **entry_cfg))

        rack_var = tk.IntVar(value=cfg.get("rack", 0))
        slot_var = tk.IntVar(value=cfg.get("slot", 1))

        rs_row = tk.Frame(s_conn, bg="#020617")
        rs_row.pack(fill="x", pady=3)
        tk.Label(rs_row, text="Rack:", width=8, anchor="w",
                 bg="#020617", fg="#e5e7eb",
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Spinbox(rs_row, from_=0, to=10, textvariable=rack_var,
                   **spin_cfg).pack(side="left", padx=(0, 16))
        tk.Label(rs_row, text="Slot:", anchor="w",
                 bg="#020617", fg="#e5e7eb",
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Spinbox(rs_row, from_=0, to=10, textvariable=slot_var,
                   **spin_cfg).pack(side="left")

        # ── Yazma OK ─────────────────────────────────────────────────────────
        s_ok = section(win, "Yazma OK  (tüm ROI = OK → 1 gönderilir)")

        ok_db_var   = tk.IntVar(value=cfg.get("ok_db", 600))
        ok_byte_var = tk.IntVar(value=cfg.get("ok_byte", 0))
        ok_bit_var  = tk.IntVar(value=cfg.get("ok_bit", 2))

        ok_row = tk.Frame(s_ok, bg="#020617")
        ok_row.pack(fill="x", pady=3)
        for label, var, lo, hi in [
            ("DB No:", ok_db_var, 1, 9999),
            ("Byte:",  ok_byte_var, 0, 9999),
            ("Bit:",   ok_bit_var,  0, 7),
        ]:
            tk.Label(ok_row, text=label, anchor="w",
                     bg="#020617", fg="#e5e7eb",
                     font=("Segoe UI", 10), width=7).pack(side="left")
            tk.Spinbox(ok_row, from_=lo, to=hi, textvariable=var,
                       **spin_cfg).pack(side="left", padx=(0, 8))

        ok_preview = tk.Label(s_ok, text="", bg="#020617", fg="#16a34a",
                              font=("Segoe UI", 9, "italic"))
        ok_preview.pack(anchor="w")

        # ── Trigger Okuma (Fotoğraf Çek Sinyali) ─────────────────────────────
        s_trig = section(win, "Trigger Okuma  (PLC → fotoğraf çek sinyali)")

        trig_db_var   = tk.IntVar(value=cfg.get("trigger_db", 600))
        trig_byte_var = tk.IntVar(value=cfg.get("trigger_byte", 0))
        trig_bit_var  = tk.IntVar(value=cfg.get("trigger_bit", 1))

        trig_row = tk.Frame(s_trig, bg="#020617")
        trig_row.pack(fill="x", pady=3)
        for label, var, lo, hi in [
            ("DB No:", trig_db_var, 1, 9999),
            ("Byte:",  trig_byte_var, 0, 9999),
            ("Bit:",   trig_bit_var,  0, 7),
        ]:
            tk.Label(trig_row, text=label, anchor="w",
                     bg="#020617", fg="#e5e7eb",
                     font=("Segoe UI", 10), width=7).pack(side="left")
            tk.Spinbox(trig_row, from_=lo, to=hi, textvariable=var,
                       **spin_cfg).pack(side="left", padx=(0, 8))

        trig_preview = tk.Label(s_trig, text="", bg="#020617", fg="#f59e0b",
                                font=("Segoe UI", 9, "italic"))
        trig_preview.pack(anchor="w")

        trig_test_lbl = tk.Label(s_trig, text="", bg="#020617",
                                 font=("Segoe UI", 9), wraplength=420)
        trig_test_lbl.pack(anchor="w", pady=(2, 0))

        def _test_read_trigger():
            _save_to_cfg()
            trig_test_lbl.config(text="Okunuyor...", fg="#f59e0b")
            def _do_read():
                ok, val, msg = self.plc.read_trigger()
                color = "#16a34a" if ok else "#dc2626"
                win.after(0, lambda: trig_test_lbl.config(text=msg, fg=color))
            threading.Thread(target=_do_read, daemon=True).start()

        tk.Button(s_trig, text="🔍  Trigger Bitini Oku (Test)",
                  bg="#0ea5e9", fg="white", activebackground="#0284c7",
                  font=("Segoe UI", 10, "bold"),
                  bd=0, padx=10, pady=6,
                  command=_test_read_trigger).pack(fill="x", pady=(6, 0))

        # ── Okuma Hızı ───────────────────────────────────────────────────────
        s_speed = section(win, "Okuma Hızı")
        poll_var = tk.DoubleVar(value=cfg.get("poll_interval", 0.10))
        row(s_speed, "Poll interval (sn):",
            lambda p: tk.Spinbox(p, from_=0.05, to=10.0, increment=0.05,
                                 textvariable=poll_var, format="%.2f",
                                 **spin_cfg))

        # ── Adres önizleme güncelleyici ───────────────────────────────────────
        def _update_previews(*_):
            try:
                ok_preview.config(
                    text=f"DB{ok_db_var.get()}.DBX{ok_byte_var.get()}.{ok_bit_var.get()}")
                trig_preview.config(
                    text=f"DB{trig_db_var.get()}.DBX{trig_byte_var.get()}.{trig_bit_var.get()}")
            except Exception:
                pass

        for v in (ok_db_var, ok_byte_var, ok_bit_var,
                  trig_db_var, trig_byte_var, trig_bit_var):
            v.trace_add("write", _update_previews)
        _update_previews()

        # ── Test Butonu ───────────────────────────────────────────────────────
        status_lbl = tk.Label(win, text="", bg="#0f172a",
                              font=("Segoe UI", 10), wraplength=440)
        status_lbl.pack(pady=(10, 0))

        def _test_connection():
            try:
                _save_to_cfg()
                status_lbl.config(text="Bağlanıyor...", fg="#f59e0b")
                # Bağlantıyı ayrı thread'de çalıştır → UI donmaz
                def _do_connect():
                    ok, msg = self.plc.connect()
                    # Sonucu UI thread'inde güncelle
                    win.after(0, lambda: status_lbl.config(
                        text=msg, fg="#16a34a" if ok else "#dc2626"))
                threading.Thread(target=_do_connect, daemon=True).start()
            except Exception as e:
                status_lbl.config(text=f"Hata: {e}", fg="#dc2626")

        tk.Button(win, text="🔌  Bağlantıyı Test Et",
                  bg="#7c3aed", fg="white", activebackground="#6d28d9",
                  font=("Segoe UI", 11, "bold"),
                  bd=0, padx=12, pady=8,
                  command=_test_connection).pack(fill="x", padx=16, pady=(6, 0))

        # ── Manuel Bit Yaz ───────────────────────────────────────────────────
        s_man = section(win, "Manuel Bit Yaz")

        man_db_var   = tk.IntVar(value=cfg.get("ok_db", 600))
        man_byte_var = tk.IntVar(value=cfg.get("ok_byte", 0))
        man_bit_var  = tk.IntVar(value=cfg.get("ok_bit", 2))

        man_row = tk.Frame(s_man, bg="#020617")
        man_row.pack(fill="x", pady=4)
        for label, var, lo, hi in [
            ("DB:", man_db_var, 1, 9999),
            ("Byte:", man_byte_var, 0, 9999),
            ("Bit:", man_bit_var, 0, 7),
        ]:
            tk.Label(man_row, text=label, anchor="w",
                     bg="#020617", fg="#e5e7eb",
                     font=("Segoe UI", 10), width=6).pack(side="left")
            tk.Spinbox(man_row, from_=lo, to=hi, textvariable=var,
                       **spin_cfg).pack(side="left", padx=(0, 8))

        man_addr_lbl = tk.Label(s_man, text="", bg="#020617",
                                fg="#f59e0b", font=("Segoe UI", 9, "italic"))
        man_addr_lbl.pack(anchor="w")

        def _update_man_addr(*_):
            try:
                man_addr_lbl.config(
                    text=f"DB{man_db_var.get()}.DBX{man_byte_var.get()}.{man_bit_var.get()}")
            except Exception:
                pass

        for v in (man_db_var, man_byte_var, man_bit_var):
            v.trace_add("write", _update_man_addr)
        _update_man_addr()

        man_status = tk.Label(s_man, text="", bg="#020617",
                              font=("Segoe UI", 9), wraplength=420)
        man_status.pack(anchor="w", pady=(2, 0))

        def _man_write(val: bool):
            try:
                _save_to_cfg()
                man_status.config(text="Yazılıyor...", fg="#f59e0b")
                db   = _safe_int(man_db_var, 600)
                byte = _safe_int(man_byte_var, 0)
                bit  = _safe_int(man_bit_var, 0)
                def _do_write():
                    ok, msg = self.plc.write_bit(db, byte, bit, val)
                    win.after(0, lambda: man_status.config(
                        text=msg, fg="#16a34a" if ok else "#dc2626"))
                threading.Thread(target=_do_write, daemon=True).start()
            except Exception as e:
                man_status.config(text=f"Hata: {e}", fg="#dc2626")

        btn_row = tk.Frame(s_man, bg="#020617")
        btn_row.pack(fill="x", pady=(6, 2))
        tk.Button(btn_row, text="▶  1 YAZ",
                  bg="#16a34a", fg="white", activebackground="#15803d",
                  font=("Segoe UI", 11, "bold"),
                  bd=0, padx=12, pady=6,
                  command=lambda: _man_write(True)).pack(side="left",
                                                         fill="x", expand=True,
                                                         padx=(0, 4))
        tk.Button(btn_row, text="■  0 YAZ",
                  bg="#dc2626", fg="white", activebackground="#b91c1c",
                  font=("Segoe UI", 11, "bold"),
                  bd=0, padx=12, pady=6,
                  command=lambda: _man_write(False)).pack(side="left",
                                                          fill="x", expand=True,
                                                          padx=(4, 0))

        # ── Kaydet / Kapat ────────────────────────────────────────────────────
        def _safe_int(var, default=0):
            try:
                return int(var.get())
            except Exception:
                return default

        def _safe_float(var, default=0.1):
            try:
                val = var.get()
                # Türkçe locale virgül sorununu düzelt
                return float(str(val).replace(",", "."))
            except Exception:
                return default

        def _save_to_cfg():
            try:
                cfg["enabled"]       = enabled_var.get()
                cfg["ip"]            = ip_var.get().strip()
                cfg["rack"]          = _safe_int(rack_var, 0)
                cfg["slot"]          = _safe_int(slot_var, 1)
                cfg["ok_db"]         = _safe_int(ok_db_var, 600)
                cfg["ok_byte"]       = _safe_int(ok_byte_var, 0)
                cfg["ok_bit"]        = _safe_int(ok_bit_var, 0)
                cfg["trigger_db"]    = _safe_int(trig_db_var, 600)
                cfg["trigger_byte"]  = _safe_int(trig_byte_var, 0)
                cfg["trigger_bit"]   = _safe_int(trig_bit_var, 1)
                cfg["poll_interval"] = _safe_float(poll_var, 0.1)
                self.plc.save_config()
            except Exception as e:
                print(f"PLC config kayıt hatası: {e}")

        def _save_close():
            _save_to_cfg()
            if cfg.get("enabled", False):
                status_lbl.config(text="Bağlanıyor...", fg="#f59e0b")
                def _do_connect():
                    ok, msg = self.plc.connect()
                    try:
                        win.after(0, lambda: status_lbl.config(
                            text=f"{'✅' if ok else '❌'} {msg}",
                            fg="#16a34a" if ok else "#dc2626"))
                        win.after(1800, win.destroy)
                    except Exception:
                        pass
                threading.Thread(target=_do_connect, daemon=True).start()
            else:
                self.plc.disconnect()
                win.destroy()

        btn_bar = tk.Frame(win, bg="#0f172a")
        btn_bar.pack(fill="x", padx=16, pady=(10, 16))
        tk.Button(btn_bar, text="💾 Kaydet ve Bağlan",
                  bg="#2563eb", fg="white", activebackground="#1d4ed8",
                  font=("Segoe UI", 11, "bold"),
                  bd=0, padx=12, pady=8,
                  command=_save_close).pack(fill="x")

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
        scale = min(cw / w, ch / h) * 0.95 * getattr(self, '_zoom_level', 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(rgb_frame, (nw, nh))
        self.display_scale = scale
        _zox = getattr(self, '_zoom_ox', None)
        _zoy = getattr(self, '_zoom_oy', None)
        ox = _zox if _zox is not None else (cw - nw) // 2
        oy = _zoy if _zoy is not None else (ch - nh) // 2
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

    def _on_canvas_zoom(self, event):
        """Canvas fare tekerleği ile zoom – fare pozisyonuna odaklanır"""
        if self.reference_image is None:
            return
        img_h, img_w = self.reference_image.shape[:2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        # Mevcut scale ve offset
        base = min(cw / img_w, ch / img_h) * 0.95
        old_scale = base * self._zoom_level
        old_ox = self._zoom_ox if self._zoom_ox is not None else (cw - int(img_w * old_scale)) // 2
        old_oy = self._zoom_oy if self._zoom_oy is not None else (ch - int(img_h * old_scale)) // 2

        # Fare altındaki görüntü koordinatı
        img_x = (event.x - old_ox) / old_scale
        img_y = (event.y - old_oy) / old_scale

        # Zoom güncelle
        if event.delta > 0:
            self._zoom_level = min(5.0, self._zoom_level * 1.15)
        else:
            self._zoom_level = max(0.2, self._zoom_level / 1.15)

        # Yeni scale ile fare noktasını aynı konuma getiren offset
        new_scale = base * self._zoom_level
        self._zoom_ox = int(event.x - img_x * new_scale)
        self._zoom_oy = int(event.y - img_y * new_scale)

        if self._preview_active:
            self._update_preview_canvas(self.reference_image)
        else:
            self.display_image(self.reference_image)
            self._redraw_roi_overlays_on_canvas()

    def _redraw_roi_overlays_on_canvas(self):
        """Zoom sonrası roi_list'teki ROI dikdörtgenlerini canvas'a yeniden çiz"""
        if not hasattr(self, 'canvas_offset') or not hasattr(self, 'display_scale'):
            return
        scale = self.display_scale
        ox, oy = self.canvas_offset
        for roi_item in self.roi_list:
            tag = f"roi_{roi_item['name']}"
            self.canvas.delete(tag)
            coords = roi_item['coords']
            cx1 = int(coords[0] * scale) + ox
            cy1 = int(coords[1] * scale) + oy
            cx2 = int((coords[0] + coords[2]) * scale) + ox
            cy2 = int((coords[1] + coords[3]) * scale) + oy
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                         outline=roi_item['color'], width=2, tags=tag)
            self.canvas.create_text(cx1 + 4, cy1 + 4, anchor='nw',
                                    text=roi_item['name'], fill=roi_item['color'],
                                    font=('Arial', 9, 'bold'), tags=tag)

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

        # Proje Adı
        ttk.Label(left_panel, text="Proje Adı", style="Section.TLabel").pack(pady=(14,4))
        self._new_project_name_var = tk.StringVar()
        tk.Entry(left_panel, textvariable=self._new_project_name_var,
                 font=('Segoe UI', 11), bg='#1e293b', fg='#e5e7eb',
                 insertbackground='white', relief='flat', bd=4
                 ).pack(fill='x', padx=14, pady=(0,8))

        # Kamera Seçimi
        ttk.Label(left_panel, text="Kamera Seçimi", style="Section.TLabel").pack(pady=(4,4))

        cam_row = ttk.Frame(left_panel, style="Card.TFrame")
        cam_row.pack(fill='x', padx=10)

        self._cam_options = []   # (source_val, label) tuples
        self.camera_combo = ttk.Combobox(cam_row, state='readonly', width=22,
                                         font=('Arial', 10))
        self.camera_combo.pack(side='left', padx=(0,5))

        def _scan_and_fill():
            cams = self.scan_cameras()
            self._cam_options = cams
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
        self.canvas.bind('<MouseWheel>', self._on_canvas_zoom)
        self.temp_rect = None
        self._zoom_level = 1.0
        self._zoom_ox = None
        self._zoom_oy = None

        # İlk açılışta kameraları otomatik tara
        self.root.after(200, _scan_and_fill)

    def _refresh_roi_listbox(self):
        """Sol paneldeki ROI listesini güncelle – OK/NOK kaydet + Eğit butonlarıyla."""
        for w in self.roi_listbox_frame.winfo_children():
            w.destroy()

        for idx, roi_item in enumerate(self.roi_list):
            name  = roi_item['name']
            color = roi_item['color']
            x, y, ww, hh = roi_item['coords']

            # ─── Kart çerçevesi ───────────────────────────────────────────────
            card = tk.Frame(self.roi_listbox_frame, bg='#1e293b',
                            bd=0, relief='flat')
            card.pack(fill='x', pady=3, padx=2)

            # Başlık satırı: renk noktası + isim + boyut + sil
            hdr = tk.Frame(card, bg='#1e293b')
            hdr.pack(fill='x', padx=6, pady=(5, 2))
            tk.Label(hdr, text="●", fg=color, bg='#1e293b',
                     font=('Arial', 12)).pack(side='left')
            tk.Label(hdr, text=f"{name}  ({ww}×{hh})",
                     bg='#1e293b', fg='#e5e7eb',
                     font=('Segoe UI', 9, 'bold'), anchor='w').pack(side='left', expand=True, fill='x')
            tk.Button(hdr, text="✕", fg='white', bg='#e74c3c',
                      font=('Arial', 8, 'bold'), width=2, cursor='hand2',
                      bd=0,
                      command=lambda i=idx: self._delete_roi(i)).pack(side='right')

            # Görüntü sayacı
            def _count(roi_name):
                proj = getattr(self, '_new_project_name_var', None)
                pname = proj.get().strip() if proj else ''
                if not pname:
                    return 0, 0
                base = os.path.join('roi_images', self._safe_name(pname), self._safe_name(roi_name))
                ok_n  = len([f for f in os.listdir(os.path.join(base,'OK'))
                              if f.lower().endswith(('.jpg','.png'))]
                             ) if os.path.isdir(os.path.join(base,'OK')) else 0
                nok_n = len([f for f in os.listdir(os.path.join(base,'NOK'))
                              if f.lower().endswith(('.jpg','.png'))]
                             ) if os.path.isdir(os.path.join(base,'NOK')) else 0
                return ok_n, nok_n

            ok_n, nok_n = _count(name)
            cnt_color = '#22c55e' if ok_n >= 5 and nok_n >= 5 else '#f59e0b'
            cnt_lbl = tk.Label(card, text=f"  OK: {ok_n}  |  NOK: {nok_n}",
                               bg='#1e293b', fg=cnt_color,
                               font=('Segoe UI', 8))
            cnt_lbl.pack(anchor='w', padx=6)

            def _refresh_count(lbl, roi_name):
                ok_n, nok_n = _count(roi_name)
                c = '#22c55e' if ok_n >= 5 and nok_n >= 5 else '#f59e0b'
                lbl.config(text=f"  OK: {ok_n}  |  NOK: {nok_n}", fg=c)

            # OK / NOK kaydet butonları
            def _capture_for_roi(roi_item_ref, label: str, count_lbl):
                proj = getattr(self, '_new_project_name_var', None)
                pname = proj.get().strip() if proj else ''
                if not pname:
                    messagebox.showwarning("Uyarı", "Önce proje adını girin!")
                    return
                # Anlık frame al
                frame = None
                cam_id = getattr(self, 'camera_var', None)
                if cam_id is not None:
                    source = self.get_camera_source(cam_id.get())
                    if source == "HIKROBOT" and self.hik_camera:
                        frame = self.hik_camera.get_frame()
                    elif self._preview_active:
                        # Canlı önizlemeden son frame'i al
                        with self._cam_reader_lock:
                            frame = self._cam_reader_frame.copy() if self._cam_reader_frame is not None else None
                        if frame is None:
                            # Direkt yakala
                            if isinstance(source, int):
                                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                            else:
                                cap = cv2.VideoCapture(source)
                            ret, frame = cap.read()
                            cap.release()
                            if not ret:
                                frame = None

                if frame is None and self.reference_image is not None:
                    # Referans görüntüyü kullan (kamera yoksa)
                    frame = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR)

                if frame is None:
                    messagebox.showwarning("Uyarı", "Kamera görüntüsü alınamadı!")
                    return

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h_img, w_img = frame_rgb.shape[:2]
                rx, ry, rw, rh = roi_item_ref['coords']
                rx = max(0, min(int(rx), w_img - 1))
                ry = max(0, min(int(ry), h_img - 1))
                rw = max(1, min(int(rw), w_img - rx))
                rh = max(1, min(int(rh), h_img - ry))
                crop = frame_rgb[ry:ry+rh, rx:rx+rw]
                if crop.size == 0:
                    messagebox.showwarning("Uyarı", "ROI bölgesi geçersiz!")
                    return

                sub = os.path.join('roi_images', self._safe_name(pname),
                                   self._safe_name(roi_item_ref['name']), label)
                os.makedirs(sub, exist_ok=True)
                unique = int(datetime.datetime.now().timestamp() * 1000)
                cv2.imwrite(os.path.join(sub, f"{self._safe_name(roi_item_ref['name'])}_{unique}.jpg"),
                            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                _refresh_count(count_lbl, roi_item_ref['name'])

            # Per-ROI algoritma seçimi
            algo_row = tk.Frame(card, bg='#1e293b')
            algo_row.pack(fill='x', padx=6, pady=(2, 0))
            tk.Label(algo_row, text="Algoritma:", bg='#1e293b', fg='#9ca3af',
                     font=('Segoe UI', 7)).pack(side='left')
            _roi_algo_var = tk.StringVar()
            # Önceden seçilmiş algoritma varsa göster
            _saved_algo = roi_item.get('algorithm')
            if _saved_algo:
                for _dn, _in in self.algo_options_map.items():
                    if _in == _saved_algo:
                        _roi_algo_var.set(_dn)
                        break
            if not _roi_algo_var.get():
                _roi_algo_var.set(list(self.algo_options_map.keys())[9])  # "ML model (HOG-SVM)"
            _roi_algo_combo = ttk.Combobox(algo_row,
                                           textvariable=_roi_algo_var,
                                           values=list(self.algo_options_map.keys()),
                                           state='readonly',
                                           font=('Arial', 7),
                                           width=22)
            _roi_algo_combo.pack(side='left', padx=(4, 0), fill='x', expand=True)

            def _on_roi_algo_change(event, ri=roi_item, var=_roi_algo_var):
                ri['algorithm'] = self.algo_options_map.get(var.get())

            _roi_algo_combo.bind('<<ComboboxSelected>>', _on_roi_algo_change)
            # İlk değeri hemen uygula
            roi_item['algorithm'] = self.algo_options_map.get(_roi_algo_var.get())

            btn_row = tk.Frame(card, bg='#1e293b')
            btn_row.pack(fill='x', padx=6, pady=(2, 2))
            tk.Button(btn_row, text="✅ OK",
                      bg='#16a34a', fg='white', font=('Segoe UI', 8, 'bold'),
                      bd=0, padx=4, pady=3,
                      command=lambda ri=roi_item, cl=cnt_lbl: _capture_for_roi(ri, 'OK', cl)
                      ).pack(side='left', fill='x', expand=True, padx=(0,2))
            tk.Button(btn_row, text="❌ NOK",
                      bg='#dc2626', fg='white', font=('Segoe UI', 8, 'bold'),
                      bd=0, padx=4, pady=3,
                      command=lambda ri=roi_item, cl=cnt_lbl: _capture_for_roi(ri, 'NOK', cl)
                      ).pack(side='left', fill='x', expand=True, padx=(2,2))

            def _train_roi(roi_name, roi_ref):
                proj = getattr(self, '_new_project_name_var', None)
                pname = proj.get().strip() if proj else ''
                if not pname:
                    messagebox.showwarning("Uyarı", "Önce proje adını girin!")
                    return
                algo = roi_ref.get('algorithm') or 'ML_MODEL'
                # Geçici current_project oluştur (DB'ye kaydedilmeden önce)
                _prev = self.current_project
                self.current_project = {
                    'id': None,
                    'name': pname,
                    'algorithm': algo,
                    'algo_threshold': 0.75,
                    'roi_list': [{'name': r['name'], 'coords': r['coords'],
                                  'algorithm': r.get('algorithm'), 'threshold': r.get('threshold')}
                                 for r in self.roi_list],
                    'roi': self.roi_list[0]['coords'] if self.roi_list else (0,0,1,1),
                    'camera_id': getattr(self, 'camera_var', tk.IntVar()).get(),
                }
                self._train_single_roi(roi_name, algo)
                self.current_project = _prev

            tk.Button(btn_row, text="🧠 Eğit",
                      bg='#7c3aed', fg='white', font=('Segoe UI', 8, 'bold'),
                      bd=0, padx=4, pady=3,
                      command=lambda rn=name, ri=roi_item: _train_roi(rn, ri)
                      ).pack(side='left', fill='x', expand=True, padx=(2,0))

            # Ince ayırıcı
            tk.Frame(self.roi_listbox_frame, bg='#0f172a', height=1).pack(fill='x')

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
            scale = min(canvas_w/w, canvas_h/h) * 0.95 * getattr(self, '_zoom_level', 1.0)
            new_w, new_h = int(w*scale), int(h*scale)

            img_resized = cv2.resize(image, (new_w, new_h))
            self.display_scale = scale
            
            # PIL Image'e çevir
            img_pil = Image.fromarray(img_resized)
            self.photo = ImageTk.PhotoImage(img_pil)
            
            # Canvas'ı temizle ve göster
            self.canvas.delete('all')
            _zox = getattr(self, '_zoom_ox', None)
            _zoy = getattr(self, '_zoom_oy', None)
            x = _zox if _zox is not None else (canvas_w - new_w) // 2
            y = _zoy if _zoy is not None else (canvas_h - new_h) // 2
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

        project_name = getattr(self, '_new_project_name_var', None)
        project_name = project_name.get().strip() if project_name else ''
        if not project_name:
            project_name = simpledialog.askstring("Proje Adı", "Proje adını girin:")
        if not project_name:
            return

        try:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR))
            img_binary = buffer.tobytes()
            now = datetime.datetime.now().isoformat()

            roi_json = json.dumps([
                {'name': r['name'], 'x': r['coords'][0], 'y': r['coords'][1],
                 'w': r['coords'][2], 'h': r['coords'][3],
                 'algorithm': r.get('algorithm') or None,
                 'threshold': r.get('threshold') or None}
                for r in self.roi_list
            ], ensure_ascii=False)

            # Proje varsayılan algoritması: ilk ROI'nin algoritması, yoksa SSIM
            first_roi_algo = (self.roi_list[0].get('algorithm') if self.roi_list else None) or 'SSIM'
            algo_key = first_roi_algo
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
            new_project_id = self.cursor.lastrowid

            # Proje oluşturma sırasında eğitilen modeller project_id=None ile
            # diske kaydedilmişti — şimdi DB'ye bağla
            self._register_pending_models(new_project_id, project_name)

            self.stop_live_preview()
            messagebox.showinfo("Başarılı", f"Proje '{project_name}' kaydedildi!")
            self.show_main_menu()

        except sqlite3.IntegrityError:
            messagebox.showerror("Hata", "Bu isimde bir proje zaten var!")
        except Exception as e:
            messagebox.showerror("Hata", f"Kayıt hatası: {str(e)}")


    def _register_pending_models(self, project_id: int, project_name: str):
        """Proje oluşturma sırasında diske kaydedilen (project_id=None) modelleri
        yeni proje ID'siyle DB'ye kaydet."""
        safe_proj = self._safe_name(project_name)
        # Her ROI için olası model dosyalarını tara
        suffix_map = {
            'SIFT_SVM':    'SIFT_SVM',
            'DAISY_SVM':   'DAISY_SVM',
            'HAAR_SVM':    'HAAR_SVM',
            'CENSURE_SVM': 'CENSURE_SVM',
            'MBLBP_SVM':   'MBLBP_SVM',
            'GLCM_SVM':    'GLCM_SVM',
            'LBP_SVM':     'LBP_SVM',
            'GABOR_SVM':   'GABOR_SVM',
            'FISHER_SVM':  'FISHER_SVM',
            '':            'HOG_SVM',   # suffix yok → HOG
        }
        now = datetime.datetime.now().isoformat()
        for roi_item in self.roi_list:
            roi_name = roi_item['name']
            safe_roi = self._safe_name(roi_name)
            roi_folder = os.path.join('roi_images', safe_proj, roi_name)
            if not os.path.isdir(roi_folder):
                continue
            for suffix, model_type_str in suffix_map.items():
                fname = f"{safe_proj}_{safe_roi}_{suffix}.pkl" if suffix \
                    else f"{safe_proj}_{safe_roi}.pkl"
                model_path = os.path.join(roi_folder, fname)
                if not os.path.exists(model_path):
                    continue
                try:
                    gmm_path = None
                    if model_type_str == 'FISHER_SVM':
                        gp = os.path.join(roi_folder, f"{safe_proj}_{safe_roi}_FISHER_GMM.pkl")
                        gmm_path = gp if os.path.exists(gp) else None
                    params = {'model_type': model_type_str,
                              'gmm_path': gmm_path} if gmm_path else {'model_type': model_type_str}
                    self.cursor.execute('''
                        INSERT INTO roi_ml_models
                            (project_id, roi_name, model_type, model_path, params_json, created_date, updated_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(project_id, roi_name) DO UPDATE SET
                            model_type=excluded.model_type,
                            model_path=excluded.model_path,
                            params_json=excluded.params_json,
                            updated_date=excluded.updated_date
                    ''', (project_id, roi_name, model_type_str, model_path,
                          json.dumps(params), now, now))
                    self.conn.commit()
                    break  # Bu ROI için bir model bulundu, diğer suffix'lere gerek yok
                except Exception as e:
                    print(f"[UYARI] Model kaydı atlandı ({roi_name}/{suffix}): {e}")

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
        try:
            self.load_project(project_id)
            self.start_settings()
        except Exception as exc:
            import traceback
            detail = traceback.format_exc()
            messagebox.showerror("Ayarlar Açılamadı",
                                 f"{exc}\n\nDetay:\n{detail}")
            self.show_main_menu()

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

        # Sol Panel (Kontroller) – kaydırılabilir
        _lp_wrapper = tk.Frame(main_container, bg='#020617', width=310)
        _lp_wrapper.pack(side='left', fill='y', padx=5, pady=5)
        _lp_wrapper.pack_propagate(False)

        _lp_canvas = tk.Canvas(_lp_wrapper, bg='#020617', highlightthickness=0)
        _lp_scrollbar = tk.Scrollbar(_lp_wrapper, orient='vertical', command=_lp_canvas.yview)
        _lp_canvas.configure(yscrollcommand=_lp_scrollbar.set)
        _lp_scrollbar.pack(side='right', fill='y')
        _lp_canvas.pack(side='left', fill='both', expand=True)

        left_panel = tk.Frame(_lp_canvas, bg='#020617')
        _lp_win_id = _lp_canvas.create_window((0, 0), window=left_panel, anchor='nw')

        def _lp_on_configure(event):
            _lp_canvas.configure(scrollregion=_lp_canvas.bbox('all'))
        left_panel.bind('<Configure>', _lp_on_configure)
        _lp_canvas.bind('<Configure>', lambda e: _lp_canvas.itemconfig(_lp_win_id, width=e.width))

        def _lp_mousewheel(event):
            _lp_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        _lp_canvas.bind('<MouseWheel>', _lp_mousewheel)
        left_panel.bind('<MouseWheel>', _lp_mousewheel)
        
        # Yöntem Seçimi (Combobox)
        tk.Label(left_panel, text="Algoritma Seçimi",
                 font=('Segoe UI', 11, 'bold'), bg='#020617', fg='#e5e7eb').pack(pady=(10, 5))
        
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
        # ── Proje Adı Değiştir ────────────────────────────────────────────────
        tk.Frame(left_panel, bg='#1f2937', height=1).pack(fill='x', padx=10, pady=(0, 6))
        tk.Label(left_panel, text="Proje Adı",
                 font=('Segoe UI', 10, 'bold'), bg='#020617', fg='#e5e7eb').pack(anchor='w', padx=14)
        _proj_name_var = tk.StringVar(value=self.current_project.get('name', ''))
        _proj_name_entry = tk.Entry(left_panel, textvariable=_proj_name_var,
                                    font=('Segoe UI', 10), bg='#1e293b', fg='#e5e7eb',
                                    insertbackground='white', relief='flat', bd=4)
        _proj_name_entry.pack(fill='x', padx=14, pady=(2, 0))

        def _rename_project():
            new_name = _proj_name_var.get().strip()
            if not new_name:
                messagebox.showwarning("Uyarı", "Proje adı boş olamaz!")
                return
            if new_name == self.current_project.get('name'):
                return
            try:
                self.cursor.execute(
                    'UPDATE projects SET name=?, updated_date=? WHERE id=?',
                    (new_name, datetime.datetime.now().isoformat(), self.current_project['id'])
                )
                self.conn.commit()
                self.current_project['name'] = new_name
                messagebox.showinfo("Kaydedildi", f"Proje adı '{new_name}' olarak güncellendi.")
            except sqlite3.IntegrityError:
                messagebox.showerror("Hata", "Bu isimde başka bir proje zaten var!")
            except Exception as e:
                messagebox.showerror("Hata", f"Kayıt hatası: {e}")

        tk.Button(left_panel, text="✏ Adı Değiştir",
                  bg='#7c3aed', fg='white', font=('Segoe UI', 9, 'bold'),
                  bd=0, padx=8, pady=4,
                  command=_rename_project).pack(anchor='e', padx=14, pady=(3, 6))

        # ── Varsayılan Algoritma ──────────────────────────────────────────────
        tk.Frame(left_panel, bg='#1f2937', height=1).pack(fill='x', padx=10, pady=(4, 6))
        tk.Label(left_panel, text="Varsayılan Algoritma",
                 font=('Segoe UI', 10, 'bold'), bg='#020617', fg='#e5e7eb').pack(anchor='w', padx=14)
        tk.Label(left_panel, text="ROI bazlı algoritma seçilmemişse kullanılır",
                 font=('Segoe UI', 8), bg='#020617', fg='#6b7280').pack(anchor='w', padx=14)

        self.algo_combo_settings = ttk.Combobox(left_panel, values=list(self.algo_options_map.keys()),
                                                state='readonly', font=('Arial', 10))
        current_algo = self.current_project.get('algorithm', 'SSIM')
        for display_name, internal_name in self.algo_options_map.items():
            if internal_name == current_algo:
                self.algo_combo_settings.set(display_name)
                break
        else:
            self.algo_combo_settings.current(0)
        self.algo_combo_settings.pack(fill='x', padx=14, pady=(4, 2))

        _thresh_row = tk.Frame(left_panel, bg='#020617')
        _thresh_row.pack(fill='x', padx=14, pady=(0, 4))
        tk.Label(_thresh_row, text="Varsayılan Eşik:", bg='#020617', fg='#9ca3af',
                 font=('Segoe UI', 9)).pack(side='left')
        self.threshold_var = tk.DoubleVar(value=self.current_project.get('algo_threshold', 0.75))
        tk.Entry(_thresh_row, textvariable=self.threshold_var, width=7,
                 font=('Arial', 10), bg='#1e293b', fg='#e5e7eb',
                 insertbackground='white', relief='flat').pack(side='left', padx=6)

        # ── İzleme Hızı ───────────────────────────────────────────────────────
        tk.Frame(left_panel, bg='#1f2937', height=1).pack(fill='x', padx=10, pady=(6, 4))
        tk.Label(left_panel, text="İzleme Hızı (ms)",
                 font=('Segoe UI', 10, 'bold'), bg='#020617', fg='#e5e7eb').pack(anchor='w', padx=14)
        interval_frame = tk.Frame(left_panel, bg='#020617')
        interval_frame.pack(pady=4, padx=14, fill='x')
        self.interval_var = tk.IntVar(value=self.monitor_interval)
        tk.Spinbox(interval_frame, from_=100, to=10000, increment=100,
                   textvariable=self.interval_var, width=8,
                   font=('Arial', 11),
                   bg="#1e293b", fg="#e5e7eb",
                   buttonbackground="#334155",
                   relief="flat").pack(side='left')
        tk.Label(interval_frame, text="ms  (min 100)",
                 bg="#020617", fg="#9ca3af",
                 font=("Segoe UI", 9)).pack(side='left', padx=6)

        # ── Genel Ayarları Kaydet ─────────────────────────────────────────────
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
                self.current_project['algorithm'] = algo_internal
                self.current_project['algo_threshold'] = thresh

                try:
                    new_interval = int(self.interval_var.get())
                    self.monitor_interval = max(100, new_interval)
                except Exception:
                    pass

                messagebox.showinfo("Başarılı", "Genel ayarlar kaydedildi!")
            except Exception as e:
                messagebox.showerror("Hata", f"Kaydetme hatası: {e}")

        ttk.Button(left_panel, text="💾 Genel Ayarları Kaydet", command=save_settings,
                   style="Success.TButton").pack(pady=(8, 12), padx=14, fill='x')

        # score/result/detail label'lar update_validation için gerekli (gizli)
        self.score_label  = tk.Label(left_panel, text="", bg='#020617', fg='#020617')
        self.result_label = tk.Label(left_panel, text="", bg='#020617', fg='#020617')
        self.detail_label = tk.Label(left_panel, text="", bg='#020617', fg='#020617')
        # ROI label var (eski save_roi_image uyumluluğu için tutuldu)
        self.roi_label_var = tk.StringVar(value="OK")

        # ── ROI Yönetimi ──────────────────────────────────────────────────────
        tk.Frame(left_panel, bg='#1f2937', height=1).pack(fill='x', padx=10, pady=(10, 4))
        tk.Label(left_panel, text="ROI Bölgeleri",
                 font=('Segoe UI', 11, 'bold'), bg='#020617', fg='#e5e7eb').pack(anchor='w', padx=14)

        _roi_lb_outer = tk.Frame(left_panel, bg='#020617')
        _roi_lb_outer.pack(fill='x', padx=10, pady=2)

        _roi_lb = tk.Listbox(
            _roi_lb_outer, bg='#1e293b', fg='#e5e7eb',
            font=('Segoe UI', 9), height=4,
            selectbackground='#2563eb', activestyle='none',
            exportselection=False
        )
        _roi_lb.pack(side='left', fill='x', expand=True)
        _roi_sb = tk.Scrollbar(_roi_lb_outer, command=_roi_lb.yview)
        _roi_sb.pack(side='right', fill='y')
        _roi_lb.config(yscrollcommand=_roi_sb.set)

        def _refresh_roi_lb():
            _roi_lb.delete(0, tk.END)
            for r in self.current_project.get('roi_list', []):
                _, _, w, h = r['coords']
                _roi_lb.insert(tk.END, f"  {r['name']}  ({w}×{h})")

        _refresh_roi_lb()

        _roi_btn_row = tk.Frame(left_panel, bg='#020617')
        _roi_btn_row.pack(fill='x', padx=10, pady=2)

        def _save_roi_list_to_db(roi_list):
            """roi_list'i DB'ye yaz (algorithm + threshold dahil)."""
            roi_json = json.dumps([
                {'name': r['name'], 'x': r['coords'][0], 'y': r['coords'][1],
                 'w': r['coords'][2], 'h': r['coords'][3],
                 'algorithm': r.get('algorithm') or None,
                 'threshold': r.get('threshold') or None}
                for r in roi_list
            ], ensure_ascii=False)
            self.cursor.execute(
                'UPDATE projects SET roi_list=?, updated_date=? WHERE id=?',
                (roi_json, datetime.datetime.now().isoformat(), self.current_project['id'])
            )
            self.conn.commit()

        def _rename_roi_from_settings():
            sel = _roi_lb.curselection()
            if not sel:
                messagebox.showwarning("Uyarı", "Yeniden adlandırılacak ROI'yi seçin!")
                return
            idx = sel[0]
            roi_list = self.current_project.get('roi_list', [])
            if idx >= len(roi_list):
                return
            old_name = roi_list[idx]['name']
            new_name = simpledialog.askstring(
                "ROI Yeniden Adlandır",
                f"'{old_name}' için yeni isim:",
                initialvalue=old_name
            )
            if not new_name or new_name.strip() == old_name:
                return
            new_name = new_name.strip()
            # Aynı isim başka ROI'de var mı?
            if any(r['name'] == new_name for i, r in enumerate(roi_list) if i != idx):
                messagebox.showwarning("Uyarı", f"'{new_name}' ismi zaten kullanılıyor!")
                return
            roi_list[idx]['name'] = new_name
            self.current_project['roi_list'] = roi_list
            try:
                _save_roi_list_to_db(roi_list)
                # ML model kaydındaki roi_name'i de güncelle
                self.cursor.execute(
                    'UPDATE roi_ml_models SET roi_name=?, updated_date=? WHERE project_id=? AND roi_name=?',
                    (new_name, datetime.datetime.now().isoformat(),
                     self.current_project['id'], old_name)
                )
                self.conn.commit()
                # Bellekteki ml_models cache'ini temizle (eski key geçersiz)
                self.ml_models = {k: v for k, v in self.ml_models.items() if k[1] != old_name}
            except Exception as e:
                messagebox.showerror("Hata", f"Kayıt hatası: {e}")
                return
            _refresh_roi_lb()

        def _delete_roi_from_settings():
            sel = _roi_lb.curselection()
            if not sel:
                messagebox.showwarning("Uyarı", "Silinecek ROI'yi seçin!")
                return
            idx = sel[0]
            roi_list = self.current_project.get('roi_list', [])
            if idx >= len(roi_list):
                return
            name = roi_list[idx]['name']
            if not messagebox.askyesno("Sil", f"'{name}' ROI'si silinsin mi?"):
                return
            roi_list.pop(idx)
            self.current_project['roi_list'] = roi_list
            self.reference_rois = []
            for rc in roi_list:
                rx, ry, rw, rh = rc['coords']
                self.reference_rois.append(self.reference_image[ry:ry+rh, rx:rx+rw])
            try:
                _save_roi_list_to_db(roi_list)
            except Exception as e:
                messagebox.showerror("Hata", f"ROI kayıt hatası: {e}")
            _refresh_roi_lb()

        tk.Button(
            _roi_btn_row, text="+ Çiz / Düzenle",
            bg='#2563eb', fg='white', font=('Segoe UI', 9, 'bold'),
            bd=0, padx=6, pady=4,
            command=lambda: self._open_settings_roi_editor(_refresh_roi_lb)
        ).pack(side='left', fill='x', expand=True, padx=(0, 2))

        tk.Button(
            _roi_btn_row, text="✏ Yeniden Adlandır",
            bg='#7c3aed', fg='white', font=('Segoe UI', 9, 'bold'),
            bd=0, padx=6, pady=4,
            command=_rename_roi_from_settings
        ).pack(side='left', fill='x', expand=True, padx=(2, 2))

        tk.Button(
            _roi_btn_row, text="✕ Sil",
            bg='#dc2626', fg='white', font=('Segoe UI', 9, 'bold'),
            bd=0, padx=6, pady=4,
            command=_delete_roi_from_settings
        ).pack(side='left', fill='x', expand=True, padx=(2, 0))

        # ── Seçili ROI Ayarları + Veri Toplama + Eğitim ─────────────────────────
        tk.Frame(left_panel, bg='#1f2937', height=1).pack(fill='x', padx=10, pady=(10, 4))
        tk.Label(left_panel, text="Seçili ROI — Eğitim & Ayarlar",
                 font=('Segoe UI', 11, 'bold'), bg='#020617', fg='#e5e7eb').pack(anchor='w', padx=14)

        _roi_sel_frame = tk.Frame(left_panel, bg='#1e293b', bd=1, relief='flat')
        _roi_sel_frame.pack(fill='x', padx=10, pady=4)

        # Seçili ROI adı
        _roi_sel_name = tk.Label(_roi_sel_frame, text="— Yukarıdan bir ROI seçin —",
                                  bg='#1e293b', fg='#94a3b8',
                                  font=('Segoe UI', 9, 'italic'))
        _roi_sel_name.pack(anchor='w', padx=8, pady=(6, 2))

        # Görüntü sayacı (OK / NOK)
        _roi_count_lbl = tk.Label(_roi_sel_frame, text="OK: 0   NOK: 0",
                                   bg='#1e293b', fg='#6b7280',
                                   font=('Segoe UI', 9))
        _roi_count_lbl.pack(anchor='w', padx=8, pady=(0, 4))

        # Görüntü kayıt butonları
        _capture_row = tk.Frame(_roi_sel_frame, bg='#1e293b')
        _capture_row.pack(fill='x', padx=8, pady=(0, 6))

        def _count_roi_images(roi_name):
            """Disk üzerindeki OK/NOK görüntü sayısını döndür."""
            proj_folder = os.path.join('roi_images', self._safe_name(self.current_project['name']))
            ok_dir  = os.path.join(proj_folder, self._safe_name(roi_name), 'OK')
            nok_dir = os.path.join(proj_folder, self._safe_name(roi_name), 'NOK')
            ok_n  = len([f for f in os.listdir(ok_dir)  if f.lower().endswith(('.jpg','.png'))]) if os.path.isdir(ok_dir)  else 0
            nok_n = len([f for f in os.listdir(nok_dir) if f.lower().endswith(('.jpg','.png'))]) if os.path.isdir(nok_dir) else 0
            return ok_n, nok_n

        def _update_count_label():
            sel = _roi_lb.curselection()
            if not sel:
                _roi_count_lbl.config(text="OK: 0   NOK: 0", fg='#6b7280')
                return
            roi_list = self.current_project.get('roi_list', [])
            if sel[0] >= len(roi_list):
                return
            roi_name = roi_list[sel[0]]['name']
            ok_n, nok_n = _count_roi_images(roi_name)
            color = '#22c55e' if ok_n >= 5 and nok_n >= 5 else '#f59e0b'
            _roi_count_lbl.config(text=f"OK: {ok_n}   NOK: {nok_n}", fg=color)

        def _save_single_roi_image(label: str):
            """Sadece seçili ROI'nin görüntüsünü kaydet (OK veya NOK)."""
            sel = _roi_lb.curselection()
            if not sel:
                messagebox.showwarning("Uyarı", "Önce listeden bir ROI seçin!")
                return
            roi_list = self.current_project.get('roi_list', [])
            if sel[0] >= len(roi_list):
                return
            rc = roi_list[sel[0]]
            roi_name = rc['name']

            frame = None
            source = self.get_camera_source(self.current_project['camera_id'])
            if source == "HIKROBOT" and self.hik_camera:
                frame = self.hik_camera.get_frame()
            else:
                frame = self._get_latest_frame()

            if frame is None:
                messagebox.showwarning("Uyarı", "Kamera görüntüsü alınamadı!")
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_img, w_img = frame_rgb.shape[:2]
            x, y, w, h = rc['coords']
            x = max(0, min(int(x), w_img - 1))
            y = max(0, min(int(y), h_img - 1))
            w = max(1, min(int(w), w_img - x))
            h = max(1, min(int(h), h_img - y))
            crop = frame_rgb[y:y+h, x:x+w]
            if crop.size == 0:
                messagebox.showwarning("Uyarı", "ROI bölgesi geçersiz (boyut 0)!")
                return

            proj_folder = os.path.join('roi_images', self._safe_name(self.current_project['name']))
            sub_folder  = os.path.join(proj_folder, self._safe_name(roi_name), label)
            os.makedirs(sub_folder, exist_ok=True)

            unique = int(datetime.datetime.now().timestamp() * 1000)
            filename = os.path.join(sub_folder, f"{self._safe_name(roi_name)}_{unique}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            _update_count_label()

        tk.Button(_capture_row, text="✅ OK Kaydet",
                  bg='#16a34a', fg='white', font=('Segoe UI', 9, 'bold'),
                  bd=0, padx=6, pady=5,
                  command=lambda: _save_single_roi_image('OK')
                  ).pack(side='left', fill='x', expand=True, padx=(0, 2))

        tk.Button(_capture_row, text="❌ NOK Kaydet",
                  bg='#dc2626', fg='white', font=('Segoe UI', 9, 'bold'),
                  bd=0, padx=6, pady=5,
                  command=lambda: _save_single_roi_image('NOK')
                  ).pack(side='left', fill='x', expand=True, padx=(2, 0))

        # Algoritma dropdown (per-ROI)
        _roi_algo_row = tk.Frame(_roi_sel_frame, bg='#1e293b')
        _roi_algo_row.pack(fill='x', padx=8, pady=2)
        tk.Label(_roi_algo_row, text="Algoritma:", bg='#1e293b', fg='#e5e7eb',
                 font=('Segoe UI', 9), width=10, anchor='w').pack(side='left')
        _roi_algo_var = tk.StringVar()
        ttk.Combobox(_roi_algo_row,
                     values=list(self.algo_options_map.keys()),
                     textvariable=_roi_algo_var,
                     state='readonly', width=22,
                     font=('Arial', 9)).pack(side='left', fill='x', expand=True)

        # Eşik (per-ROI)
        _roi_thresh_row = tk.Frame(_roi_sel_frame, bg='#1e293b')
        _roi_thresh_row.pack(fill='x', padx=8, pady=2)
        tk.Label(_roi_thresh_row, text="Eşik:", bg='#1e293b', fg='#e5e7eb',
                 font=('Segoe UI', 9), width=10, anchor='w').pack(side='left')
        _roi_thresh_var = tk.DoubleVar(value=0.75)
        tk.Entry(_roi_thresh_row, textvariable=_roi_thresh_var,
                 width=8, font=('Arial', 9),
                 bg='#0f172a', fg='#e5e7eb',
                 insertbackground='white', relief='flat').pack(side='left')

        # Kaydet + Eğit butonları
        _roi_cfg_btn_row = tk.Frame(_roi_sel_frame, bg='#1e293b')
        _roi_cfg_btn_row.pack(fill='x', padx=8, pady=(4, 8))

        def _roi_cfg_save():
            sel = _roi_lb.curselection()
            if not sel:
                messagebox.showwarning("Uyarı", "Önce listeden bir ROI seçin!", parent=self.root)
                return
            idx = sel[0]
            roi_list = self.current_project.get('roi_list', [])
            if idx >= len(roi_list):
                return
            display = _roi_algo_var.get()
            algo = self.algo_options_map.get(display) if display else None
            try:
                thr = float(_roi_thresh_var.get())
            except ValueError:
                thr = 0.75
            roi_list[idx]['algorithm'] = algo
            roi_list[idx]['threshold'] = thr
            self.current_project['roi_list'] = roi_list
            _save_roi_list_to_db(roi_list)
            messagebox.showinfo("Kaydedildi", f"'{roi_list[idx]['name']}' ROI ayarları kaydedildi.")

        def _roi_train_selected():
            sel = _roi_lb.curselection()
            if not sel:
                messagebox.showwarning("Uyarı", "Önce listeden bir ROI seçin!", parent=self.root)
                return
            idx = sel[0]
            roi_list = self.current_project.get('roi_list', [])
            if idx >= len(roi_list):
                return
            display = _roi_algo_var.get()
            algo = self.algo_options_map.get(display, 'ML_MODEL') if display else 'ML_MODEL'
            roi_name = roi_list[idx]['name']
            self._train_single_roi(roi_name, algo)
            _update_count_label()

        tk.Button(_roi_cfg_btn_row, text="💾 Ayar Kaydet",
                  bg='#0ea5e9', fg='white', font=('Segoe UI', 9, 'bold'),
                  bd=0, padx=6, pady=4,
                  command=_roi_cfg_save).pack(side='left', fill='x', expand=True, padx=(0, 2))

        tk.Button(_roi_cfg_btn_row, text="🧠 Bu ROI'yi Eğit",
                  bg='#7c3aed', fg='white', font=('Segoe UI', 9, 'bold'),
                  bd=0, padx=6, pady=4,
                  command=_roi_train_selected).pack(side='left', fill='x', expand=True, padx=(2, 0))

        # Listbox seçim → paneli doldur
        def _on_roi_lb_select(event=None):
            sel = _roi_lb.curselection()
            if not sel:
                return
            idx = sel[0]
            roi_list = self.current_project.get('roi_list', [])
            if idx >= len(roi_list):
                return
            rc = roi_list[idx]
            _roi_sel_name.config(text=f"  {rc['name']}", fg='#e5e7eb')
            roi_algo_internal = rc.get('algorithm') or self.current_project.get('algorithm', 'SSIM')
            for display_n, internal_n in self.algo_options_map.items():
                if internal_n == roi_algo_internal:
                    _roi_algo_var.set(display_n)
                    break
            roi_thr = rc.get('threshold') or self.current_project.get('algo_threshold', 0.75)
            _roi_thresh_var.set(round(float(roi_thr), 4))
            _update_count_label()

        _roi_lb.bind('<<ListboxSelect>>', _on_roi_lb_select)

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

    def _open_settings_roi_editor(self, on_save_callback=None):
        """Ayarlar ekranında referans görüntü üzerinde ROI çiz/sil/kaydet penceresi."""
        if self.reference_image is None:
            messagebox.showwarning("Uyarı", "Proje referans görüntüsü yüklenemedi!")
            return

        win = tk.Toplevel(self.root)
        win.title("ROI Düzenle")
        win.geometry("960x680")
        win.configure(bg="#0f172a")
        win.grab_set()

        # Çalışma kopyası – per-ROI algorithm/threshold da korunur
        edit_rois = [{'name': r['name'], 'coords': tuple(r['coords']),
                      'algorithm': r.get('algorithm'), 'threshold': r.get('threshold')}
                     for r in self.current_project.get('roi_list', [])]

        # ── Sol Panel ────────────────────────────────────────────────────────
        left = tk.Frame(win, bg="#020617", width=230)
        left.pack(side='left', fill='y', padx=(8, 4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="ROI Listesi", bg="#020617", fg="#e5e7eb",
                 font=("Segoe UI", 11, "bold")).pack(pady=(14, 4))

        lb = tk.Listbox(left, bg="#1e293b", fg="#e5e7eb",
                        font=("Segoe UI", 9), selectbackground="#2563eb",
                        activestyle='none')
        lb.pack(fill='both', expand=True, padx=8, pady=4)

        def _refresh():
            lb.delete(0, tk.END)
            for r in edit_rois:
                _, _, rw, rh = r['coords']
                lb.insert(tk.END, f"  {r['name']}  ({rw}×{rh})")

        _refresh()

        def _delete_selected():
            sel = lb.curselection()
            if not sel:
                return
            edit_rois.pop(sel[0])
            _refresh()
            _redraw()

        tk.Button(left, text="✕  Seçiliyi Sil",
                  bg="#dc2626", fg="white", font=("Segoe UI", 9, "bold"),
                  bd=0, padx=8, pady=6,
                  command=_delete_selected).pack(fill='x', padx=8, pady=(4, 2))

        tk.Label(left,
                 text="Sürükle-bırak → Yeni ROI\n🖱 Tekerlek → Zoom",
                 bg="#020617", fg="#9ca3af",
                 font=("Segoe UI", 8), justify='left').pack(anchor='w', padx=12, pady=(10, 4))

        btn_bottom = tk.Frame(left, bg="#020617")
        btn_bottom.pack(side='bottom', fill='x', padx=8, pady=8)

        def _save_and_close():
            self.current_project['roi_list'] = edit_rois
            self.reference_rois = []
            for rc in edit_rois:
                rx, ry, rw, rh = rc['coords']
                self.reference_rois.append(self.reference_image[ry:ry+rh, rx:rx+rw])
            if edit_rois:
                self.current_project['roi'] = edit_rois[0]['coords']
            try:
                roi_json = json.dumps([
                    {'name': r['name'], 'x': r['coords'][0], 'y': r['coords'][1],
                     'w': r['coords'][2], 'h': r['coords'][3],
                     'algorithm': r.get('algorithm'), 'threshold': r.get('threshold')}
                    for r in edit_rois
                ], ensure_ascii=False)
                first = edit_rois[0]['coords'] if edit_rois else (0, 0, 0, 0)
                self.cursor.execute(
                    '''UPDATE projects
                       SET roi_list=?, roi_x=?, roi_y=?, roi_width=?, roi_height=?,
                           updated_date=?
                       WHERE id=?''',
                    (roi_json, first[0], first[1], first[2], first[3],
                     datetime.datetime.now().isoformat(), self.current_project['id'])
                )
                self.conn.commit()
            except Exception as e:
                messagebox.showerror("Hata", f"Kayıt hatası: {e}", parent=win)
                return
            if on_save_callback:
                on_save_callback()
            win.destroy()

        tk.Button(btn_bottom, text="💾  Kaydet ve Kapat",
                  bg="#16a34a", fg="white", font=("Segoe UI", 10, "bold"),
                  bd=0, padx=8, pady=8,
                  command=_save_and_close).pack(fill='x', pady=(0, 4))

        tk.Button(btn_bottom, text="İptal",
                  bg="#374151", fg="white", font=("Segoe UI", 9),
                  bd=0, padx=8, pady=6,
                  command=win.destroy).pack(fill='x')

        # ── Sağ Panel – Canvas ───────────────────────────────────────────────
        canvas = tk.Canvas(win, bg="#2c3e50", cursor="cross")
        canvas.pack(side='right', expand=True, fill='both', padx=(4, 8), pady=8)

        zoom = [1.0]
        offset = [None, None]  # [ox, oy] – None = ortala
        _state = {'start': None, 'temp': None, 'active': False}

        def _base_scale():
            img_h, img_w = self.reference_image.shape[:2]
            cw = canvas.winfo_width() or 700
            ch = canvas.winfo_height() or 560
            return min(cw / img_w, ch / img_h) * 0.95, img_w, img_h, cw, ch

        def _params():
            base, img_w, img_h, cw, ch = _base_scale()
            sc = base * zoom[0]
            nw, nh = int(img_w * sc), int(img_h * sc)
            ox = offset[0] if offset[0] is not None else max(0, (cw - nw) // 2)
            oy = offset[1] if offset[1] is not None else max(0, (ch - nh) // 2)
            return sc, ox, oy

        def _redraw():
            sc, ox, oy = _params()
            img_h, img_w = self.reference_image.shape[:2]
            nw = max(1, int(img_w * sc))
            nh = max(1, int(img_h * sc))
            img_r = cv2.resize(self.reference_image, (nw, nh))
            photo = ImageTk.PhotoImage(Image.fromarray(img_r))
            canvas._photo = photo  # referansı tut, GC'den koru
            canvas.delete('all')
            canvas.create_image(ox, oy, anchor='nw', image=photo, tags='bg')
            for i, roi in enumerate(edit_rois):
                x, y, rw, rh = roi['coords']
                col = self._roi_colors[i % len(self._roi_colors)]
                cx1 = int(x * sc) + ox
                cy1 = int(y * sc) + oy
                cx2 = int((x + rw) * sc) + ox
                cy2 = int((y + rh) * sc) + oy
                canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                        outline=col, width=2, tags=f"roi_{i}")
                canvas.create_text(cx1 + 4, cy1 + 4, anchor='nw',
                                   text=roi['name'], fill=col,
                                   font=('Arial', 9, 'bold'), tags=f"roi_{i}")

        win.after(120, _redraw)
        canvas.bind('<Configure>', lambda __: _redraw())

        def _on_zoom(event):
            base, img_w, img_h, cw, ch = _base_scale()
            old_sc = base * zoom[0]
            old_ox = offset[0] if offset[0] is not None else (cw - int(img_w * old_sc)) // 2
            old_oy = offset[1] if offset[1] is not None else (ch - int(img_h * old_sc)) // 2
            # Fare altındaki görüntü koordinatı
            img_x = (event.x - old_ox) / old_sc
            img_y = (event.y - old_oy) / old_sc
            zoom[0] = min(5.0, zoom[0] * 1.15) if event.delta > 0 else max(0.2, zoom[0] / 1.15)
            new_sc = base * zoom[0]
            offset[0] = int(event.x - img_x * new_sc)
            offset[1] = int(event.y - img_y * new_sc)
            _redraw()

        canvas.bind('<MouseWheel>', _on_zoom)

        def _md(event):
            _state['start'] = (event.x, event.y)
            _state['active'] = True

        def _mm(event):
            if not _state['active']:
                return
            if _state['temp']:
                canvas.delete(_state['temp'])
            x1, y1 = _state['start']
            _state['temp'] = canvas.create_rectangle(
                x1, y1, event.x, event.y,
                outline='yellow', width=2, dash=(5, 5))

        def _mu(event):
            if not _state['active']:
                return
            _state['active'] = False
            if _state['temp']:
                canvas.delete(_state['temp'])
                _state['temp'] = None

            sc, ox, oy = _params()
            x1 = (min(_state['start'][0], event.x) - ox) / sc
            y1 = (min(_state['start'][1], event.y) - oy) / sc
            x2 = (max(_state['start'][0], event.x) - ox) / sc
            y2 = (max(_state['start'][1], event.y) - oy) / sc
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = max(0, int(x2)), max(0, int(y2))
            rw, rh = x2 - x1, y2 - y1

            if rw < 10 or rh < 10:
                messagebox.showwarning("Uyarı", "ROI çok küçük!", parent=win)
                return

            name = simpledialog.askstring(
                "ROI İsmi", "Bu bölge için isim girin:",
                initialvalue=f"ROI_{len(edit_rois) + 1}", parent=win)
            if not name:
                return

            edit_rois.append({'name': name, 'coords': (x1, y1, rw, rh)})
            _refresh()
            _redraw()

        canvas.bind('<ButtonPress-1>', _md)
        canvas.bind('<B1-Motion>', _mm)
        canvas.bind('<ButtonRelease-1>', _mu)

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
            
            # Proje seviyesi varsayılan algoritma ve eşik
            display_name = self.algo_combo_settings.get()
            proj_algorithm = self.algo_options_map.get(display_name, 'SSIM')
            try:
                proj_threshold = float(self.threshold_var.get())
            except ValueError:
                proj_threshold = 0.75

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

            ML_ALGOS_VAL = (
                "ML_MODEL", "ML_MODEL_SIFT", "ML_MODEL_DAISY",
                "ML_MODEL_HAAR", "ML_MODEL_CENSURE", "ML_MODEL_MBLBP",
                "ML_MODEL_GLCM", "ML_MODEL_LBP", "ML_MODEL_GABOR", "ML_MODEL_FISHER",
            )

            for idx, rc in enumerate(roi_list):
                x, y, w, h = rc['coords']
                # Per-ROI algoritma ve eşik
                algorithm = rc.get('algorithm') or proj_algorithm
                threshold = float(rc.get('threshold') or proj_threshold)
                current_roi = frame_rgb[y:y+h, x:x+w]
                ref_roi = ref_rois[idx] if idx < len(ref_rois) else self.reference_roi

                if algorithm in ML_ALGOS_VAL:
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
                if algorithm in ML_ALGOS_VAL:
                    label = f"{rc.get('name', f'ROI_{idx+1}')}: {'OK' if status else 'NOK'}"
                else:
                    label = f"{rc.get('name', f'ROI_{idx+1}')}: {score:.0%}"
                cv2.putText(display_frame, label, (x, max(y-5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

            # Genel Durum Geri Bildirimi (proje seviyesi algoritma ile özet)
            self.score_label.config(text=f"{passed_count}/{len(roi_list)} ROI geçti")
            
            all_ok = passed_count == len(roi_list)
            self.result_label.config(
                text=f"DURUM: {'OK' if all_ok else 'NG'} ({passed_count}/{len(roi_list)})",
                fg='#27ae60' if all_ok else '#e74c3c'
            )
            
            # Canvas'a göster
            self.display_monitoring_frame(display_frame)

        # Sabit 1 saniyelik döngü – kamera türünden bağımsız
        self.root.after(self.monitor_interval, self.update_validation)
    
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
        rois = []
        if roi_list_raw:
            try:
                rois = json.loads(roi_list_raw)
                roi_tuples = [(r['x'], r['y'], r['w'], r['h']) for r in rois]
                roi_names  = [r.get('name', f'ROI_{i+1}') for i, r in enumerate(rois)]
            except Exception:
                rois = []
                roi_tuples = []
                roi_names  = []

        # JSON boşsa veya parse edilemezse eski tek-ROI sütunlarına düş
        if not roi_list_raw or not roi_tuples:
            roi_tuples = [(row[3], row[4], row[5], row[6])]
            roi_names  = ['ROI_1']
            rois = [{}]  # per-ROI alanlar yok

        self.current_project = {
            'id': project_id,
            'name': row[0],
            'camera_id': row[1],
            'roi': roi_tuples[0],         # geriye uyumluluk
            'roi_list': [
                {
                    'name': roi_names[i],
                    'coords': roi_tuples[i],
                    'algorithm': rois[i].get('algorithm') if i < len(rois) else None,
                    'threshold': rois[i].get('threshold') if i < len(rois) else None,
                }
                for i in range(len(roi_tuples))
            ],
            'algorithm': row[8] or 'SSIM',
            'algo_threshold': row[9] if row[9] is not None else 0.75,
        }

        # Referans görüntüyü yükle
        if row[2] is None:
            raise ValueError("Projede referans görüntü yok. Lütfen projeyi yeniden oluşturun.")
        img_array = np.frombuffer(row[2], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Referans görüntü okunamadı (bozuk veri).")
        self.reference_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # İlk ROI'nin referans kesimini oluştur (geriye uyumluluk)
        x, y, w, h = self.current_project['roi']
        if all(v is not None for v in (x, y, w, h)) and w > 0 and h > 0:
            self.reference_roi = self.reference_image[y:y+h, x:x+w]
        else:
            self.reference_roi = self.reference_image.copy()

        # Tüm ROI'lerin referans kesimlerini de oluştur
        self.reference_rois = []
        for rc in self.current_project['roi_list']:
            rx, ry, rw, rh = rc['coords']
            if all(v is not None for v in (rx, ry, rw, rh)) and rw > 0 and rh > 0:
                self.reference_rois.append(self.reference_image[ry:ry+rh, rx:rx+rw])
            else:
                self.reference_rois.append(self.reference_image.copy())


    
    def start_monitoring(self):
        """Kontrol ekranı – trigger tabanlı fotoğraf analizi"""
        for widget in self.root.winfo_children():
            widget.destroy()

        # ── Durum değişkeni ───────────────────────────────────────────────────
        # 'WAITING' | 'ANALYZING' | 'OK' | 'NOK'
        self._ctrl_state = 'WAITING'
        self._trigger_prev = False   # yükselen kenar tespiti için

        # ── Ana çerçeve ───────────────────────────────────────────────────────
        main_frame = tk.Frame(self.root, bg='#0f172a')
        main_frame.pack(expand=True, fill='both')

        # ── Üst başlık ────────────────────────────────────────────────────────
        header = tk.Frame(main_frame, bg='#020617', height=52)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Button(header, text="⏹ Durdur",
                  bg='#dc2626', fg='white',
                  font=('Segoe UI', 10, 'bold'),
                  bd=0, padx=14, pady=6,
                  command=self.stop_monitoring).pack(side='left', padx=10, pady=8)

        tk.Label(header, text=f"Proje: {self.current_project['name']}",
                 bg='#020617', fg='#e5e7eb',
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=10)

        _plc_on = self.plc.config.get("enabled", False)
        _plc_init_text = "PLC: Bağlanıyor..." if _plc_on else "PLC: Devre Dışı"
        _plc_init_fg   = '#f59e0b'            if _plc_on else '#6b7280'
        self.plc_status_label = tk.Label(header, text=_plc_init_text,
                                         bg='#020617', fg=_plc_init_fg,
                                         font=('Segoe UI', 10))
        self.plc_status_label.pack(side='right', padx=16)

        # ── İçerik: sol (operatör paneli) + sağ (kamera) ─────────────────────
        content = tk.Frame(main_frame, bg='#0f172a')
        content.pack(expand=True, fill='both')

        # Sol – Operatör bilgilendirme paneli
        left = tk.Frame(content, bg='#0f172a', width=320)
        left.pack(side='left', fill='y', padx=(12, 6), pady=12)
        left.pack_propagate(False)

        # Büyük durum kutusu
        self._ctrl_box = tk.Frame(left, bg='#1e3a5f', bd=0, relief='flat')
        self._ctrl_box.pack(fill='x', pady=(8, 4))

        self._ctrl_icon = tk.Label(self._ctrl_box, text="⏳",
                                   font=('Segoe UI', 48),
                                   bg='#1e3a5f', fg='white')
        self._ctrl_icon.pack(pady=(18, 4))

        self._ctrl_title = tk.Label(self._ctrl_box,
                                    text="PARÇAYI YERLEŞTİRİN",
                                    font=('Segoe UI', 16, 'bold'),
                                    bg='#1e3a5f', fg='white',
                                    wraplength=280, justify='center')
        self._ctrl_title.pack(pady=(0, 6))

        self._ctrl_sub = tk.Label(self._ctrl_box,
                                  text="PLC trigger sinyali bekleniyor...",
                                  font=('Segoe UI', 10),
                                  bg='#1e3a5f', fg='#93c5fd',
                                  wraplength=280, justify='center')
        self._ctrl_sub.pack(pady=(0, 18))

        # ── Operatör Adım Göstergesi ─────────────────────────────────────────
        steps_box = tk.Frame(left, bg='#0f172a')
        steps_box.pack(fill='x', pady=(8, 0))

        _step_cfgs = [
            (1, "Parçayı yerleştirin"),
            (2, "Butona basın → kontrol başlar"),
            (3, "Kontrol sonucunu bekleyin"),
        ]
        self._step_labels = {}
        for sn, stxt in _step_cfgs:
            sf = tk.Frame(steps_box, bg='#1e293b', bd=0)
            sf.pack(fill='x', pady=1)
            num_lbl = tk.Label(sf, text=f" {sn} ", bg='#334155', fg='#94a3b8',
                               font=('Segoe UI', 10, 'bold'), width=3)
            num_lbl.pack(side='left')
            txt_lbl = tk.Label(sf, text=stxt, bg='#1e293b', fg='#64748b',
                               font=('Segoe UI', 9), anchor='w')
            txt_lbl.pack(side='left', fill='x', expand=True, padx=8, pady=5)
            self._step_labels[sn] = (sf, num_lbl, txt_lbl)

        # PLC kapalıyken manuel test butonu
        if not self.plc.config.get("enabled", False):
            tk.Label(left, text="PLC bağlı değil – manuel mod",
                     bg='#0f172a', fg='#f59e0b',
                     font=('Segoe UI', 9, 'italic')).pack(pady=(8, 0))
            tk.Button(left, text="📷  Manuel Fotoğraf Çek",
                      bg='#0ea5e9', fg='white',
                      font=('Segoe UI', 10, 'bold'),
                      bd=0, padx=10, pady=8,
                      command=self._do_capture_and_analyze).pack(
                          fill='x', pady=(4, 0))

        # ROI sonuç listesi (NOK durumunda kırmızı gösterilir)
        tk.Label(left, text="ROI Kontrol Listesi",
                 bg='#0f172a', fg='#9ca3af',
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(10, 2))

        self._roi_result_frame = tk.Frame(left, bg='#0f172a')
        self._roi_result_frame.pack(fill='x')

        # ROI satırlarını başlangıçta oluştur
        self._roi_result_labels = {}
        for rc in self.current_project.get('roi_list', []):
            row_f = tk.Frame(self._roi_result_frame, bg='#1e293b')
            row_f.pack(fill='x', pady=1)
            dot = tk.Label(row_f, text="●", fg='#6b7280',
                           bg='#1e293b', font=('Segoe UI', 12))
            dot.pack(side='left', padx=(8, 4))
            lbl = tk.Label(row_f, text=rc['name'],
                           bg='#1e293b', fg='#9ca3af',
                           font=('Segoe UI', 10), anchor='w')
            lbl.pack(side='left', fill='x', expand=True, pady=4)
            stat = tk.Label(row_f, text="─",
                            bg='#1e293b', fg='#6b7280',
                            font=('Segoe UI', 10, 'bold'))
            stat.pack(side='right', padx=8)
            self._roi_result_labels[rc['name']] = (row_f, dot, lbl, stat)

        # Sağ – Kamera görüntüsü
        right = tk.Frame(content, bg='#111827')
        right.pack(side='right', expand=True, fill='both', padx=(6, 12), pady=12)

        self.monitor_canvas = tk.Canvas(right, bg='#111827', highlightthickness=0)
        self.monitor_canvas.pack(expand=True, fill='both')

        # ── Kamerayı başlat ───────────────────────────────────────────────────
        self.monitoring_active = True
        source = self.get_camera_source(self.current_project['camera_id'])
        if source == "HIKROBOT":
            if not self.hik_camera:
                self.hik_camera = HikrobotCamera()
                ok, msg = self.hik_camera.open()
                if not ok:
                    messagebox.showerror("Hata", f"Hikrobot Kamera açılamadı: {msg}")
                    return
            if not self.hik_camera.start():
                messagebox.showerror("Hata", "Hikrobot Kamera başlatılamadı.")
                return
        else:
            self._start_cam_reader(source)

        # Başlangıç durumunu göster (adım göstergesini aktive et)
        self._set_ctrl_state('WAITING')
        # Başlangıçta referans görseli göster – canvas render edildikten sonra (200ms)
        self.root.after(200, self._show_reference_preview)
        self.update_monitoring()

    def _set_ctrl_state(self, state: str, nok_names: list = None, detail: str = ''):
        """Operatör panelini ve ROI listesini verilen duruma göre güncelle."""
        if not self.monitoring_active:
            return
        if not hasattr(self, '_ctrl_box') or not self._ctrl_box.winfo_exists():
            return
        self._ctrl_state = state

        state_cfg = {
            'WAITING':   ('#1e3a5f', '#93c5fd', '⏳',
                          'PARÇAYI YERLEŞTİRİN',
                          'Parçayı doğru konuma yerleştirin, ardından butona basın'),
            'ANALYZING': ('#78350f', '#fcd34d', '🔍',
                          'KONTROL EDİLİYOR...',
                          'Lütfen bekleyin, hareket ettirmeyin'),
            'OK':        ('#14532d', '#86efac', '✅',
                          'PARÇA TAMAM  ✓',
                          'Kaynak işlemi başlatılıyor → Yeni parça için ① adımına dönün'),
            'NOK':       ('#7f1d1d', '#fca5a5', '❌',
                          'HATALI / EKSİK PARÇA',
                          'Kırmızı ROI\'leri kontrol edin → Parçayı düzeltin → Tekrar butona basın'),
        }.get(state, ('#1e3a5f', '#93c5fd', '⏳', '', ''))

        bg, fg_sub, icon, title, sub = state_cfg
        if detail:
            sub = f"{sub}  ({detail})"

        self._ctrl_box.config(bg=bg)
        self._ctrl_icon.config(text=icon, bg=bg)
        self._ctrl_title.config(text=title, bg=bg, fg='white')
        self._ctrl_sub.config(text=sub, bg=bg, fg=fg_sub)

        # Adım göstergelerini güncelle
        if hasattr(self, '_step_labels'):
            # Hangi adım aktif?
            active_step = {
                'WAITING': 1, 'ANALYZING': 3, 'OK': None, 'NOK': None
            }.get(state, 1)
            # WAITING → adım 1 & 2 aktif (parça koy + butona bas)
            for sn, (sf, num_lbl, txt_lbl) in self._step_labels.items():
                if state == 'WAITING':
                    if sn in (1, 2):
                        sf.config(bg='#1e3a5f')
                        num_lbl.config(bg='#2563eb', fg='white')
                        txt_lbl.config(bg='#1e3a5f', fg='#93c5fd')
                    else:
                        sf.config(bg='#1e293b')
                        num_lbl.config(bg='#334155', fg='#94a3b8')
                        txt_lbl.config(bg='#1e293b', fg='#64748b')
                elif state == 'ANALYZING':
                    if sn == 3:
                        sf.config(bg='#78350f')
                        num_lbl.config(bg='#b45309', fg='white')
                        txt_lbl.config(bg='#78350f', fg='#fcd34d')
                    else:
                        sf.config(bg='#1e293b')
                        num_lbl.config(bg='#334155', fg='#94a3b8')
                        txt_lbl.config(bg='#1e293b', fg='#64748b')
                elif state == 'OK':
                    sf.config(bg='#052e16')
                    num_lbl.config(bg='#166534', fg='white')
                    txt_lbl.config(bg='#052e16', fg='#86efac')
                elif state == 'NOK':
                    if sn in (1, 2):
                        sf.config(bg='#450a0a')
                        num_lbl.config(bg='#991b1b', fg='white')
                        txt_lbl.config(bg='#450a0a', fg='#fca5a5')
                    else:
                        sf.config(bg='#1e293b')
                        num_lbl.config(bg='#334155', fg='#94a3b8')
                        txt_lbl.config(bg='#1e293b', fg='#64748b')

        # ROI sonuç satırlarını güncelle
        nok_set = set(nok_names or [])
        for name, (row_f, dot, lbl, stat) in self._roi_result_labels.items():
            if state == 'WAITING' or state == 'ANALYZING':
                row_f.config(bg='#1e293b')
                dot.config(fg='#6b7280', bg='#1e293b')
                lbl.config(fg='#9ca3af', bg='#1e293b')
                stat.config(text='─', fg='#6b7280', bg='#1e293b')
            elif name in nok_set:
                row_f.config(bg='#450a0a')
                dot.config(fg='#ef4444', bg='#450a0a')
                lbl.config(fg='#fca5a5', bg='#450a0a')
                stat.config(text='✕ NOK', fg='#ef4444', bg='#450a0a')
            else:
                row_f.config(bg='#052e16')
                dot.config(fg='#22c55e', bg='#052e16')
                lbl.config(fg='#86efac', bg='#052e16')
                stat.config(text='✓ OK', fg='#22c55e', bg='#052e16')
    
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

                kernel = 'rbf' if use_sift else 'linear'
                svm_model = SVC(kernel=kernel, probability=True)

                min_per_class = min(np.bincount(
                    np.array([list(np.unique(y)).index(l) for l in y])
                ))
                can_split = min_per_class >= 2 and len(X) >= len(np.unique(y)) * 5
                if can_split:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y)
                    svm_model.fit(X_train, y_train)
                    acc = float(accuracy_score(y_test, svm_model.predict(X_test)))
                else:
                    svm_model.fit(X, y)
                    acc = float(accuracy_score(y, svm_model.predict(X)))
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

    def _train_single_roi(self, roi_name: str, algo_internal: str):
        """Tek bir ROI için seçili algoritmayı eğit ve DB'ye kaydet."""
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.mixture import GaussianMixture

        if not self.current_project:
            messagebox.showwarning("Uyarı", "Önce bir proje seçiniz!")
            return

        ML_ALGOS = (
            'ML_MODEL', 'ML_MODEL_SIFT', 'ML_MODEL_DAISY',
            'ML_MODEL_HAAR', 'ML_MODEL_CENSURE', 'ML_MODEL_MBLBP',
            'ML_MODEL_GLCM', 'ML_MODEL_LBP', 'ML_MODEL_GABOR', 'ML_MODEL_FISHER',
        )
        if algo_internal not in ML_ALGOS:
            messagebox.showwarning("Uyarı",
                "ROI bazlı eğitim yalnızca ML algoritmaları için geçerlidir.\n"
                "Seçilen algoritma ML tabanlı değil.")
            return

        project_name = self.current_project.get('name')
        project_id   = self.current_project.get('id')
        safe_proj    = self._safe_name(project_name)
        safe_roi     = self._safe_name(roi_name)

        roi_folder = os.path.join("roi_images", safe_proj, roi_name)
        if not os.path.isdir(roi_folder):
            messagebox.showwarning("Uyarı",
                f"'{roi_name}' için eğitim veri klasörü bulunamadı:\n{roi_folder}\n\n"
                "Önce 'ROI Kaydet' ile OK/NOK görüntüleri kaydedin.")
            return

        images, labels = self._ml_load_images(roi_folder, image_size=(64, 64))
        if images.size == 0 or labels.size == 0:
            messagebox.showwarning("Uyarı", f"'{roi_name}': eğitim verisi yok.")
            return

        if len(np.unique(labels)) < 2:
            messagebox.showwarning("Uyarı",
                f"'{roi_name}': En az 2 sınıf (OK + NOK) gerekli.")
            return

        try:
            use_sift    = algo_internal == 'ML_MODEL_SIFT'
            use_daisy   = algo_internal == 'ML_MODEL_DAISY'
            use_haar    = algo_internal == 'ML_MODEL_HAAR'
            use_censure = algo_internal == 'ML_MODEL_CENSURE'
            use_mblbp   = algo_internal == 'ML_MODEL_MBLBP'
            use_glcm    = algo_internal == 'ML_MODEL_GLCM'
            use_lbp     = algo_internal == 'ML_MODEL_LBP'
            use_gabor   = algo_internal == 'ML_MODEL_GABOR'
            use_fisher  = algo_internal == 'ML_MODEL_FISHER'

            gmm = None
            if use_sift:
                X = self._ml_extract_sift_features(images, n_keypoints=50)
                model_type_str = "SIFT_SVM"; model_suffix = "SIFT_SVM"
                params = {"kernel": "rbf", "probability": True, "image_size": (64, 64), "sift_n_keypoints": 50}
            elif use_daisy:
                X = self._ml_extract_daisy_features(images, step=32, radius=16, rings=2, histograms=6, orientations=8)
                model_type_str = "DAISY_SVM"; model_suffix = "DAISY_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
            elif use_haar:
                X = self._ml_extract_haar_features(images)
                model_type_str = "HAAR_SVM"; model_suffix = "HAAR_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
            elif use_censure:
                X = self._ml_extract_censure_features(images)
                model_type_str = "CENSURE_SVM"; model_suffix = "CENSURE_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
            elif use_mblbp:
                X = self._ml_extract_multiblock_lbp_features(images)
                model_type_str = "MBLBP_SVM"; model_suffix = "MBLBP_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
            elif use_glcm:
                X = self._ml_extract_glcm_features(images)
                model_type_str = "GLCM_SVM"; model_suffix = "GLCM_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
            elif use_lbp:
                X = self._ml_extract_lbp_features(images, radius=3)
                model_type_str = "LBP_SVM"; model_suffix = "LBP_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
            elif use_gabor:
                X = self._ml_extract_gabor_features(images)
                model_type_str = "GABOR_SVM"; model_suffix = "GABOR_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64)}
            elif use_fisher:
                desc_list = self._ml_extract_fisher_features(images, gmm=None, n_components=16, n_keypoints=50)
                desc_all = [d for d in desc_list if d is not None]
                if not desc_all:
                    messagebox.showwarning("Uyarı", f"'{roi_name}': Fisher için ORB descriptor bulunamadı.")
                    return
                desc_stack = np.vstack(desc_all)
                gmm = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
                gmm.fit(desc_stack)
                X = self._ml_extract_fisher_features(images, gmm=gmm, n_components=16, n_keypoints=50)
                model_type_str = "FISHER_SVM"; model_suffix = "FISHER_SVM"
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64), "fisher_gmm_components": 16}
            else:
                X = self._ml_extract_hog_features(images)
                model_type_str = "HOG_SVM"; model_suffix = ""
                params = {"kernel": "linear", "probability": True, "image_size": (64, 64),
                          "hog_pixels_per_cell": (8, 8), "hog_cells_per_block": (2, 2), "hog_orientations": 9}

            kernel = 'rbf' if use_sift else 'linear'
            svm_model = SVC(kernel=kernel, probability=True)

            # Her sınıftan en az 2 örnek varsa test bölümü yap, yoksa tümüyle eğit
            n_classes = len(np.unique(labels))
            min_per_class = min(np.bincount(
                np.array([list(np.unique(labels)).index(l) for l in labels])
            ))
            can_split = min_per_class >= 2 and len(X) >= n_classes * 5
            if can_split:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=42, stratify=labels)
                svm_model.fit(X_train, y_train)
                acc = float(accuracy_score(y_test, svm_model.predict(X_test)))
            else:
                # Az veri – tüm set ile eğit, doğruluk eğitim seti üzerinden
                svm_model.fit(X, labels)
                acc = float(accuracy_score(labels, svm_model.predict(X)))
            params["accuracy"] = acc

            model_filename = f"{safe_proj}_{safe_roi}_{model_suffix}.pkl" if model_suffix \
                else f"{safe_proj}_{safe_roi}.pkl"
            model_path = os.path.join(roi_folder, model_filename)
            joblib.dump(svm_model, model_path)

            if use_fisher and gmm is not None:
                gmm_path = os.path.join(roi_folder, f"{safe_proj}_{safe_roi}_FISHER_GMM.pkl")
                joblib.dump(gmm, gmm_path)
                params["gmm_path"] = gmm_path

            now = datetime.datetime.now().isoformat()
            if project_id is not None:
                self.cursor.execute('''
                    INSERT INTO roi_ml_models
                        (project_id, roi_name, model_type, model_path, params_json, created_date, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, roi_name) DO UPDATE SET
                        model_type=excluded.model_type,
                        model_path=excluded.model_path,
                        params_json=excluded.params_json,
                        updated_date=excluded.updated_date
                ''', (project_id, roi_name, model_type_str, model_path, json.dumps(params), now, now))
                self.conn.commit()
            # project_id=None → proje henüz kaydedilmedi, model dosya yoluna yazıldı.
            # Proje kaydedildikten sonra Ayarlar > ROI > Eğit ile DB'ye bağlanabilir.

            # Cache'i temizle – bir sonraki analizde taze yüklensin
            self.ml_models = {k: v for k, v in self.ml_models.items() if k[1] != roi_name}

            extra = "\n\n⚠ Projeyi kaydedin, ardından Ayarlar'dan tekrar eğitin (DB'ye bağlanacak)." \
                if project_id is None else ""
            messagebox.showinfo("Eğitim Tamamlandı",
                f"'{roi_name}' – [{model_type_str}]\n"
                f"Doğruluk: {acc:.2%}\n"
                f"Model: {model_path}{extra}")

        except Exception as exc:
            import traceback
            messagebox.showerror("Eğitim Hatası",
                f"'{roi_name}' eğitimi başarısız:\n{exc}\n\n{traceback.format_exc()}")

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
        """İzleme döngüsü – trigger tabanlı fotoğraf analizi."""
        if not self.monitoring_active:
            return

        plc_enabled = self.plc.config.get("enabled", False)

        # ── PLC trigger oku ───────────────────────────────────────────────────
        if plc_enabled:
            def _read_and_process():
                ok, trig_val, msg = self.plc.read_trigger()
                self.root.after(0, lambda: self._on_trigger_read(ok, trig_val, msg))
            threading.Thread(target=_read_and_process, daemon=True).start()
        # PLC yokken ekran zaten başlangıçta referans görselle doldurulmuştur.

        # PLC aktifse poll_interval'i kullan (daha hızlı tepki), değilse monitor_interval
        if plc_enabled:
            poll_ms = max(50, int(self.plc.config.get("poll_interval", 0.1) * 1000))
        else:
            poll_ms = self.monitor_interval
        self.root.after(poll_ms, self.update_monitoring)

    def _show_reference_preview(self):
        """Referans görselini ROI kutularıyla canvas'a göster (bekleme ekranı).
        Kamera okuma YOK – CPU yükü minimumdur."""
        if not self.monitoring_active:
            return
        if self.reference_image is None:
            return
        display = self.reference_image.copy()

        roi_list = self.current_project.get('roi_list', [])
        colors = ['#ef4444', '#3b82f6', '#f97316', '#a855f7',
                  '#06b6d4', '#ec4899', '#eab308', '#84cc16']
        for idx, rc in enumerate(roi_list):
            x, y, w, h = rc['coords']
            fh, fw = display.shape[:2]
            x  = max(0, min(x, fw - 1))
            y  = max(0, min(y, fh - 1))
            x2 = max(0, min(x + w, fw))
            y2 = max(0, min(y + h, fh))
            col_hex = colors[idx % len(colors)]
            r = int(col_hex[1:3], 16)
            g = int(col_hex[3:5], 16)
            b = int(col_hex[5:7], 16)
            cv2.rectangle(display, (x, y), (x2, y2), (r, g, b), 2)
            cv2.putText(display, rc.get('name', f'ROI_{idx+1}'),
                        (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (r, g, b), 2, cv2.LINE_AA)

        self.display_monitoring_frame(display)

    def _on_trigger_read(self, read_ok: bool, trig_val: bool, msg: str):
        """Trigger okuma sonucunu UI thread'inde işle."""
        if not self.monitoring_active:
            return

        if not read_ok:
            self.plc_status_label.config(text=f"PLC: HATA – {msg}", fg='#f59e0b')
            self._set_ctrl_state('WAITING')
            # Hata durumunda referans görseli yeniden göster
            self._show_reference_preview()
            return

        trig_str = "▶ 1 (Tetik)" if trig_val else "● 0 (Bekliyor)"
        self.plc_status_label.config(
            text=f"PLC: Bağlı  |  {trig_str}",
            fg='#22c55e' if trig_val else '#6b7280'
        )

        prev = self._trigger_prev
        self._trigger_prev = trig_val

        if trig_val and not prev:
            # ↑ Yükselen kenar: fotoğraf çek + analiz et
            self._do_capture_and_analyze()
        elif not trig_val and prev:
            # ↓ Düşen kenar: PLC sıfırladı → tekrar bekleme ekranı
            self._set_ctrl_state('WAITING')
            self._show_reference_preview()
        # trig_val=0 ve prev=0: zaten bekleme ekranında, tekrar çizmeye gerek yok
        # trig_val=1 ve prev=1: analiz yapıldı, sonuç ekranda, bekliyoruz

    def _do_capture_and_analyze(self):
        """Fotoğraf çek ve ROI analizini yap."""
        if not self.monitoring_active:
            return

        try:
            self._do_capture_and_analyze_inner()
        except Exception as exc:
            import traceback
            print(f"[HATA] Analiz sırasında beklenmedik hata:\n{traceback.format_exc()}")
            try:
                if self.monitoring_active:
                    self._set_ctrl_state('WAITING')
                    self.plc_status_label.config(text=f"Analiz hatası: {exc}", fg='#dc2626')
            except Exception:
                pass

    def _do_capture_and_analyze_inner(self):
        """Fotoğraf çek ve ROI analizini yap (iç mantık)."""
        # Bekleme durumunu göster
        self._set_ctrl_state('ANALYZING')
        self.root.update_idletasks()

        # Kameradan anlık frame al
        source = self.get_camera_source(self.current_project['camera_id'])
        if source == "HIKROBOT" and self.hik_camera:
            frame = self.hik_camera.get_frame()
        else:
            frame = self._get_latest_frame()

        if frame is None:
            if self.monitoring_active:
                self.plc_status_label.config(text="Kamera görüntüsü yok", fg='#dc2626')
                self._set_ctrl_state('WAITING')
            if self.plc.config.get("enabled"):
                threading.Thread(
                    target=self.plc.write_ok_signal, args=(False,), daemon=True
                ).start()
            return

        self.frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── ROI analizi ───────────────────────────────────────────────────────
        proj_algorithm = self.current_project.get('algorithm', 'SSIM')
        proj_threshold = float(self.current_project.get('algo_threshold', 0.75))
        roi_list  = self.current_project.get('roi_list', [])
        ref_rois  = getattr(self, 'reference_rois', [])

        if not roi_list:
            roi_list = [{'name': 'ROI_1', 'coords': self.current_project['roi']}]
            ref_rois = [self.reference_roi]

        # ROI listesi hâlâ boşsa analiz yapılamaz
        if not roi_list:
            if self.monitoring_active:
                self._set_ctrl_state('WAITING')
                self.plc_status_label.config(text="ROI tanımlı değil", fg='#f59e0b')
            return

        results = []   # (passed, score, (x,y,w,h), name, algorithm)
        display_frame = frame_rgb.copy()

        ML_ALGOS = (
            'ML_MODEL', 'ML_MODEL_SIFT', 'ML_MODEL_DAISY',
            'ML_MODEL_HAAR', 'ML_MODEL_CENSURE', 'ML_MODEL_MBLBP',
            'ML_MODEL_GLCM', 'ML_MODEL_LBP', 'ML_MODEL_GABOR', 'ML_MODEL_FISHER',
        )

        for idx, rc in enumerate(roi_list):
            try:
                x, y, w, h = rc['coords']
                name = rc.get('name', f'ROI_{idx+1}')
                # Per-ROI algoritma ve eşik – tanımlı değilse proje varsayılanı
                algorithm = rc.get('algorithm') or proj_algorithm
                threshold = float(rc.get('threshold') or proj_threshold)
                fh, fw = frame_rgb.shape[:2]
                x  = max(0, min(x, fw - 1))
                y  = max(0, min(y, fh - 1))
                x2 = max(0, min(x + w, fw))
                y2 = max(0, min(y + h, fh))

                # Sıfır boyutlu kesim kontrolü
                if x2 <= x or y2 <= y:
                    print(f"[UYARI] {name} ROI geçersiz boyut: ({x},{y})-({x2},{y2})")
                    results.append((False, 0.0, (x, y, max(1, x2-x), max(1, y2-y)), name, algorithm))
                    continue

                current_roi = frame_rgb[y:y2, x:x2]
                ref_roi = ref_rois[idx] if idx < len(ref_rois) else self.reference_roi

                if algorithm in ML_ALGOS:
                    model_key = (self.current_project.get('id'), name, algorithm)
                    model = self.ml_models.get(model_key)
                    if model is None:
                        expected_type = {
                            'ML_MODEL_SIFT': 'SIFT_SVM', 'ML_MODEL_DAISY': 'DAISY_SVM',
                            'ML_MODEL_HAAR': 'HAAR_SVM', 'ML_MODEL_CENSURE': 'CENSURE_SVM',
                            'ML_MODEL_MBLBP': 'MBLBP_SVM', 'ML_MODEL_GLCM': 'GLCM_SVM',
                            'ML_MODEL_LBP': 'LBP_SVM', 'ML_MODEL_GABOR': 'GABOR_SVM',
                            'ML_MODEL_FISHER': 'FISHER_SVM',
                        }.get(algorithm, 'HOG_SVM')
                        self.cursor.execute(
                            "SELECT model_path, model_type FROM roi_ml_models "
                            "WHERE project_id=? AND roi_name=?",
                            (self.current_project.get('id'), name),
                        )
                        db_row = self.cursor.fetchone()
                        if db_row and db_row[1] == expected_type and os.path.exists(db_row[0]):
                            try:
                                model = joblib.load(db_row[0])
                                self.ml_models[model_key] = model
                            except Exception as e:
                                print(f"[HATA] Model yüklenemedi ({name}): {e}")
                                model = None
                    if model is not None:
                        roi_gray = cv2.cvtColor(current_roi, cv2.COLOR_RGB2GRAY)
                        roi_res  = cv2.resize(roi_gray, (64, 64))
                        feat = self._extract_feat_for_algo(algorithm, roi_res, name)
                        pred = model.predict(feat)[0]
                        passed = str(pred).upper() == 'OK'
                        score  = 1.0 if passed else 0.0
                    else:
                        print(f"[UYARI] {name} için model bulunamadı, NOK sayılıyor")
                        passed, score = False, 0.0
                else:
                    score  = self._score_roi(current_roi, ref_roi, algorithm)
                    passed = score >= threshold

                results.append((passed, score, (x, y, x2 - x, y2 - y), name, algorithm))

            except Exception as e:
                print(f"[HATA] ROI '{rc.get('name', idx)}' analiz hatası: {e}")
                results.append((False, 0.0, (0, 0, 1, 1), rc.get('name', f'ROI_{idx+1}'), proj_algorithm))

        # ── Görüntü üzerine overlay çiz ───────────────────────────────────────
        overlay = display_frame.copy()
        for passed, score, (x, y, w, h), name, _algo in results:
            color = (0, 220, 80) if passed else (220, 40, 40)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.30, display_frame, 0.70, 0, display_frame)

        for passed, score, (x, y, w, h), name, _algo in results:
            color = (0, 220, 80) if passed else (220, 40, 40)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
            # Her ROI kendi algoritmasına göre etiketlenir
            lbl_txt = f"{name}: {'OK' if passed else 'NOK'}" if _algo in ML_ALGOS \
                else f"{name}: {score:.0%}"
            cv2.putText(display_frame, lbl_txt, (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if passed:
                cx, cy = x + w // 2, y + h // 2
                cv2.drawMarker(display_frame, (cx, cy), color,
                               cv2.MARKER_TILTED_CROSS, min(w, h) // 2, 4)
            else:
                m = 12
                cv2.line(display_frame, (x + m, y + m), (x + w - m, y + h - m), color, 5)
                cv2.line(display_frame, (x + w - m, y + m), (x + m, y + h - m), color, 5)

        # ── Sonuç ─────────────────────────────────────────────────────────────
        all_ok    = bool(results) and all(r[0] for r in results)
        nok_names = [r[3] for r in results if not r[0]]

        if all_ok:
            fh, fw = display_frame.shape[:2]
            cv2.rectangle(display_frame, (8, 8), (fw - 8, fh - 8), (0, 220, 80), 16)
            (tw, th), _ = cv2.getTextSize("OK", cv2.FONT_HERSHEY_DUPLEX, 5, 8)
            cv2.putText(display_frame, "OK",
                        ((fw - tw) // 2, (fh + th) // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 5, (0, 220, 80), 8, cv2.LINE_AA)

        if not self.monitoring_active:
            return  # izleme durduruldu, UI widget'larına dokunma

        self.display_monitoring_frame(display_frame)

        ok_count = len(results) - len(nok_names)
        self._set_ctrl_state(
            'OK' if all_ok else 'NOK',
            nok_names,
            detail=f"{ok_count}/{len(results)} bölge geçti"
        )

        # ── DB kaydı ──────────────────────────────────────────────────────────
        first_score = results[0][1] if results else 0.0
        self.log_monitoring(first_score, "OK" if all_ok else "FAIL")

        # ── PLC'ye sonuç yaz ──────────────────────────────────────────────────
        if self.plc.config.get("enabled"):
            def _plc_write(value):
                ok, msg = self.plc.write_ok_signal(value)
                bit_val = "1" if value else "0"
                color   = '#16a34a' if value else '#dc2626'
                label   = f"PLC: ● {bit_val}" if ok else f"PLC: HATA – {msg}"
                fg      = color if ok else '#f59e0b'
                # Widget yalnızca izleme hâlâ aktifse güncellenir
                if self.monitoring_active:
                    try:
                        self.root.after(0, lambda: self.plc_status_label.config(
                            text=label, fg=fg) if self.monitoring_active else None)
                    except Exception:
                        pass
            threading.Thread(target=_plc_write, args=(all_ok,), daemon=True).start()

    def _extract_feat_for_algo(self, algorithm: str, roi_resized, roi_name: str):
        """ML algoritmasına göre öznitelik çıkar (tek ROI, tek görüntü)."""
        if algorithm == 'ML_MODEL_SIFT':
            return self._ml_extract_sift_features([roi_resized], n_keypoints=50)
        if algorithm == 'ML_MODEL_DAISY':
            return self._ml_extract_daisy_features(
                [roi_resized], step=32, radius=16, rings=2, histograms=6, orientations=8)
        if algorithm == 'ML_MODEL_HAAR':
            return self._ml_extract_haar_features([roi_resized])
        if algorithm == 'ML_MODEL_CENSURE':
            return self._ml_extract_censure_features([roi_resized])
        if algorithm == 'ML_MODEL_MBLBP':
            return self._ml_extract_multiblock_lbp_features([roi_resized])
        if algorithm == 'ML_MODEL_GLCM':
            return self._ml_extract_glcm_features([roi_resized])
        if algorithm == 'ML_MODEL_LBP':
            return self._ml_extract_lbp_features([roi_resized], radius=3)
        if algorithm == 'ML_MODEL_GABOR':
            return self._ml_extract_gabor_features([roi_resized])
        if algorithm == 'ML_MODEL_FISHER':
            gmm_obj = None
            try:
                self.cursor.execute(
                    "SELECT params_json FROM roi_ml_models WHERE project_id=? AND roi_name=?",
                    (self.current_project.get('id'), roi_name),
                )
                prow = self.cursor.fetchone()
                if prow and prow[0]:
                    p = json.loads(prow[0])
                    gmm_path = p.get("gmm_path")
                    if gmm_path and os.path.exists(gmm_path):
                        gmm_obj = joblib.load(gmm_path)
            except Exception:
                pass
            if gmm_obj is None:
                return np.zeros((1, 1), dtype=np.float32)
            return self._ml_extract_fisher_features(
                [roi_resized], gmm=gmm_obj, n_components=16, n_keypoints=50)
        # Varsayılan: HOG
        from skimage.feature import hog
        return hog(roi_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False).reshape(1, -1)


    
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
        try:
            if not hasattr(self, 'monitor_canvas') or not self.monitor_canvas.winfo_exists():
                return
        except Exception:
            return
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
            if self.current_project is None:
                return
            now = datetime.datetime.now().isoformat()
            self.cursor.execute('''
                INSERT INTO monitoring_logs (project_id, timestamp, status, similarity_score)
                VALUES (?, ?, ?, ?)
            ''', (self.current_project['id'], now, status, float(similarity)))
            self.conn.commit()
        except Exception as e:
            print(f"[LOG HATA] {e}")
    
    def stop_monitoring(self):
        """İzlemeyi durdur"""
        self.monitoring_active = False  # önce kapat – callback'lerin widget'a ulaşmasını önle
        # PLC OK bitini sıfırla (izleme bitti, parça gitti) — thread'de çalıştır
        if self.plc.config.get("enabled"):
            threading.Thread(target=self.plc.write_ok_signal, args=(False,), daemon=True).start()
        # Arka plan kamera okuma thread'ini durdur
        self._stop_cam_reader()
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.hik_camera:
            try:
                self.hik_camera.stop()
            except Exception:
                pass
            self.hik_camera = None   # bir sonraki başlatmada open() yeniden çağrılsın
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

    def _on_closing():
        """X butonuna basınca temiz kapat."""
        app.monitoring_active = False
        app._stop_cam_reader()
        if app.hik_camera:
            try:
                app.hik_camera.stop()
            except Exception:
                pass
        if app.camera:
            try:
                app.camera.release()
            except Exception:
                pass
        if app.plc.connected:
            try:
                app.plc.disconnect()
            except Exception:
                pass
        try:
            app.conn.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_closing)
    root.mainloop()