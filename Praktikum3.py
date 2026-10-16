import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# FUNGSI BANTU MANUAL (tanpa cv2.add, cv2.absdiff, dll)
# ============================================================

def manual_add(img1, img2):
    """Penjumlahan dua citra secara manual"""
    result = np.zeros_like(img1)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for k in range(img1.shape[2]):
                val = int(img1[i, j, k]) + int(img2[i, j, k])
                result[i, j, k] = min(255, max(0, val))
    return result

def manual_subtract(img1, img2):
    """Pengurangan dua citra secara manual"""
    result = np.zeros_like(img1)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for k in range(img1.shape[2]):
                val = int(img1[i, j, k]) - int(img2[i, j, k])
                result[i, j, k] = min(255, max(0, val))
    return result

def manual_absdiff(img1, img2):
    """Selisih absolut dua citra secara manual"""
    result = np.zeros_like(img1)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for k in range(img1.shape[2]):
                val = abs(int(img1[i, j, k]) - int(img2[i, j, k]))
                result[i, j, k] = min(255, max(0, val))
    return result

def manual_convert_scale_abs(img, alpha=1.0, beta=0):
    """Konversi skala dan absolut secara manual"""
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                val = alpha * float(img[i, j, k]) + beta
                result[i, j, k] = min(255, max(0, int(abs(val))))
    return result

def manual_normalize(img, low=0, high=255):
    """Normalisasi citra secara manual"""
    min_val = 255
    max_val = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i, j, k] < min_val:
                    min_val = img[i, j, k]
                if img[i, j, k] > max_val:
                    max_val = img[i, j, k]
    
    result = np.zeros_like(img)
    range_val = max_val - min_val
    if range_val == 0:
        return img.copy()
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                val = (float(img[i, j, k]) - min_val) / range_val * (high - low) + low
                result[i, j, k] = min(255, max(0, int(val)))
    return result

def rgb_to_grayscale(img):
    """Konversi RGB ke Grayscale secara manual"""
    h, w = img.shape[:2]
    gray = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # Rumus luminance: 0.299R + 0.587G + 0.114B
            gray[i, j] = int(0.299 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 2])
    return gray

def grayscale_to_rgb(gray):
    """Konversi Grayscale ke RGB secara manual"""
    h, w = gray.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb[i, j, 0] = gray[i, j]
            rgb[i, j, 1] = gray[i, j]
            rgb[i, j, 2] = gray[i, j]
    return rgb

def apply_kernel_2d(gray_img, kernel):
    """Aplikasi kernel konvolusi 2D pada citra grayscale secara manual"""
    h, w = gray_img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Padding
    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            padded[i + pad_h, j + pad_w] = gray_img[i, j]
    
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            sum_val = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    sum_val += padded[i + ki, j + kj] * kernel[ki, kj]
            result[i, j] = sum_val
    
    return result

def apply_kernel_3d(rgb_img, kernel):
    """Aplikasi kernel konvolusi pada citra RGB secara manual"""
    h, w = rgb_img.shape[:2]
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    result = np.zeros_like(rgb_img, dtype=np.float64)
    
    for channel in range(3):
        # Padding untuk channel ini
        padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                padded[i + pad_h, j + pad_w] = rgb_img[i, j, channel]
        
        for i in range(h):
            for j in range(w):
                sum_val = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        sum_val += padded[i + ki, j + kj] * kernel[ki, kj]
                result[i, j, channel] = sum_val
    
    # Clip ke 0-255
    for i in range(h):
        for j in range(w):
            for c in range(3):
                result[i, j, c] = min(255, max(0, int(result[i, j, c])))
    
    return result.astype(np.uint8)


# ============================================================
# 1. SMOOTHING / BLURRING
# ============================================================

def mean_filter(img, kernel_size=3):
    """Mean Filter secara manual"""
    h, w = img.shape[:2]
    pad = kernel_size // 2
    
    if len(img.shape) == 3:
        result = np.zeros_like(img, dtype=np.float64)
        for channel in range(3):
            padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float64)
            for i in range(h):
                for j in range(w):
                    padded[i + pad, j + pad] = img[i, j, channel]
            
            for i in range(h):
                for j in range(w):
                    sum_val = 0.0
                    count = 0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            sum_val += padded[i + ki, j + kj]
                            count += 1
                    result[i, j, channel] = sum_val / count
        
        for i in range(h):
            for j in range(w):
                for c in range(3):
                    result[i, j, c] = min(255, max(0, int(result[i, j, c])))
        return result.astype(np.uint8)
    else:
        result = np.zeros((h, w), dtype=np.float64)
        padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                padded[i + pad, j + pad] = img[i, j]
        
        for i in range(h):
            for j in range(w):
                sum_val = 0.0
                count = 0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        sum_val += padded[i + ki, j + kj]
                        count += 1
                result[i, j] = sum_val / count
        
        for i in range(h):
            for j in range(w):
                result[i, j] = min(255, max(0, int(result[i, j])))
        return result.astype(np.uint8)


def gaussian_filter(img, kernel_size=3, sigma=1.0):
    """Gaussian Filter secara manual"""
    # Buat kernel Gaussian secara manual
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    total = 0.0
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            # Rumus Gaussian: G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
            val = (1.0 / (2.0 * 3.14159265358979 * sigma * sigma)) * \
                  np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            kernel[i, j] = val
            total += val
    
    # Normalisasi kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= total
    
    return apply_kernel_3d(img, kernel)


# ============================================================
# 2. SHARPENING
# ============================================================

def laplacian_sharpening(img, alpha=1.0):
    """Laplacian-based Sharpening secara manual
    Sharpened = Original + alpha * Laplacian
    """
    # Kernel Laplacian
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float64)
    
    h, w = img.shape[:2]
    result = np.zeros_like(img, dtype=np.float64)
    
    for channel in range(3):
        laplacian = apply_kernel_2d_channel(img, channel, laplacian_kernel)
        for i in range(h):
            for j in range(w):
                val = float(img[i, j, channel]) + alpha * laplacian[i, j]
                result[i, j, channel] = min(255, max(0, int(val)))
    
    return result.astype(np.uint8)


def apply_kernel_2d_channel(img, channel, kernel):
    """Aplikasi kernel 2D pada satu channel"""
    h, w = img.shape[:2]
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            padded[i + pad_h, j + pad_w] = img[i, j, channel]
    
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            sum_val = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    sum_val += padded[i + ki, j + kj] * kernel[ki, kj]
            result[i, j] = sum_val
    
    return result


def high_pass_filter(img):
    """High-pass Filtering secara manual
    HPF = Original - Low-pass (Mean filtered)
    """
    low_pass = mean_filter(img, kernel_size=5)
    h, w = img.shape[:2]
    result = np.zeros_like(img, dtype=np.float64)
    
    for i in range(h):
        for j in range(w):
            for c in range(3):
                val = float(img[i, j, c]) - float(low_pass[i, j, c]) + 128
                result[i, j, c] = min(255, max(0, int(val)))
    
    return result.astype(np.uint8)


# ============================================================
# 3. EDGE DETECTION
# ============================================================

def sobel_edge_detection(img):
    """Sobel Edge Detection secara manual"""
    # Kernel Sobel X dan Y
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    gray = rgb_to_grayscale(img)
    h, w = gray.shape
    
    gx = apply_kernel_2d(gray, sobel_x)
    gy = apply_kernel_2d(gray, sobel_y)
    
    # Magnitude = sqrt(Gx^2 + Gy^2)
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            magnitude = np.sqrt(gx[i, j] ** 2 + gy[i, j] ** 2)
            result[i, j] = min(255, magnitude)
    
    return result.astype(np.uint8)


def prewitt_edge_detection(img):
    """Prewitt Edge Detection secara manual"""
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)
    
    gray = rgb_to_grayscale(img)
    h, w = gray.shape
    
    gx = apply_kernel_2d(gray, prewitt_x)
    gy = apply_kernel_2d(gray, prewitt_y)
    
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            magnitude = np.sqrt(gx[i, j] ** 2 + gy[i, j] ** 2)
            result[i, j] = min(255, magnitude)
    
    return result.astype(np.uint8)


def roberts_edge_detection(img):
    """Roberts Edge Detection secara manual"""
    roberts_x = np.array([
        [1,  0],
        [0, -1]
    ], dtype=np.float64)
    
    roberts_y = np.array([
        [ 0, 1],
        [-1, 0]
    ], dtype=np.float64)
    
    gray = rgb_to_grayscale(img)
    h, w = gray.shape
    
    # Untuk kernel 2x2, padding = 0 (tidak ada center pixel)
    # Kita gunakan pendekatan manual
    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)
    
    for i in range(h - 1):
        for j in range(w - 1):
            gx[i, j] = abs(float(gray[i, j]) - float(gray[i + 1, j + 1]))
            gy[i, j] = abs(float(gray[i, j + 1]) - float(gray[i + 1, j]))
    
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            magnitude = np.sqrt(gx[i, j] ** 2 + gy[i, j] ** 2)
            result[i, j] = min(255, magnitude)
    
    return result.astype(np.uint8)


# ============================================================
# 4. EMBOSSING
# ============================================================

def embossing(img):
    """Embossing secara manual"""
    # Kernel Emboss
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ], dtype=np.float64)
    
    h, w = img.shape[:2]
    result = np.zeros_like(img, dtype=np.float64)
    
    for channel in range(3):
        convolved = apply_kernel_2d_channel(img, channel, emboss_kernel)
        for i in range(h):
            for j in range(w):
                val = convolved[i, j] + 128  # Offset agar tidak negatif
                result[i, j, channel] = min(255, max(0, int(val)))
    
    return result.astype(np.uint8)


# ============================================================
# APLIKASI GUI
# ============================================================

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengolahan Citra Digital")
        self.root.geometry("1200x800")
        
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame utama
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame kontrol (kiri)
        control_frame = ttk.LabelFrame(main_frame, text="Kontrol", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Tombol Load Image
        ttk.Button(control_frame, text="Muat Citra", 
                   command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Label status
        self.status_label = ttk.Label(control_frame, text="Belum ada citra")
        self.status_label.pack(pady=10)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Smoothing
        smooth_frame = ttk.LabelFrame(control_frame, text="Smoothing/Blurring", padding="5")
        smooth_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(smooth_frame, text="Mean Filter (3x3)", 
                   command=lambda: self.apply_filter("mean_3")).pack(fill=tk.X, pady=2)
        ttk.Button(smooth_frame, text="Mean Filter (5x5)", 
                   command=lambda: self.apply_filter("mean_5")).pack(fill=tk.X, pady=2)
        ttk.Button(smooth_frame, text="Gaussian Filter (3x3)", 
                   command=lambda: self.apply_filter("gaussian_3")).pack(fill=tk.X, pady=2)
        ttk.Button(smooth_frame, text="Gaussian Filter (5x5)", 
                   command=lambda: self.apply_filter("gaussian_5")).pack(fill=tk.X, pady=2)
        
        # Sharpening
        sharp_frame = ttk.LabelFrame(control_frame, text="Sharpening", padding="5")
        sharp_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(sharp_frame, text="Laplacian Sharpening", 
                   command=lambda: self.apply_filter("laplacian")).pack(fill=tk.X, pady=2)
        ttk.Button(sharp_frame, text="High-pass Filter", 
                   command=lambda: self.apply_filter("highpass")).pack(fill=tk.X, pady=2)
        
        # Edge Detection
        edge_frame = ttk.LabelFrame(control_frame, text="Edge Detection", padding="5")
        edge_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(edge_frame, text="Sobel", 
                   command=lambda: self.apply_filter("sobel")).pack(fill=tk.X, pady=2)
        ttk.Button(edge_frame, text="Prewitt", 
                   command=lambda: self.apply_filter("prewitt")).pack(fill=tk.X, pady=2)
        ttk.Button(edge_frame, text="Roberts", 
                   command=lambda: self.apply_filter("roberts")).pack(fill=tk.X, pady=2)
        
        # Embossing
        ttk.Button(control_frame, text="Embossing", 
                   command=lambda: self.apply_filter("emboss")).pack(fill=tk.X, pady=5)
        
        # Frame gambar (kanan)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Figure untuk matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 6))
        self.ax1.set_title("Citra Asli")
        self.ax2.set_title("Hasil Pemrosesan")
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            # Load menggunakan PIL, konversi ke numpy array
            pil_img = Image.open(file_path)
            self.original_image = np.array(pil_img)
            
            # Jika grayscale, konversi ke RGB
            if len(self.original_image.shape) == 2:
                self.original_image = grayscale_to_rgb(
                    np.stack([self.original_image]*3, axis=-1)
                )
            elif self.original_image.shape[2] == 4:  # RGBA
                self.original_image = self.original_image[:, :, :3]
            
            self.status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            self.display_images()
    
    def apply_filter(self, filter_type):
        if self.original_image is None:
            messagebox.showwarning("Peringatan", "Muat citra terlebih dahulu!")
            return
        
        img = self.original_image.copy()
        
        if filter_type == "mean_3":
            self.processed_image = mean_filter(img, kernel_size=3)
        elif filter_type == "mean_5":
            self.processed_image = mean_filter(img, kernel_size=5)
        elif filter_type == "gaussian_3":
            self.processed_image = gaussian_filter(img, kernel_size=3, sigma=1.0)
        elif filter_type == "gaussian_5":
            self.processed_image = gaussian_filter(img, kernel_size=5, sigma=1.5)
        elif filter_type == "laplacian":
            self.processed_image = laplacian_sharpening(img, alpha=1.0)
        elif filter_type == "highpass":
            self.processed_image = high_pass_filter(img)
        elif filter_type == "sobel":
            result = sobel_edge_detection(img)
            self.processed_image = grayscale_to_rgb(result)
        elif filter_type == "prewitt":
            result = prewitt_edge_detection(img)
            self.processed_image = grayscale_to_rgb(result)
        elif filter_type == "roberts":
            result = roberts_edge_detection(img)
            self.processed_image = grayscale_to_rgb(result)
        elif filter_type == "emboss":
            self.processed_image = embossing(img)
        
        self.display_images()
    
    def display_images(self):
        self.ax1.clear()
        self.ax2.clear()
        
        if self.original_image is not None:
            self.ax1.imshow(self.original_image)
            self.ax1.set_title("Citra Asli")
            self.ax1.axis('off')
        
        if self.processed_image is not None:
            self.ax2.imshow(self.processed_image)
            self.ax2.set_title("Hasil Pemrosesan")
            self.ax2.axis('off')
        else:
            self.ax2.set_title("Hasil Pemrosesan")
            self.ax2.axis('off')
        
        self.canvas.draw()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
