import cv2
import numpy as np
import io
import os
from typing import List, Dict
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available, using fallback metrics")

def compress_image_bytes(image_bytes: bytes, max_bytes: int) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == 'RGBA': img = img.convert('RGB')
    quality, step, min_side = 95, 5, 32
    while quality > 10:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        if len(buf.getvalue()) <= max_bytes: return buf.getvalue()
        quality -= step
    while True:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        if len(buf.getvalue()) <= max_bytes: return buf.getvalue()
        w, h = img.size
        if w <= min_side or h <= min_side: break
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
        img = img.resize((max(int(w*0.9),min_side), max(int(h*0.9),min_side)), resample)
    return None

def validate_file_type(filename: str, data: bytes = None) -> bool:
    if not filename: return False
    ext = os.path.splitext(filename.lower())[1]
    if ext not in {'.png','.jpg','.jpeg','.bmp','.tiff','.gif','.mp4','.avi','.mov','.mkv'}: return False
    if data:
        magic = {b'\x89PNG\r\n\x1a\n':'.png', b'\xff\xd8\xff':'.jpg', b'BM':'.bmp', b'GIF8':'.gif', b'\x00\x00\x00\x18ftyp':'.mp4', b'RIFF':'.avi'}
        for m, e in magic.items():
            if data.startswith(m): return ext == e
    return True

def cleanup_temp_files(file_paths: List[str]):
    for path in file_paths:
        try:
            if path and os.path.exists(path): os.remove(path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {path}: {e}")

def _fallback_psnr(i1, i2):
    mse = np.mean((i1.astype(np.float64) - i2.astype(np.float64))**2)
    return float('inf') if mse==0 else 20*np.log10(255.0/np.sqrt(mse))

def _fallback_ssim(i1, i2):
    i1,i2 = i1.astype(np.float64)/255.0, i2.astype(np.float64)/255.0
    m1,m2=np.mean(i1),np.mean(i2)
    s12=np.mean((i1-m1)*(i2-m2))
    C1,C2=0.01**2,0.03**2
    num=(2*m1*m2+C1)*(2*s12+C2)
    den=(m1**2+m2**2+C1)*(np.var(i1)+np.var(i2)+C2)
    return float(num/den)

def calculate_image_metrics(orig_path: str, stego_path: str) -> Dict[str, float]:
    try:
        orig, stego = cv2.imread(orig_path), cv2.imread(stego_path)
        if orig is None or stego is None: return {"error": "Could not load images"}
        if orig.shape != stego.shape: stego = cv2.resize(stego, (orig.shape[1], orig.shape[0]))
        orig_g, stego_g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
        if SKIMAGE_AVAILABLE:
            psnr_val = psnr(orig_g, stego_g, data_range=255)
            ssim_val = ssim(orig_g, stego_g, data_range=255)
        else:
            psnr_val, ssim_val = _fallback_psnr(orig_g, stego_g), _fallback_ssim(orig_g, stego_g)
        mse = np.mean((orig.astype(np.float64) - stego.astype(np.float64))**2)
        return {"psnr": float(psnr_val), "ssim": float(ssim_val), "mse": float(mse)}
    except Exception as e:
        return {"error": str(e)}