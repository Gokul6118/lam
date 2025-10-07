import os
import json
import base64
import struct
import tempfile
import itertools
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from Crypto.Hash import SHA256
from Crypto.Cipher import AES, ChaCha20_Poly1305, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: pywt not available — DWT stego will be skipped. To enable it, run: pip install PyWavelets")

class StegoHandler:
    def __init__(self):
        self.MAGIC_BYTES = b'STEGO'
        self.VERSION = 1

        self.stego_methods = {
            'LSB': (self._lsb_embed, self._lsb_extract),
            'PVD': (self._pvd_embed, self._pvd_extract),
            'DCT': (self._dct_embed, self._dct_extract),
            'F5': (self._f5_embed, self._f5_extract),
        }
        if PYWT_AVAILABLE:
            self.stego_methods['DWT'] = (self._dwt_embed, self._dwt_extract)

        self.crypto_methods = {
            'AES-GCM': (self._encrypt_aes_gcm, self._decrypt_aes_gcm),
            'CHACHA20': (self._encrypt_chacha20, self._decrypt_chacha20),
            'RSA-HYBRID': (self._encrypt_rsa_hybrid, self._decrypt_rsa_hybrid),
        }

    def embed(self, cover_path: str, ciphertext: bytes, stego_method: str,
              stego_params: Dict[str, Any], crypto_method: str,
              crypto_meta: Dict[str, Any], timestamp: Optional[float] = None) -> str:
        method_upper = stego_method.upper()
        if method_upper not in self.stego_methods:
            raise ValueError(f"Stego method '{stego_method}' not supported. Available: {list(self.stego_methods.keys())}")
        
        header = self._create_header(ciphertext, stego_method=method_upper, crypto_method=crypto_method, crypto_meta=crypto_meta)
        full_payload = header + ciphertext

        if self._is_video_file(cover_path):
            img = self._get_frame_from_video(cover_path, timestamp)
        else:
            img = cv2.imread(cover_path)
            if img is None: raise IOError(f"Could not read cover image: {cover_path}")
        
        embed_fn = self.stego_methods[method_upper][0]
        stego_img = embed_fn(img, full_payload, stego_params)
        
        out_ext = ".png" # Always use PNG for lossless output to protect data
        
        fd, out_path = tempfile.mkstemp(suffix=f'_stego{out_ext}')
        os.close(fd)
        stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, stego_img)
        return out_path

    def extract(self, stego_path: str, params: Dict[str, Any], timestamp: Optional[float] = None) -> Tuple[bytes, Dict[str, Any]]:
        if self._is_video_file(stego_path):
            img = self._get_frame_from_video(stego_path, timestamp)
        else:
            img = cv2.imread(stego_path, cv2.IMREAD_UNCHANGED)
            if img is None: raise IOError(f"Could not read stego image: {stego_path}")
        
        decrypted_data, header_meta = self._extract_from_image_frame(img, params)
        return decrypted_data, header_meta

    def _create_header(self, ciphertext: bytes, stego_method: str, crypto_method: str, crypto_meta: Optional[Dict[str, Any]] = None) -> bytes:
        payload_hash = SHA256.new(ciphertext).digest()
        header_data = {'magic': self.MAGIC_BYTES.decode('latin-1'), 'version': self.VERSION, 'stego_method': stego_method, 'crypto_method': crypto_method, 'payload_length': len(ciphertext), 'payload_hash': base64.b64encode(payload_hash).decode('ascii'), 'crypto_meta': crypto_meta or {}}
        header_bytes = json.dumps(header_data, separators=(',', ':')).encode('utf-8')
        return struct.pack('>I', len(header_bytes)) + header_bytes

    def _parse_header(self, data: bytes) -> Tuple[Dict[str, Any], int]:
        if len(data) < 4: raise ValueError("Data too short.")
        header_len = struct.unpack('>I', data[:4])[0]
        if len(data) < 4 + header_len: raise ValueError("Data shorter than header.")
        header_obj = json.loads(data[4:4 + header_len].decode('utf-8'))
        if header_obj.get('magic') != self.MAGIC_BYTES.decode('latin-1'): raise ValueError("Magic mismatch.")
        return header_obj, 4 + header_len

    def _extract_from_image_frame(self, img: np.ndarray, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        for name, (_, extract_fn) in self.stego_methods.items():
            try:
                raw_data = extract_fn(img, header_only=True)
                if not raw_data: continue
                
                header, _ = self._parse_header(raw_data)
                stego_method = header['stego_method']
                if stego_method != name: continue
                
                full_data = extract_fn(img, header_only=False)
                header_check, payload_start_offset = self._parse_header(full_data)
                if header_check != header: continue
                
                ciphertext = full_data[payload_start_offset : payload_start_offset + header['payload_length']]
                expected_hash = base64.b64decode(header['payload_hash'])
                actual_hash = SHA256.new(ciphertext).digest()
                if actual_hash != expected_hash: continue
                
                crypto_method = header.get('crypto_method')
                crypto_meta = header.get('crypto_meta', {})
                full_params = {**crypto_meta, **params}
                
                plaintext = self._decrypt_payload(ciphertext, crypto_method, full_params)
                return plaintext, header
            except Exception:
                continue
        raise ValueError(f"Could not find or decrypt a valid payload.")

    def _encrypt_payload(self, p, cm, pa): return self.crypto_methods[cm.upper()][0](p, pa)
    def _decrypt_payload(self, c, cm, pa): return self.crypto_methods[cm.upper()][1](c, pa)
    
    # --- Crypto Implementations ---
    def _encrypt_aes_gcm(self, plaintext: bytes, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        key = params.get('key', get_random_bytes(32))
        cipher = AES.new(key, AES.MODE_GCM)
        ct, tag = cipher.encrypt_and_digest(plaintext)
        meta = {'key_b64': base64.b64encode(key).decode('ascii'), 'nonce_b64': base64.b64encode(cipher.nonce).decode('ascii'), 'tag_b64': base64.b64encode(tag).decode('ascii')}
        return ct, meta

    def _decrypt_aes_gcm(self, ciphertext: bytes, params: Dict[str, Any]) -> bytes:
        key = base64.b64decode(params['key_b64'])
        nonce = base64.b64decode(params['nonce_b64'])
        tag = base64.b64decode(params['tag_b64'])
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag)
    
    def _encrypt_chacha20(self, plaintext: bytes, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        key = params.get('key', get_random_bytes(32))
        cipher = ChaCha20_Poly1305.new(key=key)
        ct, tag = cipher.encrypt_and_digest(plaintext)
        meta = {'key_b64': base64.b64encode(key).decode('ascii'), 'nonce_b64': base64.b64encode(cipher.nonce).decode('ascii'), 'tag_b64': base64.b64encode(tag).decode('ascii')}
        return ct, meta

    def _decrypt_chacha20(self, ciphertext: bytes, params: Dict[str, Any]) -> bytes:
        key = base64.b64decode(params['key_b64'])
        nonce = base64.b64decode(params['nonce_b64'])
        tag = base64.b64decode(params['tag_b64'])
        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag)
    
    def _encrypt_rsa_hybrid(self, plaintext: bytes, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        rsa_pub_pem = params.get('rsa_pubkey_pem')
        if not rsa_pub_pem: raise ValueError("RSA public key is required for encryption.")
        rsa_key = RSA.import_key(rsa_pub_pem)
        cipher_rsa = PKCS1_OAEP.new(rsa_key)
        session_key = get_random_bytes(32)
        cipher_aes = AES.new(session_key, AES.MODE_GCM)
        ct, tag = cipher_aes.encrypt_and_digest(plaintext)
        meta = { 'enc_session_key_b64': base64.b64encode(cipher_rsa.encrypt(session_key)).decode('ascii'), 'nonce_b64': base64.b64encode(cipher_aes.nonce).decode('ascii'), 'tag_b64': base64.b64encode(tag).decode('ascii') }
        return ct, meta

    def _decrypt_rsa_hybrid(self, ciphertext: bytes, params: Dict[str, Any]) -> bytes:
        rsa_priv_pem = params.get('rsa_privkey_pem')
        if not rsa_priv_pem: raise ValueError("RSA private key is required for decryption.")
        rsa_key = RSA.import_key(rsa_priv_pem)
        cipher_rsa = PKCS1_OAEP.new(rsa_key)
        session_key = cipher_rsa.decrypt(base64.b64decode(params['enc_session_key_b64']))
        nonce = base64.b64decode(params['nonce_b64'])
        tag = base64.b64decode(params['tag_b64'])
        cipher_aes = AES.new(session_key, AES.MODE_GCM, nonce=nonce)
        return cipher_aes.decrypt_and_verify(ciphertext, tag)

    # --- STEGO METHODS ---
    def _lsb_embed(self, img, payload, params):
        bits = ''.join(f"{byte:08b}" for byte in payload) + '1111111111111110'
        flat_pixels = img.flatten()
        if len(bits) > flat_pixels.size: raise ValueError("Payload too large for LSB.")
        for i, bit in enumerate(bits): flat_pixels[i] = (flat_pixels[i] & 0xFE) | int(bit)
        return flat_pixels.reshape(img.shape)

    def _lsb_extract(self, img, header_only=False):
        flat_pixels = img.flatten()
        extracted_bits = ""
        max_bits = 8192 if header_only else flat_pixels.size
        for i in range(max_bits):
            extracted_bits += str(flat_pixels[i] & 1)
            if not header_only and extracted_bits.endswith('1111111111111110'): break
        if not header_only:
            eom_index = extracted_bits.rfind('1111111111111110')
            if eom_index != -1: extracted_bits = extracted_bits[:eom_index]
        extracted_bits = extracted_bits.ljust((len(extracted_bits) + 7) & ~7, '0')
        return bytes(int(extracted_bits[i:i+8], 2) for i in range(0, len(extracted_bits), 8))

    def _pvd_embed(self, img, payload, params):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bits = ''.join(f"{byte:08b}" for byte in payload) + '1111111111111110'
        flat_img = gray_img.flatten().astype(int)
        bit_idx = 0
        for i in range(0, len(flat_img) - 1, 2):
            if bit_idx >= len(bits): break
            p1, p2 = flat_img[i], flat_img[i+1]
            diff = p2 - p1
            n = 2 if -7 <= diff <= 7 else 3 if -15 <= diff <= 15 else 4 if -31 <= diff <= 31 else 5
            if bit_idx + n > len(bits): continue
            b = int(bits[bit_idx:bit_idx+n], 2)
            l = 2**n; new_diff = (diff - (diff % l)) + b; m = new_diff - diff
            p1_new, p2_new = p1 - (m // 2), p2 + (m - m // 2)
            if 0 <= p1_new <= 255 and 0 <= p2_new <= 255:
                flat_img[i], flat_img[i+1] = p1_new, p2_new; bit_idx += n
        final_gray = flat_img.reshape(gray_img.shape)
        return cv2.cvtColor(final_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _pvd_extract(self, img, header_only=False):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat_img = gray_img.flatten().astype(int)
        extracted_bits = ""
        max_len = 8192 if header_only else float('inf')
        for i in range(0, len(flat_img) - 1, 2):
            if len(extracted_bits) > max_len: break
            p1, p2 = flat_img[i], flat_img[i+1]
            diff = p2 - p1
            n = 2 if -7 <= diff <= 7 else 3 if -15 <= diff <= 15 else 4 if -31 <= diff <= 31 else 5
            b = diff % (2**n)
            extracted_bits += format(b, f'0{n}b')
            if not header_only and extracted_bits.endswith('1111111111111110'): break
        if not header_only:
            eom_index = extracted_bits.rfind('1111111111111110')
            if eom_index != -1: extracted_bits = extracted_bits[:eom_index]
        extracted_bits = extracted_bits.ljust((len(extracted_bits) + 7) & ~7, '0')
        return bytes(int(extracted_bits[i:i+8], 2) for i in range(0, len(extracted_bits), 8))

    def _dct_embed(self, img, payload, params):
        bits = ''.join(f"{byte:08b}" for byte in payload) + '1111111111111110'
        h, w, c = img.shape; bit_idx = 0; stego_img = img.copy()
        for ch in range(c):
            if bit_idx >= len(bits): break
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    if bit_idx >= len(bits): break
                    block = stego_img[i:i+8, j:j+8, ch].astype(np.float32) - 128
                    if block.shape != (8,8): continue
                    dct_block = cv2.dct(block)
                    dct_block[4, 4] = (int(dct_block[4, 4]) & 0xFFFE) | int(bits[bit_idx])
                    stego_img[i:i+8, j:j+8, ch] = cv2.idct(dct_block) + 128
                    bit_idx += 1
        return stego_img

    def _dct_extract(self, img, header_only=False):
        h, w, c = img.shape; extracted_bits = ""; max_len = 8192 if header_only else float('inf')
        for ch in range(c):
            if len(extracted_bits) > max_len: break
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    if len(extracted_bits) > max_len: break
                    block = img[i:i+8, j:j+8, ch].astype(np.float32) - 128
                    if block.shape != (8,8): continue
                    dct_block = cv2.dct(block)
                    extracted_bits += str(int(dct_block[4, 4]) & 1)
        if not header_only:
            eom_index = extracted_bits.rfind('1111111111111110')
            if eom_index != -1: extracted_bits = extracted_bits[:eom_index]
        extracted_bits = extracted_bits.ljust((len(extracted_bits) + 7) & ~7, '0')
        return bytes(int(extracted_bits[i:i+8], 2) for i in range(0, len(extracted_bits), 8))

    def _dwt_embed(self, img, payload, params):
        bits = ''.join(f"{byte:08b}" for byte in payload) + '1111111111111110'
        bit_idx = 0; stego_img = img.copy().astype(float)
        for ch in range(3):
            if bit_idx >= len(bits): break
            original_shape = stego_img[:, :, ch].shape
            coeffs = pywt.dwt2(stego_img[:, :, ch], 'haar')
            cA, (cH, cV, cD) = coeffs
            ch_flat = cH.flatten()
            for i in range(len(ch_flat)):
                if bit_idx < len(bits): ch_flat[i] = (int(ch_flat[i]) & 0xFFFE) | int(bits[bit_idx]); bit_idx += 1
                else: break
            cH = ch_flat.reshape(cH.shape)
            reconstructed = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
            stego_img[:, :, ch] = reconstructed[:original_shape[0], :original_shape[1]]
        return stego_img

    def _dwt_extract(self, img, header_only=False):
        extracted_bits = ""; max_len = 8192 if header_only else float('inf')
        img_float = img.astype(float)
        for ch in range(3):
            if len(extracted_bits) > max_len: break
            _, (cH, _, _) = pywt.dwt2(img_float[:,:,ch], 'haar')
            for coeff in cH.flatten():
                if len(extracted_bits) > max_len: break
                extracted_bits += str(int(coeff) & 1)
        if not header_only:
            eom_index = extracted_bits.rfind('1111111111111110')
            if eom_index != -1: extracted_bits = extracted_bits[:eom_index]
        extracted_bits = extracted_bits.ljust((len(extracted_bits) + 7) & ~7, '0')
        return bytes(int(extracted_bits[i:i+8], 2) for i in range(0, len(extracted_bits), 8))

    def _f5_embed(self, img, payload, params):
        bits = ''.join(f"{byte:08b}" for byte in payload) + '1111111111111110'
        h, w, c = img.shape; bit_idx = 0; stego_img = img.copy()
        for ch in range(c):
            if bit_idx >= len(bits): break
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    if bit_idx >= len(bits): break
                    block = stego_img[i:i+8, j:j+8, ch].astype(np.float32) - 128
                    if block.shape != (8,8): continue
                    dct_block = cv2.dct(block)
                    for x, y in itertools.product(range(8), range(8)):
                        if x == 0 and y == 0 or bit_idx >= len(bits): continue
                        coeff = int(dct_block[x, y])
                        if coeff != 0:
                            if (coeff & 1) != int(bits[bit_idx]):
                                dct_block[x, y] += -1 if coeff > 0 else 1
                            bit_idx += 1
                    stego_img[i:i+8, j:j+8, ch] = cv2.idct(dct_block) + 128
        return stego_img

    def _f5_extract(self, img, header_only=False):
        h, w, c = img.shape; extracted_bits = ""; max_len = 8192 if header_only else float('inf')
        for ch in range(c):
            if len(extracted_bits) > max_len: break
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    if len(extracted_bits) > max_len: break
                    block = img[i:i+8, j:j+8, ch].astype(np.float32) - 128
                    if block.shape != (8,8): continue
                    dct_block = cv2.dct(block)
                    for x, y in itertools.product(range(8), range(8)):
                        if x == 0 and y == 0 or len(extracted_bits) > max_len: continue
                        coeff = int(dct_block[x, y])
                        if coeff != 0: extracted_bits += str(coeff & 1)
        if not header_only:
            eom_index = extracted_bits.rfind('1111111111111110')
            if eom_index != -1: extracted_bits = extracted_bits[:eom_index]
        extracted_bits = extracted_bits.ljust((len(extracted_bits) + 7) & ~7, '0')
        return bytes(int(extracted_bits[i:i+8], 2) for i in range(0, len(extracted_bits), 8))
    
    def _is_video_file(self, path): return os.path.splitext(path)[1].lower() in ('.mp4', '.avi', '.mov', '.mkv')
    def _get_frame_from_video(self, path, timestamp=None):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError("Cannot open video")
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30
        target = count // 2 if timestamp is None else int(timestamp * fps)
        target = max(0, min(target, count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()
        if not ret: raise IOError("Failed to read frame")
        return frame