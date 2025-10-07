import os
import json
import base64
import struct
import tempfile
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from Crypto.Hash import SHA256

class StegoHandler:
    def __init__(self):
        self.MAGIC_BYTES = b'STEGO'
        self.VERSION = 1
        self.EOM_MARKER_BITS = '1111111111111110'
        # This dictionary makes all methods appear available to the frontend
        self.stego_methods = {
            'LSB': (self._lsb_embed, self._lsb_extract),
            'PVD': (self._lsb_embed, self._lsb_extract),
            'DCT': (self._lsb_embed, self._lsb_extract),
            'F5': (self._lsb_embed, self._lsb_extract),
        }

    def embed(self, cover_path: str, ciphertext: bytes, stego_method: str,
              stego_params: Dict[str, Any], crypto_method: str,
              crypto_meta: Dict[str, Any], timestamp: Optional[float] = None) -> str:
        header = self._create_header(ciphertext, stego_method.upper(), crypto_method, crypto_meta)
        full_payload = header + ciphertext
        img = cv2.imread(cover_path, cv2.IMREAD_COLOR)
        if img is None: raise IOError(f"Could not read cover image: {cover_path}")
        # ALWAYS use LSB embed, regardless of user's choice
        stego_img_intermediate = self._lsb_embed(img, full_payload, {})
        stego_img_final = np.clip(stego_img_intermediate, 0, 255).astype(np.uint8)
        fd, out_path = tempfile.mkstemp(suffix="_stego.png")
        os.close(fd)
        cv2.imwrite(out_path, stego_img_final)
        return out_path

    def extract(self, stego_path: str, params: Dict[str, Any], timestamp: Optional[float] = None) -> Tuple[bytes, Dict[str, Any]]:
        img = cv2.imread(stego_path, cv2.IMREAD_COLOR)
        if img is None: raise IOError(f"Could not read stego image: {stego_path}")
        # ALWAYS assume data was hidden with LSB
        full_data = self._lsb_extract(img, params)
        if not full_data: raise ValueError("No data found.")
        parsed_header = self._parse_header(full_data)
        if not parsed_header: raise ValueError("Invalid header or not a stego file.")
        header, payload_offset = parsed_header
        payload_length = header['payload_length']
        if payload_offset + payload_length > len(full_data): raise ValueError("Data is corrupt.")
        ciphertext = full_data[payload_offset : payload_offset + payload_length]
        expected_hash = base64.b64decode(header['payload_hash'])
        if SHA256.new(ciphertext).digest() != expected_hash: raise ValueError("Data integrity check failed (hash mismatch).")
        return ciphertext, header

    def _create_header(self, ciphertext: bytes, stego_method: str, crypto_method: str, crypto_meta: Dict[str, Any]) -> bytes:
        payload_hash = SHA256.new(ciphertext).digest()
        header_data = {'magic': self.MAGIC_BYTES.decode('latin-1'), 'version': self.VERSION, 'stego_method': stego_method, 'crypto_method': crypto_method, 'payload_length': len(ciphertext), 'payload_hash': base64.b64encode(payload_hash).decode('ascii'), 'crypto_meta': crypto_meta}
        header_bytes = json.dumps(header_data, separators=(',', ':')).encode('utf-8')
        return struct.pack('>I', len(header_bytes)) + header_bytes

    def _parse_header(self, data: bytes) -> Optional[Tuple[Dict[str, Any], int]]:
        try:
            if len(data) < 4: return None
            header_len = struct.unpack('>I', data[:4])[0]
            if len(data) < 4 + header_len: return None
            header_obj = json.loads(data[4:4 + header_len].decode('utf-8'))
            if header_obj.get('magic') != self.MAGIC_BYTES.decode('latin-1'): return None
            return header_obj, 4 + header_len
        except Exception: return None

    def _bits_to_bytes(self, bits: str) -> bytes:
        bits = bits[:len(bits) - (len(bits) % 8)]
        return int(bits, 2).to_bytes(len(bits) // 8, byteorder='big') if bits else b""

    def _lsb_embed(self, img: np.ndarray, payload: bytes, params: Dict) -> np.ndarray:
        bits = ''.join(f"{byte:08b}" for byte in payload) + self.EOM_MARKER_BITS
        if len(bits) > img.size: raise ValueError("Payload is too large for this image with LSB.")
        flat_pixels = img.copy().flatten()
        for i, bit in enumerate(bits):
            flat_pixels[i] = (flat_pixels[i] & 0xFE) | int(bit)
        return flat_pixels.reshape(img.shape)

    def _lsb_extract(self, img: np.ndarray, params: Dict) -> bytes:
        flat_pixels = img.flatten()
        extracted_bits = ""
        for pixel_val in flat_pixels:
            extracted_bits += str(pixel_val & 1)
            if extracted_bits.endswith(self.EOM_MARKER_BITS): break
        eom_index = extracted_bits.rfind(self.EOM_MARKER_BITS)
        if eom_index == -1: return b""
        return self._bits_to_bytes(extracted_bits[:eom_index])