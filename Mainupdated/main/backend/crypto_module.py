import base64
from typing import Tuple, Dict, Any
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256

class CryptoHandler:
    def __init__(self):
        self.supported_methods = {
            'AES-256': self._aes_encrypt,
            'RSA-2048': self._rsa_encrypt
        }
        self.decrypt_methods = {
            'AES-256': self._aes_decrypt,
            'RSA-2048': self._rsa_decrypt
        }

    def encrypt(self, data: bytes, method: str, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported encryption method: {method}")
        return self.supported_methods[method](data, params)

    def decrypt(self, ciphertext: bytes, method: str, params: Dict[str, Any], meta: Dict[str, Any]) -> bytes:
        if method not in self.decrypt_methods:
            raise ValueError(f"Unsupported decryption method: {method}")
        return self.decrypt_methods[method](ciphertext, params, meta)

    def _aes_encrypt(self, data: bytes, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        if 'password' in params and params.get('password'):
            salt = get_random_bytes(32)
            key = PBKDF2(params['password'].encode(), salt, 32, count=1000000, hmac_hash_module=SHA256)
            key_b64, salt_b64 = None, base64.b64encode(salt).decode()
        else:
            key = get_random_bytes(32)
            key_b64, salt_b64 = base64.b64encode(key).decode(), None
        iv = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        meta = {'alg': 'AES-256-GCM', 'key': key_b64, 'salt': salt_b64, 'iv': base64.b64encode(iv).decode(), 'tag_length': len(tag)}
        return ciphertext + tag, meta

    def _aes_decrypt(self, ciphertext_with_tag: bytes, params: Dict[str, Any], meta: Dict[str, Any]) -> bytes:
        if meta.get('key'):
            key = base64.b64decode(meta['key'])
        elif 'password' in params and params.get('password'):
            salt = base64.b64decode(meta['salt'])
            key = PBKDF2(params['password'].encode(), salt, 32, count=1000000, hmac_hash_module=SHA256)
        else:
            raise ValueError("AES password is required for decryption.")
        iv = base64.b64decode(meta['iv'])
        tag_length = meta['tag_length']
        ciphertext, tag = ciphertext_with_tag[:-tag_length], ciphertext_with_tag[-tag_length:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
        return cipher.decrypt_and_verify(ciphertext, tag)

    def _rsa_encrypt(self, data: bytes, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        if 'public_key' not in params or not params['public_key']:
            key_pair = RSA.generate(2048)
            params['private_key'] = key_pair.export_key().decode()
            params['public_key'] = key_pair.publickey().export_key().decode()
        session_key = get_random_bytes(32)
        iv = get_random_bytes(16)
        cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
        encrypted_data = cipher_aes.encrypt(pad(data, AES.block_size))
        public_key = RSA.import_key(params['public_key'])
        cipher_rsa = PKCS1_OAEP.new(public_key)
        encrypted_session_key = cipher_rsa.encrypt(session_key)
        meta = {'alg': 'RSA-2048-hybrid', 'priv_key': params.get('private_key'), 'pub_key': params.get('public_key'), 'iv': base64.b64encode(iv).decode(), 'aes_key_len': len(encrypted_session_key)}
        return encrypted_session_key + encrypted_data, meta

    def _rsa_decrypt(self, ciphertext: bytes, params: Dict[str, Any], meta: Dict[str, Any]) -> bytes:
        private_key = RSA.import_key(params['private_key'])
        aes_key_len = meta['aes_key_len']
        encrypted_session_key, encrypted_data = ciphertext[:aes_key_len], ciphertext[aes_key_len:]
        cipher_rsa = PKCS1_OAEP.new(private_key)
        session_key = cipher_rsa.decrypt(encrypted_session_key)
        iv = base64.b64decode(meta['iv'])
        cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
        return unpad(cipher_aes.decrypt(encrypted_data), AES.block_size)