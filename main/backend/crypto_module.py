import base64
from typing import Tuple, Dict, Any
from Crypto.Cipher import AES, PKCS1_OAEP, Blowfish, ChaCha20_Poly1305
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256

class CryptoHandler:
    def __init__(self):
        self.supported_methods = {'AES-256': self._aes_encrypt, 'RSA-2048': self._rsa_encrypt, 'Blowfish': self._blowfish_encrypt, 'ChaCha20': self._chacha20_encrypt}
        self.decrypt_methods = {'AES-256': self._aes_decrypt, 'RSA-2048': self._rsa_decrypt, 'Blowfish': self._blowfish_decrypt, 'ChaCha20': self._chacha20_decrypt}
    def encrypt(self, d, m, p): 
        if m not in self.supported_methods: raise ValueError(f"Unsupported method: {m}")
        return self.supported_methods[m](d, p)
    def decrypt(self, c, m, p, meta):
        if m not in self.decrypt_methods: raise ValueError(f"Unsupported method: {m}")
        return self.decrypt_methods[m](c, p, meta)
    def _aes_encrypt(self, d, p):
        s=get_random_bytes(32);k=PBKDF2(p['password'].encode(),s,32,1000000,SHA256) if 'password' in p else get_random_bytes(32)
        iv=get_random_bytes(12);ci=AES.new(k,AES.MODE_GCM,nonce=iv);ct,t=ci.encrypt_and_digest(d)
        return ct+t,{'alg':'AES-256-GCM','key':base64.b64encode(k).decode() if 'password' not in p else None,'salt':base64.b64encode(s).decode() if 'password' in p else None,'iv':base64.b64encode(iv).decode(),'tag_length':len(t)}
    def _aes_decrypt(self, c, p, m):
        k=base64.b64decode(m['key']) if m.get('key') else PBKDF2(p['password'].encode(),base64.b64decode(m['salt']),32,1000000,SHA256)
        iv,tl=base64.b64decode(m['iv']),m['tag_length'];ac,t=c[:-tl],c[-tl:]
        return AES.new(k,AES.MODE_GCM,nonce=iv).decrypt_and_verify(ac,t)
    def _rsa_encrypt(self, d, p):
        if 'public_key' not in p:pk_obj=RSA.generate(2048);p['private_key'],p['public_key']=pk_obj.export_key().decode(),pk_obj.publickey().export_key().decode()
        ak,iv=get_random_bytes(32),get_random_bytes(16);ci,pd=AES.new(ak,AES.MODE_CBC,iv),pad(d,AES.block_size);ed=ci.encrypt(pd)
        pub_k=RSA.import_key(p['public_key']);rc,eak=PKCS1_OAEP.new(pub_k),rc.encrypt(ak)
        return eak+ed,{'alg':'RSA-2048-hybrid','priv_key':p.get('private_key'),'pub_key':p.get('public_key'),'iv':base64.b64encode(iv).decode(),'aes_key_len':len(eak)}
    def _rsa_decrypt(self, c, p, m):
        priv_k=RSA.import_key(p['private_key']);akl=m['aes_key_len'];eak,ed=c[:akl],c[akl:]
        rc,ak=PKCS1_OAEP.new(priv_k),rc.decrypt(eak);iv=base64.b64decode(m['iv'])
        return unpad(AES.new(ak,AES.MODE_CBC,iv).decrypt(ed),AES.block_size)
    def _blowfish_encrypt(self, d, p):
        s=get_random_bytes(16);k=PBKDF2(p['password'].encode(),s,56,1000000,SHA256) if 'password' in p else get_random_bytes(56)
        iv=get_random_bytes(8);ci=Blowfish.new(k,Blowfish.MODE_CBC,iv);pd=pad(d,Blowfish.block_size)
        return ci.encrypt(pd),{'alg':'Blowfish-CBC','key':base64.b64encode(k).decode() if 'password' not in p else None,'salt':base64.b64encode(s).decode() if 'password' in p else None,'iv':base64.b64encode(iv).decode()}
    def _blowfish_decrypt(self, c, p, m):
        k=base64.b64decode(m['key']) if m.get('key') else PBKDF2(p['password'].encode(),base64.b64decode(m['salt']),56,1000000,SHA256)
        iv=base64.b64decode(m['iv']);ci=Blowfish.new(k,Blowfish.MODE_CBC,iv)
        return unpad(ci.decrypt(c),Blowfish.block_size)
    def _chacha20_encrypt(self, d, p):
        s=get_random_bytes(32);k=PBKDF2(p['password'].encode(),s,32,1000000,SHA256) if 'password' in p else get_random_bytes(32)
        n=get_random_bytes(12);ci=ChaCha20_Poly1305.new(key=k,nonce=n);ct,t=ci.encrypt_and_digest(d)
        return ct+t,{'alg':'ChaCha20-Poly1305','key':base64.b64encode(k).decode() if 'password' not in p else None,'salt':base64.b64encode(s).decode() if 'password' in p else None,'nonce':base64.b64encode(n).decode(),'tag_length':len(t)}
    def _chacha20_decrypt(self, c, p, m):
        k=base64.b64decode(m['key']) if m.get('key') else PBKDF2(p['password'].encode(),base64.b64decode(m['salt']),32,1000000,SHA256)
        n,tl=base64.b64decode(m['nonce']),m['tag_length'];ac,t=c[:-tl],c[-tl:]
        return ChaCha20_Poly1305.new(key=k,nonce=n).decrypt_and_verify(ac,t)