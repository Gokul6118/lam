import os
import tempfile
import json
import traceback
import base64
from pathlib import Path

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn

from crypto_module import CryptoHandler
from stego_module import StegoHandler
from utils import cleanup_temp_files

# --- FastAPI App Initialization ---
app = FastAPI(title="Steganography API", version="10.0.0-final")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Stego-Method", "X-Crypto-Method", "X-Generated-Private-Key"]
)

api_router = APIRouter(prefix="/api")

# --- Singleton Handlers ---
crypto_handler = CryptoHandler()
stego_handler = StegoHandler()

# --- API Endpoints ---
@api_router.get("/algorithms")
async def get_algorithms():
    return {
        "encryption": list(crypto_handler.supported_methods.keys()),
        "steganography": list(stego_handler.stego_methods.keys())
    }

@api_router.post("/embed")
async def embed_secret(
    background_tasks: BackgroundTasks,
    cover_file: UploadFile = File(...),
    encryption_method: str = Form(...),
    stego_method: str = Form(...),
    secret_text: str = Form(None),
    secret_file: UploadFile = File(None),
    key_params: str = Form("{}"),
    stego_params: str = Form("{}"),
    timestamp: float = Form(None)
):
    if not secret_text and not secret_file:
        raise HTTPException(status_code=400, detail="Either secret_text or secret_file must be provided.")
    
    temp_cover_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(cover_file.filename).suffix) as temp_cover:
            temp_cover.write(await cover_file.read())
            temp_cover_path = temp_cover.name

        secret_data = secret_text.encode('utf-8') if secret_text else await secret_file.read()
        key_params_dict = json.loads(key_params)
        
        encrypted_data, crypto_meta = crypto_handler.encrypt(secret_data, encryption_method, key_params_dict)
        if secret_file and secret_file.filename:
            crypto_meta["filename"] = secret_file.filename

        stego_path = stego_handler.embed(
            cover_path=temp_cover_path, ciphertext=encrypted_data,
            stego_method=stego_method, stego_params=key_params_dict,
            crypto_method=encryption_method, crypto_meta=crypto_meta,
            timestamp=timestamp
        )
        
        background_tasks.add_task(cleanup_temp_files, [temp_cover_path, stego_path])

        headers = {}
        if crypto_meta.get("priv_key") and not key_params_dict.get("public_key"):
            encoded_key = base64.b64encode(crypto_meta["priv_key"].encode("utf-8")).decode("utf-8")
            headers["X-Generated-Private-Key"] = encoded_key

        return FileResponse(
            stego_path, media_type="image/png",
            filename=f"stego_{Path(cover_file.filename).stem}.png",
            background=background_tasks,
            headers=headers
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    finally:
        cleanup_temp_files([temp_cover_path])

@api_router.post("/extract")
async def extract_secret(
    background_tasks: BackgroundTasks,
    stego_file: UploadFile = File(...),
    key_params: str = Form("{}"),
    timestamp: float = Form(None)
):
    temp_stego_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(stego_file.filename).suffix) as tmp_file:
            contents = await stego_file.read()
            tmp_file.write(contents)
            temp_stego_path = tmp_file.name
        
        background_tasks.add_task(cleanup_temp_files, [temp_stego_path])
        key_params_dict = json.loads(key_params)
        
        encrypted_bytes, header_meta = stego_handler.extract(
            stego_path=temp_stego_path, 
            params=key_params_dict, 
            timestamp=timestamp
        )
        
        encryption_method = header_meta['crypto_method']
        crypto_meta = header_meta.get("crypto_meta", {})
        
        decrypted_data = crypto_handler.decrypt(encrypted_bytes, encryption_method, key_params_dict, crypto_meta)

        try:
            content = decrypted_data.decode('utf-8')
            return JSONResponse(content={
                "type": "text", "content": content,
                "stego_method": header_meta['stego_method'], "crypto_method": encryption_method
            })
        except UnicodeDecodeError:
            output_filename = crypto_meta.get("filename", "extracted_secret.bin")
            headers = {
                "Content-Disposition": f"attachment; filename=\"{output_filename}\"",
                "X-Stego-Method": header_meta['stego_method'],
                "X-Crypto-Method": encryption_method
            }
            return Response(content=decrypted_data, media_type="application/octet-stream", headers=headers)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Steganography System API is running."}

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)