import os
import tempfile # FIXED: Added missing import for tempfile
import json
from pathlib import Path
import traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

import uvicorn

from crypto_module import CryptoHandler
from stego_module import StegoHandler
from utils import cleanup_temp_files

# --- FastAPI app ---
app = FastAPI(title="Steganography System", version="1.2.1") # Incremented version

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Handlers ---
crypto_handler = CryptoHandler()
stego_handler = StegoHandler()

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Steganography System API", "status": "running"}

@app.get("/algorithms")
async def get_algorithms():
    """Dynamically get the list of supported algorithms from the backend."""
    return {
        "encryption": list(crypto_handler.supported_methods.keys()),
        "steganography": list(stego_handler.stego_methods.keys())
    }

@app.post("/embed")
async def embed_secret(
    background_tasks: BackgroundTasks,
    cover_file: UploadFile = File(...),
    secret_text: str = Form(None),
    secret_file: UploadFile = File(None),
    encryption_method: str = Form(...),
    stego_method: str = Form(...),
    key_params: str = Form("{}"),
    stego_params: str = Form("{}"),
    timestamp: float = Form(None)
):
    if not secret_text and not secret_file:
        raise HTTPException(status_code=400, detail="Either secret_text or secret_file must be provided")

    temp_cover_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(cover_file.filename).suffix) as temp_cover:
            temp_cover.write(await cover_file.read())
            temp_cover_path = temp_cover.name

        secret_data = secret_text.encode('utf-8') if secret_text else await secret_file.read()
        key_params_dict = json.loads(key_params)
        stego_params_dict = json.loads(stego_params)

        encrypted_data, crypto_meta = crypto_handler.encrypt(secret_data, encryption_method, key_params_dict)
        if secret_file and secret_file.filename:
            crypto_meta["filename"] = secret_file.filename

        stego_path = stego_handler.embed(
            cover_path=temp_cover_path,
            ciphertext=encrypted_data,
            stego_method=stego_method,
            stego_params=stego_params_dict,
            crypto_method=encryption_method,
            crypto_meta=crypto_meta,
            timestamp=timestamp
        )

        background_tasks.add_task(cleanup_temp_files, [stego_path])

        return FileResponse(
            stego_path,
            media_type="application/octet-stream",
            filename=f"stego_{Path(cover_file.filename).name}",
            background=background_tasks
        )
    except Exception as e:
        print(f"Error in /embed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during embedding: {str(e)}")
    finally:
        cleanup_temp_files([temp_cover_path])

@app.post("/extract")
async def extract_secret(
    stego_file: UploadFile = File(...),
    key_params: str = Form("{}"),
    timestamp: float = Form(None)
):
    if not stego_file.filename:
        raise HTTPException(status_code=400, detail="Stego file is required")

    temp_stego_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(stego_file.filename).suffix) as tmp_file:
            tmp_file.write(await stego_file.read())
            temp_stego_path = tmp_file.name

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
            return JSONResponse(content={"type": "text", "content": content})
        except UnicodeDecodeError:
            output_filename = crypto_meta.get("filename", "extracted_secret.bin")
            return Response(
                content=decrypted_data,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={output_filename}"}
            )
    except Exception as e:
        print(f"Error in /extract: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during extraction: {str(e)}")
    finally:
        cleanup_temp_files([temp_stego_path])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)