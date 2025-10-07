# 🔐 Steganography System

A comprehensive steganography system that allows you to hide secret messages and files within images and videos using various encryption and steganography algorithms.

## ✨ Features

- **Multiple Encryption Methods**: AES-256, RSA-2048, Blowfish, Twofish, ChaCha20
- **Advanced Steganography**: LSB, DCT, DWT, F5, PVD algorithms
- **Media Support**: Images (PNG, JPG, BMP, TIFF, GIF) and Videos (MP4, AVI, MOV, MKV, WMV, FLV)
- **User-Friendly Interface**: Modern React frontend with drag-and-drop support
- **Robust Backend**: FastAPI-based backend with comprehensive error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the backend server:**
   ```bash
   python main.py
   ```

   The backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## 🧪 Testing

### Backend Testing

Run the backend test script to verify functionality:

```bash
cd backend
python test_backend.py
```

### Manual Testing

1. Open `http://localhost:5173` in your browser
2. Use the **Embed** page to hide secrets in images/videos
3. Use the **Extract** page to recover hidden secrets

## 📚 Usage Guide

### Embedding Secrets

1. **Select Cover Media**: Choose an image or video file
2. **Enter Secret**: Type text or upload a file to hide
3. **Choose Algorithms**: Select encryption and steganography methods
4. **Set Parameters**: Configure algorithm-specific options
5. **Embed**: Click "Embed Secret" to process
6. **Download**: The stego file will be automatically downloaded

### Extracting Secrets

1. **Upload Stego File**: Select the file containing hidden secrets
2. **Select Method**: Choose the encryption method used during embedding
3. **Provide Key**: Enter password or private key
4. **Extract**: Click "Extract Secret" to recover the hidden data
5. **View Result**: The secret will be displayed or downloaded

## 🔧 Algorithm Details

### Encryption Methods

| Method | Type | Key Size | Mode | Description |
|--------|------|----------|------|-------------|
| AES-256 | Symmetric | 256 bits | GCM | Advanced Encryption Standard with authentication |
| RSA-2048 | Asymmetric | 2048 bits | Hybrid | Public-key encryption with AES key wrapping |
| Blowfish | Symmetric | 448 bits | CBC | Fast symmetric block cipher |
| Twofish | Symmetric | 256 bits | CBC | Advanced symmetric block cipher |
| ChaCha20 | Symmetric | 256 bits | Poly1305 | Stream cipher with authentication |

### Steganography Methods

| Method | Capacity | Robustness | Description |
|--------|----------|------------|-------------|
| LSB | High | Low | Least Significant Bit - high capacity, low robustness |
| DCT | Medium | Medium | Discrete Cosine Transform - balanced approach |
| DWT | Medium | High | Discrete Wavelet Transform - high robustness |
| F5 | Low | High | JPEG steganography - very robust |
| PVD | Medium | Medium | Pixel Value Differencing - good balance |

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all Python dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CORS Issues**: The backend allows all origins for development. Check browser console for errors.

3. **File Upload Issues**: Ensure file types are supported and file sizes are reasonable.

4. **Algorithm Errors**: Some algorithms may not work with all file types. Try different combinations.

### Performance Tips

- Use LSB for high-capacity requirements
- Use F5 or DWT for high-robustness requirements
- For videos, specify timestamps to avoid processing entire files
- Large files may take longer to process

## 🔒 Security Notes

- **Encryption Keys**: Keep your encryption keys secure and private
- **File Validation**: The system validates file types but always scan files from untrusted sources
- **Algorithm Selection**: Choose algorithms based on your security requirements
- **Testing**: Test with sample files before using with sensitive data

## 📁 Project Structure

```
mainproject/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # Main API server
│   ├── crypto_module.py    # Encryption/decryption logic
│   ├── stego_module.py     # Steganography algorithms
│   ├── utils.py            # Utility functions
│   ├── requirements.txt    # Python dependencies
│   └── test_backend.py     # Backend testing script
├── frontend/                # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.tsx         # Main app component
│   │   └── main.tsx        # App entry point
│   ├── package.json        # Node.js dependencies
│   └── vite.config.ts      # Vite configuration
└── README.md               # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV for image and video processing
- PyCryptodome for cryptographic operations
- PyWavelets for wavelet transforms
- FastAPI for the backend framework
- React for the frontend framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages in the browser console
3. Check the backend server logs
4. Create an issue in the repository

---

**Happy Steganography! 🔐✨**
