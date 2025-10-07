import React, { useState, useRef, DragEvent } from 'react';
import axios from 'axios';

interface ExtractedData {
  type: string;
  content?: string;
  filename?: string;
  stego_method?: string;
  timestamp?: number;
}

const ExtractPage: React.FC = () => {
  const [stegoFile, setStegoFile] = useState<File | null>(null);
  const [encryptionMethod, setEncryptionMethod] = useState('AUTO'); // Auto detect by default
  const [password, setPassword] = useState('');
  const [privateKey, setPrivateKey] = useState('');
  const [timestamp, setTimestamp] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [extractedData, setExtractedData] = useState<ExtractedData | null>(null);
  const [extractedPreviewUrl, setExtractedPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) setStegoFile(files[0]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) setStegoFile(files[0]);
  };

  const isVideoFile = (file: File) => {
    const videoExt = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'];
    return videoExt.some(ext => file.name.toLowerCase().endsWith(ext));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!stegoFile) {
      setError('Please select a stego file');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setExtractedData(null);
    setExtractedPreviewUrl(null);

    try {
      const formData = new FormData();
      formData.append('stego_file', stegoFile);
      formData.append('encryption_method', encryptionMethod);

      // Only send keys/password if not auto
      const keyParams: any = {};
      if (encryptionMethod !== 'AUTO') {
        if (encryptionMethod === 'RSA-2048' && privateKey.trim()) keyParams.private_key = privateKey.trim();
        if (password.trim()) keyParams.password = password.trim();
      }
      formData.append('key_params', JSON.stringify(keyParams));

      if (stegoFile && isVideoFile(stegoFile) && timestamp) {
        formData.append('timestamp', timestamp);
      }

      const response = await axios.post('/api/extract', formData, {
        responseType: 'blob',
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const blob = response.data;

      // Try to detect JSON (text secret)
      let textData: string | null = null;
      try {
        textData = await blob.text();
      } catch {
        textData = null;
      }

      if (textData) {
        try {
          const jsonData = JSON.parse(textData);
          if (jsonData.type === 'text') {
            setExtractedData(jsonData);
            setResult('✅ Secret text successfully extracted!');
            return;
          }
        } catch {
          // Not JSON, fallback to file handling
        }
      }

      // Fallback: treat as file
      let filename = 'extracted_secret';
      const disposition = response.headers['content-disposition'];
      if (disposition) {
        const match = disposition.match(/filename="?([^";]+)"?/);
        if (match && match[1]) filename = match[1];
      }

      const url = window.URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();

      setResult(`✅ Secret file successfully extracted as ${filename}!`);

      if (blob.type.startsWith('image/') || blob.type.startsWith('video/')) {
        setExtractedPreviewUrl(url);
        setExtractedData({ type: blob.type.startsWith('image/') ? 'image' : 'video', filename });
      } else {
        window.URL.revokeObjectURL(url);
      }
    } catch (err: any) {
      if (err.response?.data) {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const errorData = JSON.parse(reader.result as string);
            setError(errorData.detail || 'An error occurred during extraction.');
          } catch {
            setError('An unknown error occurred during extraction.');
          }
        };
        reader.readAsText(err.response.data);
      } else {
        setError(err.message || 'An error occurred while extracting the secret');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>🔍 Extract Secret (Auto Detect Encryption)</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Stego Media (Image/Video)</label>
          <div
            className="file-upload"
            onDrop={handleFileDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            {stegoFile ? stegoFile.name : 'Click to select or drag & drop'}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>
        </div>

        <div className="form-group">
          <label>Encryption Method</label>
          <select
            value={encryptionMethod}
            className="form-control"
            onChange={(e) => setEncryptionMethod(e.target.value)}
          >
            <option value="AUTO">🔎 Auto Detect</option>
            <option value="AES-256">AES-256</option>
            <option value="RSA-2048">RSA-2048</option>
            <option value="Blowfish">Blowfish</option>
            <option value="Twofish">Twofish</option>
            <option value="ChaCha20">ChaCha20</option>
          </select>
        </div>

        {encryptionMethod !== 'AUTO' && (
          <div className="form-group">
            <label>{encryptionMethod === 'RSA-2048' ? 'Private Key (PEM)' : 'Password (optional)'}</label>
            {encryptionMethod === 'RSA-2048' ? (
              <textarea
                className="form-control"
                value={privateKey}
                onChange={(e) => setPrivateKey(e.target.value)}
                placeholder="Paste your RSA private key here"
                rows={6}
              />
            ) : (
              <input
                className="form-control"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password if used"
              />
            )}
          </div>
        )}

        {stegoFile && isVideoFile(stegoFile) && (
          <div className="form-group">
            <label>Timestamp (seconds)</label>
            <input
              type="number"
              className="form-control"
              value={timestamp}
              onChange={(e) => setTimestamp(e.target.value)}
              step="0.1"
              min="0"
              placeholder="0.0"
            />
          </div>
        )}

        <button type="submit" className="btn btn-block" disabled={isLoading}>
          {isLoading ? 'Extracting...' : '🔍 Extract Secret'}
        </button>
      </form>

      {isLoading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Processing your request...</p>
        </div>
      )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}
      {result && (
        <div className="result-section">
          <h3>Result</h3>
          <p>{result}</p>
        </div>
      )}

      {extractedData?.type === 'text' && (
        <div className="result-section">
          <h4>Extracted Text:</h4>
          <pre>{extractedData.content}</pre>
        </div>
      )}

      {extractedPreviewUrl && extractedData && (
        <div className="result-section">
          {extractedData.type === 'video' ? (
            <video src={extractedPreviewUrl} controls style={{ maxWidth: '100%' }} />
          ) : (
            <img src={extractedPreviewUrl} alt="Extracted Preview" style={{ maxWidth: '100%', borderRadius: '8px' }} />
          )}
        </div>
      )}
    </div>
  );
};

export default ExtractPage;
  