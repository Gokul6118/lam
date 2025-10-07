import React, { useState, useRef, DragEvent, useEffect } from 'react';
import axios from 'axios';

interface Algorithms {
  encryption: string[];
  steganography: string[];
}

const EmbedPage: React.FC = () => {
  const [coverFile, setCoverFile] = useState<File | null>(null);
  const [secretText, setSecretText] = useState('');
  const [secretFile, setSecretFile] = useState<File | null>(null);
  const [encryptionMethod, setEncryptionMethod] = useState('');
  const [stegoMethod, setStegoMethod] = useState('');
  const [timestamp, setTimestamp] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [algorithms, setAlgorithms] = useState<Algorithms | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const secretFileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchAlgorithms = async () => {
      try {
        const response = await axios.get('/api/algorithms');
        setAlgorithms(response.data);
        
        if (response.data.encryption && response.data.encryption.length > 0) {
          setEncryptionMethod(response.data.encryption[0]);
        }
        // Set default stego method to the first one in the filtered list
        if (response.data.steganography && response.data.steganography.length > 0) {
          const filteredStego = response.data.steganography.filter(
            (method: string) => method !== 'LSB_COLOR' && method !== 'BITPLANE'
          );
          if (filteredStego.length > 0) {
            setStegoMethod(filteredStego[0]);
          }
        }
        console.log('Algorithms loaded and defaults set:', response.data);
      } catch (err) {
        console.error('Failed to fetch algorithms:', err);
        setError('Failed to load algorithm information. Please ensure the backend is running and refresh the page.');
      }
    };
    fetchAlgorithms();
  }, []);

  const handleFileDrop = (e: DragEvent<HTMLDivElement>, type: 'cover' | 'secret') => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (type === 'cover') {
        setCoverFile(file);
      } else {
        setSecretFile(file);
        setSecretText('');
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>, type: 'cover' | 'secret') => {
    const file = e.target.files?.[0];
    if (file) {
      if (type === 'cover') {
        setCoverFile(file);
      } else {
        setSecretFile(file);
        setSecretText('');
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!coverFile) {
      setError('Please select a cover file');
      return;
    }
    if (!secretText && !secretFile) {
      setError('Please provide either secret text or a secret file');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('cover_file', coverFile);
      if (secretText) {
        formData.append('secret_text', secretText);
      } else if (secretFile) {
        formData.append('secret_file', secretFile);
      }
      formData.append('encryption_method', encryptionMethod);
      formData.append('stego_method', stegoMethod);
      if (timestamp) {
        formData.append('timestamp', timestamp);
      }

      const response = await axios.post('/api/embed', formData, {
        responseType: 'blob',
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      const friendlyFileName = coverFile.name.substring(0, coverFile.name.lastIndexOf('.')) || coverFile.name;
      const extension = (coverFile.name.split('.').pop() || 'png');
      link.href = url;
      link.setAttribute('download', `stego_${friendlyFileName}.${extension}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      setResult(`Secret successfully embedded! File downloaded as stego_${friendlyFileName}.${extension}`);

    } catch (err: any) {
      if (err.response?.data) {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const errorData = JSON.parse(reader.result as string);
            setError(errorData.detail || 'An unknown error occurred during embedding.');
          } catch {
            setError('An error occurred. The server response was not in the expected format.');
          }
        };
        reader.readAsText(err.response.data);
      } else {
        setError('A network error occurred. Could not connect to the backend.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const isVideoFile = (file: File | null) => {
    if (!file) return false;
    const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'];
    return videoExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  };

  return (
    <div className="card">
      <h2>🔐 Embed Secret</h2>
      <form onSubmit={handleSubmit}>
        
        <div className="form-group">
          <label>Cover Media (Image/Video)</label>
          <div 
            className="file-upload"
            onDrop={(e) => handleFileDrop(e, 'cover')}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="file-upload-text">
              {coverFile ? coverFile.name : 'Click to select or drag & drop'}
            </div>
            <input
              ref={fileInputRef} type="file"
              accept="image/*,video/*"
              onChange={(e) => handleFileSelect(e, 'cover')}
              className="hidden-input"
            />
          </div>
        </div>

        <div className="form-group">
          <label>Secret to Hide</label>
          <textarea
            className="form-control"
            value={secretText}
            onChange={(e) => { setSecretText(e.target.value); setSecretFile(null); }}
            placeholder="Enter your secret message here..."
            rows={4}
            disabled={!!secretFile}
          />
          <div className="or-divider">OR</div>
          <div 
            className="file-upload"
            onDrop={(e) => handleFileDrop(e, 'secret')}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => secretFileInputRef.current?.click()}
          >
            <div className="file-upload-text">
              {secretFile ? secretFile.name : 'Click to select or drag & drop a secret file'}
            </div>
            <input
              ref={secretFileInputRef}
              type="file"
              onChange={(e) => handleFileSelect(e, 'secret')}
              style={{ display: 'none' }}
            />
          </div>
        </div>

        <div className="form-group">
          <label>Algorithms</label>
          <div className="select-group">
            <div>
              <label>Encryption Method</label>
              <select
                className="form-control"
                value={encryptionMethod}
                onChange={(e) => setEncryptionMethod(e.target.value)}
                disabled={!algorithms}
              >
                {algorithms ? (
                  algorithms.encryption.map(method => (
                    <option key={method} value={method}>{method}</option>
                  ))
                ) : (
                  <option>Loading...</option>
                )}
              </select>
            </div>
            <div>
              <label>Steganography Method</label>
              <select
                className="form-control"
                value={stegoMethod}
                onChange={(e) => setStegoMethod(e.target.value)}
                disabled={!algorithms}
              >
                {algorithms ? (
                  algorithms.steganography
                    // FIXED: Filter out unwanted methods from the UI
                    .filter(method => method !== 'LSB_COLOR' && method !== 'BITPLANE')
                    .map(method => (
                      <option key={method} value={method}>{method}</option>
                    ))
                ) : (
                  <option>Loading...</option>
                )}
              </select>
            </div>
          </div>
        </div>

        {isVideoFile(coverFile) && (
          <div className="form-group">
            <label>Timestamp for Video Frame (in seconds)</label>
            <input
              type="number"
              className="form-control"
              value={timestamp}
              onChange={(e) => setTimestamp(e.target.value)}
              placeholder="e.g., 10.5 (defaults to middle frame)"
              step="0.1"
              min="0"
            />
          </div>
        )}

        <button type="submit" className="btn btn-block" disabled={isLoading}>
          {isLoading ? 'Embedding...' : '🔐 Embed Secret'}
        </button>
      </form>

      {isLoading && <div className="loading"><div className="loading-spinner"></div></div>}
      {error && <div className="error-message"><strong>Error:</strong> {error}</div>}
      {result && <div className="result-section"><h3>✅ Success!</h3><p>{result}</p></div>}
    </div>
  );
};

export default EmbedPage;