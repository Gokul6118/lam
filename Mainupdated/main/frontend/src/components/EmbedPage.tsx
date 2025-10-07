import React, { useState, useRef, DragEvent, useEffect } from 'react';
import axios from 'axios';

// --- TYPE DEFINITIONS ---
interface Algorithms {
  encryption: string[];
  steganography: string[];
}

// --- COMPONENT ---
const EmbedPage: React.FC = () => {
  // --- STATE ---
  const [coverFile, setCoverFile] = useState<File | null>(null);
  const [secretText, setSecretText] = useState('');
  const [secretFile, setSecretFile] = useState<File | null>(null);
  const [encryptionMethod, setEncryptionMethod] = useState('');
  const [stegoMethod, setStegoMethod] = useState('');
  const [password, setPassword] = useState('');
  const [publicKey, setPublicKey] = useState('');
  const [generatedPrivateKey, setGeneratedPrivateKey] = useState<string | null>(null);
  const [isKeyCopied, setIsKeyCopied] = useState(false);
  const [timestamp, setTimestamp] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [algorithms, setAlgorithms] = useState<Algorithms | null>(null);

  // --- REFS ---
  const coverFileInputRef = useRef<HTMLInputElement>(null);
  const secretFileInputRef = useRef<HTMLInputElement>(null);

  // --- EFFECTS ---
  useEffect(() => {
    const fetchAlgorithms = async () => {
      try {
        const response = await axios.get('/api/algorithms');
        const data: Algorithms = response.data;
        if (data && data.encryption && data.steganography) {
            setAlgorithms(data);
            if (data.encryption.length > 0) setEncryptionMethod(data.encryption[0]);
            if (data.steganography.length > 0) setStegoMethod(data.steganography[0]);
        } else {
            throw new Error("Invalid algorithm data from server.");
        }
      } catch (err) {
        console.error('Failed to fetch algorithms:', err);
        setError('Failed to load algorithm information. Please ensure the backend is running.');
      }
    };
    fetchAlgorithms();
  }, []);

  // --- HANDLERS ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!coverFile) {
      setError('Please select a cover file.');
      return;
    }
    if (!secretText && !secretFile) {
      setError('Please provide either secret text or a secret file.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setGeneratedPrivateKey(null);
    setIsKeyCopied(false);

    try {
      const formData = new FormData();
      formData.append('cover_file', coverFile);
      formData.append('encryption_method', encryptionMethod);
      formData.append('stego_method', stegoMethod);

      if (secretText) formData.append('secret_text', secretText);
      else if (secretFile) formData.append('secret_file', secretFile);
      if (timestamp) formData.append('timestamp', timestamp);

      const keyParams: any = {};
      if (password) keyParams.password = password;
      if (publicKey) keyParams.public_key = publicKey;
      formData.append('key_params', JSON.stringify(keyParams));

      const response = await axios.post('/api/embed', formData, { responseType: 'blob' });
      
      const genKeyHeader = response.headers['x-generated-private-key'];
      if (genKeyHeader && typeof genKeyHeader === 'string') {
        const decodedKey = atob(genKeyHeader);
        setGeneratedPrivateKey(decodedKey);
      }

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      const friendlyFileName = coverFile.name.substring(0, coverFile.name.lastIndexOf('.')) || coverFile.name;
      
      link.href = url;
      link.setAttribute('download', `stego_${friendlyFileName}.png`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      setResult('Secret successfully embedded! Your file has been downloaded.');

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

  const handleCopyKey = () => {
    if (generatedPrivateKey) {
      navigator.clipboard.writeText(generatedPrivateKey);
      setIsKeyCopied(true);
      setTimeout(() => setIsKeyCopied(false), 2000);
    }
  };

  const isVideoFile = (file: File | null) => {
    if (!file) return false;
    const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'];
    return videoExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  };
  
  const requiresPassword = ['AES-256', 'Blowfish', 'Salsa20', 'ChaCha20'].includes(encryptionMethod) || stegoMethod === 'F5';
  const requiresPublicKey = encryptionMethod === 'RSA-2048';

  return (
    <div className="card">
      <h2>🔐 Embed Secret</h2>
      <form onSubmit={handleSubmit}>
        
        <div className="form-group">
          <label>1. Cover Media (Image/Video)</label>
          <div className="file-upload" onClick={() => coverFileInputRef.current?.click()}>
            <div className="file-upload-text">{coverFile ? coverFile.name : 'Click to select or drag & drop'}</div>
            <input ref={coverFileInputRef} type="file" accept="image/*,video/*" onChange={(e) => setCoverFile(e.target.files?.[0] || null)} style={{ display: 'none' }} />
          </div>
        </div>

        <div className="form-group">
          <label>2. Secret to Hide</label>
          <textarea className="form-control" value={secretText} onChange={(e) => { setSecretText(e.target.value); setSecretFile(null); }} placeholder="Enter your secret message here..." rows={4} disabled={!!secretFile} />
          <div className="or-divider">OR</div>
          <div className="file-upload" onClick={() => secretFileInputRef.current?.click()}>
            <div className="file-upload-text">{secretFile ? secretFile.name : 'Click to select a secret file'}</div>
            <input ref={secretFileInputRef} type="file" onChange={(e) => { setSecretFile(e.target.files?.[0] || null); setSecretText(''); }} style={{ display: 'none' }}/>
          </div>
        </div>

        <div className="form-group">
          <label>3. Algorithms</label>
          <div className="select-group">
            <div>
              <label>Encryption Method</label>
              <select className="form-control" value={encryptionMethod} onChange={(e) => setEncryptionMethod(e.target.value)} disabled={!algorithms}>
                {algorithms?.encryption.map((method: string) => <option key={method} value={method}>{method}</option>) || <option>Loading...</option>}
              </select>
            </div>
            <div>
              <label>Steganography Method</label>
              <select className="form-control" value={stegoMethod} onChange={(e) => setStegoMethod(e.target.value)} disabled={!algorithms}>
                {algorithms?.steganography.map((method: string) => <option key={method} value={method}>{method}</option>) || <option>Loading...</option>}
              </select>
            </div>
          </div>
        </div>
        
        {(requiresPassword || requiresPublicKey) && (
          <div className="form-group">
            <label>4. Encryption Key</label>
            {requiresPassword && <input className="form-control" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter Password (for AES, F5, etc.)"/>}
            {requiresPublicKey && <textarea className="form-control" value={publicKey} onChange={(e) => setPublicKey(e.target.value)} placeholder="Paste Public Key (optional, will be generated if blank)" rows={2} style={{ marginTop: requiresPassword ? '10px' : '0' }}/>}
          </div>
        )}

        {isVideoFile(coverFile) && (
          <div className="form-group">
            <label>Timestamp for Video (Optional)</label>
            <input type="number" className="form-control" value={timestamp} onChange={(e) => setTimestamp(e.target.value)} placeholder="e.g., 10.5 (seconds)" step="0.1" min="0" />
          </div>
        )}

        <button type="submit" className="btn btn-block" disabled={isLoading}>{isLoading ? 'Embedding...' : '🔐 Embed Secret'}</button>
      </form>

      {isLoading && <div className="loading"><div className="loading-spinner"></div><p>Embedding secret...</p></div>}
      {error && <div className="error-message"><strong>Error:</strong> {error}</div>}
      {result && <div className="result-section"><h3>✅ Success!</h3><p>{result}</p></div>}
      
      {generatedPrivateKey && (
        <div className="result-section tech-details">
          <h4>Generated RSA Private Key:</h4>
          <p><strong>IMPORTANT: Save this key! You will need it to extract your secret.</strong></p>
          <pre>{generatedPrivateKey}</pre>
          <button onClick={handleCopyKey} className="btn btn-small">{isKeyCopied ? 'Copied!' : 'Copy Key'}</button>
        </div>
      )}
    </div>
  );
};

export default EmbedPage;