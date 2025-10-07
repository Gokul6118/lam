import React, { useState, useRef, DragEvent } from 'react';
import axios from 'axios';

const StegnoGANPage: React.FC = () => {
  const [trainingImages, setTrainingImages] = useState<File[]>([]);
  const [epochs, setEpochs] = useState(100);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch model status on component mount
  React.useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const response = await axios.get('/api/stegnogan/status');
      setModelStatus(response.data);
    } catch (err) {
      console.error('Failed to fetch model status:', err);
    }
  };

  const handleFileDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    setTrainingImages(prev => [...prev, ...imageFiles]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
      setTrainingImages(prev => [...prev, ...imageFiles]);
    }
  };

  const removeImage = (index: number) => {
    setTrainingImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (trainingImages.length === 0) {
      setError('Please select at least one training image');
      return;
    }

    setIsTraining(true);
    setError(null);
    setTrainingStatus('Starting training...');

    try {
      const formData = new FormData();
      trainingImages.forEach(image => {
        formData.append('training_images', image);
      });
      formData.append('epochs', epochs.toString());

      const response = await axios.post('/api/stegnogan/train', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setTrainingStatus('Training completed successfully!');
      await fetchModelStatus(); // Refresh model status
      
    } catch (err: any) {
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Training failed. Please try again.');
      }
      setTrainingStatus(null);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="card">
      <h2>🤖 StegnoGAN++ Training</h2>
      
      {/* Model Status */}
      {modelStatus && (
        <div className="card" style={{ marginBottom: '2rem', background: modelStatus.is_trained ? '#d4edda' : '#fff3cd' }}>
          <h3>📊 Model Status</h3>
          <p><strong>Status:</strong> {modelStatus.status}</p>
          <p><strong>Trained:</strong> {modelStatus.is_trained ? 'Yes' : 'No'}</p>
          <p><strong>Device:</strong> {modelStatus.device}</p>
          {!modelStatus.is_trained && (
            <p style={{ color: '#856404', marginTop: '1rem' }}>
              ⚠️ Model needs training before it can be used for embedding and extraction.
            </p>
          )}
        </div>
      )}

      <form onSubmit={handleTraining}>
        {/* Training Images Upload */}
        <div className="form-group">
          <label>Training Images</label>
          <div 
            className="file-upload"
            onDrop={handleFileDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="file-upload-text">
              Click to select or drag & drop training images
            </div>
            <div className="file-upload-subtext">
              Supported: PNG, JPG, JPEG (at least 10 images recommended)
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>
          
          {/* Display selected images */}
          {trainingImages.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <h4>Selected Images ({trainingImages.length}):</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {trainingImages.map((image, index) => (
                  <div key={index} style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    background: '#f8f9fa', 
                    padding: '0.5rem', 
                    borderRadius: '4px',
                    border: '1px solid #dee2e6'
                  }}>
                    <span style={{ marginRight: '0.5rem', fontSize: '0.9rem' }}>
                      {image.name}
                    </span>
                    <button
                      type="button"
                      onClick={() => removeImage(index)}
                      style={{
                        background: '#dc3545',
                        color: 'white',
                        border: 'none',
                        borderRadius: '50%',
                        width: '20px',
                        height: '20px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Training Parameters */}
        <div className="form-group">
          <label>Training Epochs</label>
          <input
            type="number"
            className="form-control"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value))}
            min="10"
            max="1000"
            step="10"
          />
          <div className="timestamp-info">
            Number of training epochs. More epochs = better quality but longer training time.
            Recommended: 100-500 epochs.
          </div>
        </div>

        {/* Submit Button */}
        <button type="submit" className="btn btn-block" disabled={isTraining || trainingImages.length === 0}>
          {isTraining ? 'Training Model...' : '🚀 Start Training'}
        </button>
      </form>

      {/* Training Status */}
      {isTraining && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Training StegnoGAN++ model...</p>
          <p style={{ fontSize: '0.9rem', color: '#666' }}>
            This may take several minutes depending on the number of images and epochs.
          </p>
        </div>
      )}

      {/* Training Status Message */}
      {trainingStatus && (
        <div className="result-section">
          <h3>✅ Training Complete!</h3>
          <p>{trainingStatus}</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Instructions */}
      <div className="card" style={{ marginTop: '2rem' }}>
        <h3>📋 How to Train StegnoGAN++</h3>
        <ol style={{ paddingLeft: '1.5rem', lineHeight: '1.6' }}>
          <li><strong>Prepare training images:</strong> Collect a diverse set of images (at least 10, preferably 50+).</li>
          <li><strong>Upload images:</strong> Drag and drop or select multiple images for training.</li>
          <li><strong>Set epochs:</strong> Choose the number of training epochs (100-500 recommended).</li>
          <li><strong>Start training:</strong> Click the training button and wait for completion.</li>
          <li><strong>Use the model:</strong> Once trained, the model can be used for embedding and extraction.</li>
        </ol>
        
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#e3f2fd', borderRadius: '8px' }}>
          <strong>💡 Tips:</strong>
          <ul style={{ marginTop: '0.5rem', paddingLeft: '1.5rem' }}>
            <li>Use diverse images for better generalization</li>
            <li>Higher epochs improve quality but increase training time</li>
            <li>The model will automatically fall back to LSB if not trained</li>
            <li>Training progress will be shown in the status section</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default StegnoGANPage;






