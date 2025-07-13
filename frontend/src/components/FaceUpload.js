'use client';

import { useState, useCallback, useEffect } from 'react';

export default function FaceUpload({ onFaceUpload }) {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedModel, setSelectedModel] = useState('mediapipe_enhanced');
  const [availableModels, setAvailableModels] = useState({});

  // Fetch available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/face/available-models');
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.models);
      }
    } catch (err) {
      console.error('Failed to fetch available models:', err);
    }
  };

  const handleFileUpload = useCallback(async (file) => {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    setLoading(true);
    setError('');
    setDetectionResult(null);

    try {
      // Create FormData
      const formData = new FormData();
      formData.append('file', file);

      // Upload and detect faces with selected model
      const response = await fetch(`http://127.0.0.1:8000/api/face/detect-faces?model=${selectedModel}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to detect faces');
      }

      const result = await response.json();
      setDetectionResult(result);
      setUploadedImage(URL.createObjectURL(file));
      
      // Call the callback with face data if faces were found
      if (result.faces_found > 0 && onFaceUpload) {
        onFaceUpload({
          file: file,
          faces: result.faces,
          imageUrl: URL.createObjectURL(file),
          model: result.model_used
        });
      }
    } catch (err) {
      setError('Error detecting faces: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedModel, onFaceUpload]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileInput = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Model Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Face Detection Model
        </label>
        <select
          value={selectedModel}
          onChange={handleModelChange}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {Object.entries(availableModels).map(([key, model]) => (
            <option key={key} value={key} disabled={model.available === false}>
              {model.name} {model.available === false ? '(Not Available)' : ''}
            </option>
          ))}
        </select>
        {availableModels[selectedModel] && (
          <div className="mt-2 text-sm text-gray-600">
            <p className="font-medium">{availableModels[selectedModel].description}</p>
            <p className="text-xs mt-1">
              <strong>Best for:</strong> {availableModels[selectedModel].best_for}
            </p>
            {availableModels[selectedModel].features && (
              <p className="text-xs mt-1">
                <strong>Features:</strong> {availableModels[selectedModel].features.join(', ')}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Upload Area */}
      <div className="mb-8">
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragOver
              ? 'border-indigo-500 bg-indigo-50'
              : 'border-gray-300 hover:border-indigo-400'
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="space-y-4">
            <div className="text-4xl">📷</div>
            <h3 className="text-lg font-medium text-gray-700">
              Upload a face image
            </h3>
            <p className="text-gray-500">
              Drag and drop an image here, or click to select
            </p>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="inline-block px-6 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 cursor-pointer"
            >
              Choose File
            </label>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <p className="mt-2 text-gray-600">Detecting faces with {availableModels[selectedModel]?.name || selectedModel}...</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-800 rounded-lg max-w-lg mx-auto">
          {error}
        </div>
      )}

      {/* Results */}
      {detectionResult && (
        <div className="space-y-6">
          <div className="text-center">
            <h3 className="text-xl font-semibold text-gray-800 mb-2">
              Face Detection Results
            </h3>
            <p className="text-gray-600">
              Found {detectionResult.faces_found} face(s) using {detectionResult.model_used} model
            </p>
            {detectionResult.faces_found > 0 && (
              <div className="mt-4 p-4 bg-green-100 border border-green-400 text-green-800 rounded-lg max-w-md mx-auto">
                <p className="font-medium">✅ Face detected! Ready for face swap.</p>
              </div>
            )}
          </div>

          {/* Image Display */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Original Image */}
            <div>
              <h4 className="font-medium text-gray-800 mb-2">Original Image</h4>
              {uploadedImage && (
                <img
                  src={uploadedImage}
                  alt="Uploaded"
                  className="w-full rounded-lg border border-gray-200"
                />
              )}
            </div>

            {/* Annotated Image */}
            <div>
              <h4 className="font-medium text-gray-800 mb-2">
                Detected Faces ({detectionResult.model_used})
              </h4>
              {detectionResult.annotated_image && (
                <img
                  src={detectionResult.annotated_image}
                  alt="Faces detected"
                  className="w-full rounded-lg border border-gray-200"
                />
              )}
            </div>
          </div>

          {/* Face Details */}
          {detectionResult.faces && detectionResult.faces.length > 0 && (
            <div>
              <h4 className="font-medium text-gray-800 mb-4">Face Details</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {detectionResult.faces.map((face, index) => (
                  <div
                    key={index}
                    className="p-4 border border-gray-200 rounded-lg bg-white"
                  >
                    <h5 className="font-medium text-gray-800 mb-2">
                      Face {index + 1}
                    </h5>
                    <div className="space-y-1 text-sm text-gray-600">
                      <p>
                        Confidence: {(face.confidence * 100).toFixed(1)}%
                      </p>
                      <p>
                        Position: ({face.bbox.x}, {face.bbox.y})
                      </p>
                      <p>
                        Size: {face.bbox.width} × {face.bbox.height}
                      </p>
                      {face.model && (
                        <p>
                          Model: {face.model}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
} 