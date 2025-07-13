'use client';

import { useState, useCallback, useEffect } from 'react';

export default function FaceUpload({ onFaceUpload }) {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedModel, setSelectedModel] = useState('yolo');
  const [availableModels, setAvailableModels] = useState({});
  const [activeTab, setActiveTab] = useState('upload'); // 'upload', 'results', 'debug'
  const [debugMode, setDebugMode] = useState(false);

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

      // Upload and detect faces with YOLO model
      const response = await fetch(`http://127.0.0.1:8000/api/face/detect-faces?model=yolo`, {
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
  }, [onFaceUpload]);

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

  const renderBoundingBoxOverlay = (face, imageWidth, imageHeight) => {
    if (!face.bbox) return null;

    const { x, y, width, height } = face.bbox;
    const scaleX = 100 / imageWidth;
    const scaleY = 100 / imageHeight;

    return (
      <div
        className="absolute border-2 border-red-500 bg-red-500 bg-opacity-20"
        style={{
          left: `${x * scaleX}%`,
          top: `${y * scaleY}%`,
          width: `${width * scaleX}%`,
          height: `${height * scaleY}%`,
        }}
      >
        <div className="absolute -top-6 left-0 bg-red-500 text-white text-xs px-1 rounded">
          {face.confidence ? `${(face.confidence * 100).toFixed(1)}%` : 'Face'}
        </div>
      </div>
    );
  };

  const renderDebugInfo = () => {
    if (!detectionResult || !detectionResult.faces) return null;

    return (
      <div className="space-y-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800 mb-2">🔍 Debug Information</h4>
          <div className="text-sm text-yellow-700 space-y-2">
            <p><strong>Model Used:</strong> {detectionResult.model_used}</p>
            <p><strong>Image Dimensions:</strong> {detectionResult.image_shape?.width} × {detectionResult.image_shape?.height}px</p>
            <p><strong>Faces Detected:</strong> {detectionResult.faces_found}</p>
            <p><strong>Processing Time:</strong> {detectionResult.processing_time || 'N/A'}ms</p>
          </div>
        </div>

        {/* Bounding Box Analysis */}
        <div>
          <h4 className="font-medium text-gray-800 mb-3">📐 Bounding Box Analysis</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {detectionResult.faces.map((face, index) => (
              <div key={index} className="bg-white border border-gray-200 rounded-lg p-4">
                <h5 className="font-medium text-gray-800 mb-2">Face {index + 1}</h5>
                <div className="space-y-2 text-sm">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-gray-600">X Position:</span>
                      <span className="font-mono ml-2">{face.bbox.x}px</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Y Position:</span>
                      <span className="font-mono ml-2">{face.bbox.y}px</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Width:</span>
                      <span className="font-mono ml-2">{face.bbox.width}px</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Height:</span>
                      <span className="font-mono ml-2">{face.bbox.height}px</span>
                    </div>
                  </div>
                  
                  <div className="pt-2 border-t border-gray-200">
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <span className="text-gray-600">Coverage %:</span>
                        <span className="font-mono ml-2">
                          {((face.bbox.width * face.bbox.height) / (detectionResult.image_shape.width * detectionResult.image_shape.height) * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Aspect Ratio:</span>
                        <span className="font-mono ml-2">
                          {(face.bbox.width / face.bbox.height).toFixed(2)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {face.pose && face.pose.success && (
                    <div className="pt-2 border-t border-gray-200">
                      <h6 className="font-medium text-gray-700 mb-1">Pose Analysis:</h6>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-gray-600">Yaw:</span>
                          <span className="font-mono ml-1">{face.pose.yaw.toFixed(1)}°</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Pitch:</span>
                          <span className="font-mono ml-1">{face.pose.pitch.toFixed(1)}°</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Roll:</span>
                          <span className="font-mono ml-1">{face.pose.roll.toFixed(1)}°</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Debug Info */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-800 mb-2">💡 Debug Information</h4>
          <p className="text-sm text-blue-700 mb-3">
            Check the <code>debug_faces</code> directory in the backend for saved cropped faces from both source images and GIF frames.
          </p>
          <div className="text-xs text-blue-600 space-y-1">
            <p>• Source faces are saved as <code>source_face_*.jpg</code></p>
            <p>• GIF faces are saved as <code>gif_face_*.jpg</code></p>
            <p>• These help debug face detection and swapping issues</p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Tab Navigation */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'upload'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              📤 Upload
            </button>
            {detectionResult && (
              <>
                <button
                  onClick={() => setActiveTab('results')}
                  className={`py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === 'results'
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  🎯 Results
                </button>
                <button
                  onClick={() => setActiveTab('debug')}
                  className={`py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === 'debug'
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  🔍 Debug
                </button>
              </>
            )}
          </nav>
        </div>
      </div>

      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div className="space-y-6">
          {/* Model Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Face Detection Model
            </label>
            <select
              value={selectedModel}
              onChange={handleModelChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              disabled
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
              <p className="mt-2 text-gray-600">Detecting faces with YOLO...</p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-800 rounded-lg max-w-lg mx-auto">
              {error}
            </div>
          )}
        </div>
      )}

      {/* Results Tab */}
      {activeTab === 'results' && detectionResult && (
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
                <div className="relative">
                  <img
                    src={uploadedImage}
                    alt="Uploaded"
                    className="w-full h-auto max-h-96 object-contain rounded-lg border border-gray-200"
                    style={{
                      maxWidth: '100%',
                      height: 'auto'
                    }}
                  />
                  {detectionResult.image_shape && (
                    <div className="mt-2 text-xs text-gray-500">
                      Image dimensions: {detectionResult.image_shape.width} × {detectionResult.image_shape.height}px
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Annotated Image */}
            <div>
              <h4 className="font-medium text-gray-800 mb-2">
                Detected Faces (YOLO)
              </h4>
              {detectionResult.annotated_image && (
                <div className="relative">
                  <img
                    src={detectionResult.annotated_image}
                    alt="Faces detected"
                    className="w-full h-auto max-h-96 object-contain rounded-lg border border-gray-200"
                    style={{
                      maxWidth: '100%',
                      height: 'auto'
                    }}
                  />
                  {detectionResult.faces_found > 0 && (
                    <div className="mt-2 text-xs text-gray-500">
                      {detectionResult.faces_found} face(s) detected with pose analysis
                    </div>
                  )}
                </div>
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
                      {face.pose && face.pose.success && (
                        <div className="mt-2 pt-2 border-t border-gray-200">
                          <p className="font-medium text-xs text-gray-700 mb-1">Pose Analysis:</p>
                          <div className="space-y-1 text-xs">
                            <p>Yaw: {face.pose.yaw.toFixed(1)}° (Left-Right)</p>
                            <p>Pitch: {face.pose.pitch.toFixed(1)}° (Up-Down)</p>
                            <p>Roll: {face.pose.roll.toFixed(1)}° (Head Tilt)</p>
                            <p>Pose Confidence: {(face.pose.confidence * 100).toFixed(1)}%</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Debug Tab */}
      {activeTab === 'debug' && detectionResult && (
        renderDebugInfo()
      )}
    </div>
  );
} 