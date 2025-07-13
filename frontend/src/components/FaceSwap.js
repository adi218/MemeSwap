import React, { useState, useEffect } from 'react';

const FaceSwap = ({ selectedGif, uploadedFace }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
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

  const handleFaceSwap = async () => {
    if (!uploadedFace || !selectedGif) {
      setError('Please complete the previous steps first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('source_image', uploadedFace.file);

      const response = await fetch(`http://127.0.0.1:8000/api/face/swap-face-on-gif?gif_url=${encodeURIComponent(selectedGif.url)}&model=${selectedModel}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Face swap failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Model Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Face Detection Model for Face Swap
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
          </div>
        )}
      </div>

      {/* Summary of Selected Items */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Selected GIF */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-medium text-gray-800 mb-3">Selected GIF</h3>
          {selectedGif ? (
            <div className="space-y-3">
              <img
                src={selectedGif.url}
                alt={selectedGif.title || 'Selected GIF'}
                className="w-full h-48 object-cover rounded-lg"
              />
              <p className="text-sm text-gray-600 truncate">
                {selectedGif.title || 'Selected GIF'}
              </p>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <p>No GIF selected</p>
            </div>
          )}
        </div>

        {/* Uploaded Face */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-medium text-gray-800 mb-3">Uploaded Face</h3>
          {uploadedFace ? (
            <div className="space-y-3">
              <img
                src={uploadedFace.imageUrl}
                alt="Uploaded face"
                className="w-full h-48 object-cover rounded-lg"
              />
              <p className="text-sm text-gray-600">
                {uploadedFace.faces.length} face(s) detected
              </p>
              {uploadedFace.model && (
                <p className="text-xs text-gray-500">
                  Detection model: {uploadedFace.model}
                </p>
              )}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <p>No face uploaded</p>
            </div>
          )}
        </div>
      </div>

      {/* Action Button */}
      <div className="text-center mb-8">
        <button
          onClick={handleFaceSwap}
          disabled={!uploadedFace || !selectedGif || isLoading}
          className={`px-8 py-4 rounded-lg font-semibold text-lg transition-colors ${
            !uploadedFace || !selectedGif || isLoading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-indigo-500 text-white hover:bg-indigo-600 shadow-lg'
          }`}
        >
          {isLoading ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Creating your meme with {availableModels[selectedModel]?.name || selectedModel}...</span>
            </div>
          ) : (
            '🎭 Generate Face Swap!'
          )}
        </button>
        
        {(!uploadedFace || !selectedGif) && (
          <p className="mt-2 text-sm text-gray-500">
            Please complete the previous steps to enable face swap
          </p>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg max-w-lg mx-auto">
          <p className="font-semibold">Error:</p>
          <p>{error}</p>
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="space-y-6">
          <div className="text-center">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Your Meme is Ready! 🎉</h3>
            <p className="text-gray-600">Download your face-swapped GIF below</p>
            {result.model_used && (
              <p className="text-sm text-gray-500 mt-1">
                Generated using {availableModels[result.model_used]?.name || result.model_used} model
              </p>
            )}
          </div>
          
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="text-center">
              <img
                src={`data:image/gif;base64,${result.output_data}`}
                alt="Face swap result"
                className="max-w-full h-64 object-contain mx-auto rounded-lg shadow-lg"
              />
              <div className="mt-6">
                <a
                  href={`data:image/gif;base64,${result.output_data}`}
                  download="meme_face_swap.gif"
                  className="inline-block px-8 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors font-medium"
                >
                  📥 Download Meme
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FaceSwap; 