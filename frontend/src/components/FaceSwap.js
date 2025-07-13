import React, { useState, useEffect } from 'react';

const FaceSwap = ({ selectedGif, uploadedFace }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.3); // Default from backend config
  const [dropThreshold, setDropThreshold] = useState(3); // Default from backend config

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

      const response = await fetch(
        `http://127.0.0.1:8000/api/face/swap-face-on-gif?gif_url=${encodeURIComponent(selectedGif.url)}&confidence_threshold=${confidenceThreshold}&drop_threshold=${dropThreshold}`, 
        {
          method: 'POST',
          body: formData,
        }
      );

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

  return (
    <div className="max-w-4xl mx-auto">
      {/* Model Info */}
      <div className="mb-6">
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h3 className="font-medium text-purple-800 mb-2">🤖 MediaPipe Enhanced Face Detection</h3>
          <p className="text-sm text-purple-700">
            Using MediaPipe enhanced detection for precise face detection and swapping. Cropped faces are saved to <code>debug_faces</code> directory for debugging.
          </p>
        </div>
      </div>

      {/* Configuration Options */}
      <div className="mb-6">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-medium text-blue-800 mb-3">⚙️ Detection Settings</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-blue-700 mb-2">
                Confidence Threshold: {confidenceThreshold}
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-blue-600 mt-1">
                <span>0.1 (More Sensitive)</span>
                <span>0.9 (Less Sensitive)</span>
              </div>
              <p className="text-xs text-blue-600 mt-1">
                Lower values detect faces more easily but may include false positives
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-blue-700 mb-2">
                Drop Threshold: {dropThreshold} frames
              </label>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={dropThreshold}
                onChange={(e) => setDropThreshold(parseInt(e.target.value))}
                className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-blue-600 mt-1">
                <span>1 frame</span>
                <span>10 frames</span>
              </div>
              <p className="text-xs text-blue-600 mt-1">
                Number of consecutive frames without face detection before dropping overlay
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Summary of Selected Items */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Selected GIF */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-medium text-gray-800 mb-3">Selected GIF</h3>
          {selectedGif ? (
            <div className="space-y-3">
              <div className="relative">
                <img
                  src={selectedGif.url}
                  alt={selectedGif.title || 'Selected GIF'}
                  className="w-full h-auto max-h-64 object-contain rounded-lg"
                  style={{
                    maxWidth: '100%',
                    height: 'auto'
                  }}
                />
                {selectedGif.dimensions && (
                  <div className="mt-2 text-xs text-gray-500">
                    GIF dimensions: {selectedGif.dimensions.width} × {selectedGif.dimensions.height}px
                  </div>
                )}
              </div>
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
              <div className="relative">
                <img
                  src={uploadedFace.imageUrl}
                  alt="Uploaded face"
                  className="w-full h-auto max-h-64 object-contain rounded-lg"
                  style={{
                    maxWidth: '100%',
                    height: 'auto'
                  }}
                />
                {uploadedFace.faces && uploadedFace.faces.length > 0 && (
                  <div className="mt-2 text-xs text-gray-500">
                    {uploadedFace.faces.length} face(s) detected
                    {uploadedFace.faces[0].pose && uploadedFace.faces[0].pose.success && (
                      <span className="ml-2">
                        • Pose: Yaw {uploadedFace.faces[0].pose.yaw.toFixed(1)}°, 
                        Pitch {uploadedFace.faces[0].pose.pitch.toFixed(1)}°, 
                        Roll {uploadedFace.faces[0].pose.roll.toFixed(1)}°
                      </span>
                    )}
                  </div>
                )}
              </div>
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
              <span>Creating your meme with MediaPipe...</span>
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
        <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-800 rounded-lg">
          <h4 className="font-medium mb-2">Error</h4>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="space-y-6">
          <div className="text-center">
            <h3 className="text-xl font-semibold text-gray-800 mb-2">
              🎉 Face Swap Complete!
            </h3>
            <p className="text-gray-600">
              Your face-swapped GIF has been generated successfully
            </p>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h4 className="font-medium text-gray-800 mb-4">Result</h4>
            <div className="space-y-4">
              {/* Result GIF */}
              <div className="text-center">
                <img
                  src={`data:image/gif;base64,${result.output_data}`}
                  alt="Face-swapped GIF"
                  className="w-full h-auto max-h-96 object-contain rounded-lg border border-gray-200"
                  style={{
                    maxWidth: '100%',
                    height: 'auto'
                  }}
                />
              </div>

              {/* Download Link */}
              <div className="text-center">
                <a
                  href={`data:image/gif;base64,${result.output_data}`}
                  download="face_swapped.gif"
                  className="inline-block px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                >
                  📥 Download GIF
                </a>
              </div>

              {/* Debug Info */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h5 className="font-medium text-blue-800 mb-2">🔍 Debug Information</h5>
                <div className="text-sm text-blue-700 space-y-1">
                  <p><strong>Model Used:</strong> {result.model_used}</p>
                  <p><strong>Confidence Threshold:</strong> {result.confidence_threshold || confidenceThreshold}</p>
                  <p><strong>Drop Threshold:</strong> {result.drop_threshold || dropThreshold} frames</p>
                  <p><strong>Source Image:</strong> {result.source_image}</p>
                  <p><strong>Target GIF:</strong> {result.target_gif}</p>
                  <p><strong>Output Path:</strong> {result.output_path}</p>
                  <p className="text-xs mt-2">
                    💡 Check the <code>debug_faces</code> directory for saved cropped faces
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FaceSwap; 