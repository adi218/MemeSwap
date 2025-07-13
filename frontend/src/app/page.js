'use client';

import { useState } from 'react';
import GifSearch from '../components/GifSearch';
import FaceUpload from '../components/FaceUpload';
import FaceSwap from '../components/FaceSwap';

export default function Home() {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedGif, setSelectedGif] = useState(null);
  const [uploadedFace, setUploadedFace] = useState(null);

  const handleGifSelect = (gif) => {
    setSelectedGif(gif);
    setCurrentStep(3);
  };

  const handleFaceUpload = (faceData) => {
    setUploadedFace(faceData);
    setCurrentStep(4);
  };

  const resetFlow = () => {
    setCurrentStep(1);
    setSelectedGif(null);
    setUploadedFace(null);
  };

  const steps = [
    {
      id: 1,
      title: 'Search GIFs',
      description: 'Find the perfect GIF for your meme',
      icon: '🔍'
    },
    {
      id: 2,
      title: 'Choose GIF',
      description: 'Select your target GIF',
      icon: '🎯'
    },
    {
      id: 3,
      title: 'Upload Face',
      description: 'Upload a face image with model selection',
      icon: '📷'
    },
    {
      id: 4,
      title: 'Generate Meme',
      description: 'Create your face-swapped meme',
      icon: '🎭'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            MemeSwap 🎭
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Create hilarious memes by swapping faces on GIFs using advanced AI detection models
          </p>
        </div>

        {/* Step Progress */}
        <div className="mb-12">
          <div className="flex justify-center">
            <div className="flex space-x-4 md:space-x-8">
              {steps.map((step, index) => (
                <div key={step.id} className="flex items-center">
                  <div
                    className={`flex items-center justify-center w-12 h-12 rounded-full border-2 text-lg font-semibold transition-all ${
                      currentStep >= step.id
                        ? 'bg-indigo-500 text-white border-indigo-500'
                        : 'bg-white text-gray-400 border-gray-300'
                    }`}
                  >
                    {step.icon}
                  </div>
                  {index < steps.length - 1 && (
                    <div
                      className={`w-8 h-1 md:w-16 transition-all ${
                        currentStep > step.id ? 'bg-indigo-500' : 'bg-gray-300'
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Step Labels */}
          <div className="flex justify-center mt-4">
            <div className="flex space-x-4 md:space-x-8">
              {steps.map((step) => (
                <div
                  key={step.id}
                  className={`text-center transition-all ${
                    currentStep >= step.id ? 'text-indigo-600' : 'text-gray-400'
                  }`}
                >
                  <div className="text-sm font-medium">{step.title}</div>
                  <div className="text-xs">{step.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Model Selection Info */}
        <div className="mb-8 max-w-4xl mx-auto">
          <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">
              🤖 AI Model Selection
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="p-3 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-800">MediaPipe Standard</h4>
                <p className="text-blue-600 text-xs mt-1">Fast detection with basic bounding boxes</p>
              </div>
              <div className="p-3 bg-green-50 rounded-lg">
                <h4 className="font-medium text-green-800">MediaPipe Enhanced</h4>
                <p className="text-green-600 text-xs mt-1">468 landmarks including hair and ears</p>
              </div>
              <div className="p-3 bg-purple-50 rounded-lg">
                <h4 className="font-medium text-purple-800">YOLO Detection</h4>
                <p className="text-purple-600 text-xs mt-1">Robust detection for various angles</p>
              </div>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="max-w-6xl mx-auto">
          {currentStep === 1 && (
            <div className="space-y-8">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">
                  Step 1: Search for GIFs
                </h2>
                <p className="text-gray-600">
                  Search for the perfect GIF to create your meme
                </p>
              </div>
              <GifSearch onGifSelect={handleGifSelect} />
            </div>
          )}

          {currentStep === 2 && (
            <div className="text-center py-12">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                Step 2: Choose a GIF
              </h2>
              <p className="text-gray-600 mb-8">
                Please go back to step 1 and select a GIF first
              </p>
              <button
                onClick={() => setCurrentStep(1)}
                className="px-6 py-3 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors"
              >
                ← Back to Search
              </button>
            </div>
          )}

          {currentStep === 3 && (
            <div className="space-y-8">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">
                  Step 3: Upload Your Face
                </h2>
                <p className="text-gray-600">
                  Upload a clear face image and choose your detection model
                </p>
              </div>
              
              {/* Selected GIF Preview */}
              {selectedGif && (
                <div className="max-w-md mx-auto mb-8">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3 text-center">
                    Target GIF
                  </h3>
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <img
                      src={selectedGif.url}
                      alt={selectedGif.title || 'Selected GIF'}
                      className="w-full h-48 object-cover rounded-lg"
                    />
                    <p className="text-sm text-gray-600 mt-2 truncate">
                      {selectedGif.title || 'Selected GIF'}
                    </p>
                  </div>
                </div>
              )}
              
              <FaceUpload onFaceUpload={handleFaceUpload} />
            </div>
          )}

          {currentStep === 4 && (
            <div className="space-y-8">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">
                  Step 4: Generate Your Meme
                </h2>
                <p className="text-gray-600">
                  Create your face-swapped meme with your chosen AI model
                </p>
              </div>
              <FaceSwap selectedGif={selectedGif} uploadedFace={uploadedFace} />
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="text-center mt-12">
          <button
            onClick={resetFlow}
            className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
          >
            🔄 Start Over
          </button>
        </div>
      </div>
    </div>
  );
}
