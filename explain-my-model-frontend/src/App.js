import React, { useState, useEffect } from 'react';
import { Upload, X, Loader, Eye, Layers, Image, Brain, Sparkles, Info } from 'lucide-react';

export default function ExplainMyModelDashboard() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [selectedModel, setSelectedModel] = useState('resnet50');
  const [selectedLayer, setSelectedLayer] = useState('layer4');
  const [availableModels, setAvailableModels] = useState([]);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [error, setError] = useState(null);

  // Fetch available models on mount
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      setAvailableModels(data.models);
    } catch (err) {
      console.error('Error fetching models:', err);
      // Fallback to default models
      setAvailableModels([
        {
          id: 'resnet50',
          name: 'ResNet-50',
          layers: [
            { id: 'layer1', name: 'Layer 1' },
            { id: 'layer2', name: 'Layer 2' },
            { id: 'layer3', name: 'Layer 3' },
            { id: 'layer4', name: 'Layer 4' }
          ]
        }
      ]);
    }
  };

  const getCurrentModelLayers = () => {
    const model = availableModels.find(m => m.id === selectedModel);
    return model?.layers || [];
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    setImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
    setResults(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const analyzeImage = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', image);
    formData.append('model_name', selectedModel);
    formData.append('layer', selectedLayer);
    
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Failed to analyze image. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
  };

  const handleModelChange = (modelId) => {
    setSelectedModel(modelId);
    const model = availableModels.find(m => m.id === modelId);
    if (model && model.layers.length > 0) {
      setSelectedLayer(model.layers[model.layers.length - 1].id);
    }
    setResults(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 md:mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="relative">
              <Eye className="w-10 h-10 md:w-12 md:h-12 text-purple-400" />
              <Sparkles className="w-5 h-5 text-yellow-400 absolute -top-1 -right-1 animate-pulse" />
            </div>
            <h1 className="text-3xl md:text-5xl font-bold text-white">Explain My Model</h1>
          </div>
          <p className="text-gray-300 text-sm md:text-lg max-w-2xl mx-auto">
            Visualize how convolutional neural networks see and understand images through Grad-CAM and activation maps
          </p>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mb-6 bg-red-500/20 border border-red-500/50 rounded-xl p-4 backdrop-blur-lg">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-200 font-medium">Analysis Failed</p>
                <p className="text-red-300/80 text-sm mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
          {/* Left Column - Upload & Controls */}
          <div className="space-y-6">
            {/* Upload Area */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 md:p-6 border border-white/20 shadow-2xl">
              <h2 className="text-lg md:text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Image
              </h2>
              
              {!imagePreview ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  className="border-2 border-dashed border-purple-400 rounded-xl p-8 md:p-12 text-center cursor-pointer hover:border-purple-300 hover:bg-white/5 transition-all"
                  onClick={() => document.getElementById('file-input').click()}
                >
                  <Image className="w-12 h-12 md:w-16 md:h-16 mx-auto mb-4 text-purple-400" />
                  <p className="text-gray-300 mb-2 text-sm md:text-base">Drag and drop an image here</p>
                  <p className="text-gray-500 text-xs md:text-sm">or click to browse</p>
                  <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="relative group">
                  <img
                    src={imagePreview}
                    alt="Uploaded"
                    className="w-full h-48 md:h-64 object-cover rounded-xl shadow-lg"
                  />
                  <button
                    onClick={clearImage}
                    className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full transition-all opacity-0 group-hover:opacity-100 shadow-lg"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>

            {/* Controls */}
            {imagePreview && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 md:p-6 border border-white/20 shadow-2xl">
                <h2 className="text-lg md:text-xl font-semibold text-white mb-4 flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Model Configuration
                </h2>
                
                <div className="space-y-4">
                  {/* Model Selection */}
                  <div>
                    <label className="block text-gray-300 mb-2 text-sm font-medium">
                      Select Model
                    </label>
                    <select
                      value={selectedModel}
                      onChange={(e) => handleModelChange(e.target.value)}
                      className="w-full bg-white/5 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-400/20 transition-all"
                    >
                      {availableModels.map(model => (
                        <option key={model.id} value={model.id} className="bg-slate-800">
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Layer Selection */}
                  <div>
                    <label className="block text-gray-300 mb-2 text-sm font-medium flex items-center gap-2">
                      <Layers className="w-4 h-4" />
                      Target Layer
                    </label>
                    <select
                      value={selectedLayer}
                      onChange={(e) => setSelectedLayer(e.target.value)}
                      className="w-full bg-white/5 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-400/20 transition-all"
                    >
                      {getCurrentModelLayers().map(layer => (
                        <option key={layer.id} value={layer.id} className="bg-slate-800">
                          {layer.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Heatmap Opacity */}
                  <div>
                    <label className="block text-gray-300 mb-2 text-sm font-medium">
                      Heatmap Overlay: {Math.round(heatmapOpacity * 100)}%
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={heatmapOpacity}
                      onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                      className="w-full accent-purple-500"
                    />
                  </div>

                  {/* Analyze Button */}
                  <button
                    onClick={analyzeImage}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-3 md:py-4 rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-purple-500/50 disabled:shadow-none transform hover:scale-[1.02] disabled:scale-100"
                  >
                    {loading ? (
                      <>
                        <Loader className="w-5 h-5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Eye className="w-5 h-5" />
                        Analyze Image
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {results && (
              <>
                {/* Predictions */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 md:p-6 border border-white/20 shadow-2xl">
                  <h2 className="text-lg md:text-xl font-semibold text-white mb-4">
                    Top Predictions
                  </h2>
                  <div className="space-y-3">
                    {results.predictions.map((pred, idx) => (
                      <div key={idx} className="bg-white/5 rounded-lg p-3 md:p-4 hover:bg-white/10 transition-colors">
                        <div className="flex justify-between items-center mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-purple-400 font-bold text-sm">#{idx + 1}</span>
                            <span className="text-white font-medium text-sm md:text-base">{pred.class}</span>
                          </div>
                          <span className="text-purple-400 font-semibold text-sm md:text-base">
                            {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${pred.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Grad-CAM Visualization */}
                {results.gradcam && (
                  <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 md:p-6 border border-white/20 shadow-2xl">
                    <h2 className="text-lg md:text-xl font-semibold text-white mb-4">
                      Grad-CAM Heatmap
                    </h2>
                    <div className="relative rounded-xl overflow-hidden shadow-xl">
                      <img
                        src={results.gradcam}
                        alt="Grad-CAM"
                        className="w-full h-auto"
                      />
                    </div>
                    <p className="text-gray-400 text-xs md:text-sm mt-3 leading-relaxed">
                      Warmer colors (red/yellow) highlight regions that most influenced the model's top prediction
                    </p>
                  </div>
                )}

                {/* Activation Maps */}
                {results.activations && (
                  <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 md:p-6 border border-white/20 shadow-2xl">
                    <h2 className="text-lg md:text-xl font-semibold text-white mb-4">
                      Feature Activation Maps
                    </h2>
                    <div className="rounded-xl overflow-hidden shadow-xl">
                      <img
                        src={results.activations}
                        alt="Activations"
                        className="w-full h-auto"
                      />
                    </div>
                    <p className="text-gray-400 text-xs md:text-sm mt-3 leading-relaxed">
                      16 feature channels showing what patterns the network detects at this layer
                    </p>
                  </div>
                )}
              </>
            )}

            {!results && !loading && imagePreview && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 md:p-12 border border-white/20 text-center shadow-2xl">
                <Eye className="w-12 h-12 md:w-16 md:h-16 mx-auto mb-4 text-gray-500" />
                <p className="text-gray-400 text-sm md:text-base">Click "Analyze Image" to see visualizations</p>
              </div>
            )}

            {!imagePreview && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 md:p-12 border border-white/20 text-center shadow-2xl">
                <Image className="w-12 h-12 md:w-16 md:h-16 mx-auto mb-4 text-gray-500" />
                <p className="text-gray-400 text-sm md:text-base">Upload an image to get started</p>
              </div>
            )}
          </div>
        </div>

        {/* Info Footer */}
        <div className="mt-8 md:mt-12 bg-white/5 backdrop-blur-lg rounded-2xl p-4 md:p-6 border border-white/20 shadow-2xl">
          <h3 className="text-base md:text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <Info className="w-5 h-5" />
            How It Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs md:text-sm text-gray-300">
            <div className="bg-white/5 rounded-lg p-4">
              <p className="font-medium text-purple-400 mb-2">1. Choose & Upload</p>
              <p>Select a pre-trained model and upload any image for analysis</p>
            </div>
            <div className="bg-white/5 rounded-lg p-4">
              <p className="font-medium text-purple-400 mb-2">2. Neural Processing</p>
              <p>The CNN processes the image through multiple layers, extracting features at each stage</p>
            </div>
            <div className="bg-white/5 rounded-lg p-4">
              <p className="font-medium text-purple-400 mb-2">3. Explainable AI</p>
              <p>Grad-CAM reveals which image regions drove the model's decision-making process</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}