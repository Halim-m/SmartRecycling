import React, { useState, useEffect } from 'react';
import { Recycle, Github, AlertTriangle, Camera, Upload } from 'lucide-react';
import ImageUploader from './components/ImageUploader';
import ClassificationResult from './components/ClassificationResult';
import RecyclingGuide from './components/RecyclingGuide';
import RealtimeCamera from './components/RealtimeCamera';
import { classifyImage, getHealth } from './services/api';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' veya 'realtime'

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const health = await getHealth();
      setApiStatus(health.model_loaded ? 'ready' : 'no_model');
    } catch (err) {
      setApiStatus('offline');
    }
  };

  const handleUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await classifyImage(file);
      setResult(response);
    } catch (err) {
      console.error('Classification error:', err);
      setError(
        err.response?.data?.detail ||
        'Sınıflandırma sırasında bir hata oluştu. Lütfen tekrar deneyin.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary-500 rounded-xl flex items-center justify-center">
                <Recycle className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">Smart Recycling</h1>
                <p className="text-xs text-gray-500">AI Destekli Atık Sınıflandırma</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  apiStatus === 'ready' ? 'bg-green-500 animate-pulse' :
                  apiStatus === 'no_model' ? 'bg-yellow-500' :
                  apiStatus === 'offline' ? 'bg-red-500' :
                  'bg-gray-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {apiStatus === 'ready' ? 'Hazır' :
                   apiStatus === 'no_model' ? 'Model Yok' :
                   apiStatus === 'offline' ? 'Çevrimdışı' :
                   'Kontrol ediliyor...'}
                </span>
              </div>

              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
              >
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* API Offline Warning */}
        {apiStatus === 'offline' && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-medium text-red-800">API Sunucusu Çevrimdışı</p>
              <p className="text-sm text-red-600 mt-1">
                Backend sunucusu çalışmıyor. Lütfen <code className="bg-red-100 px-1 rounded">
                python -m uvicorn api.main:app --reload</code> komutu ile başlatın.
              </p>
            </div>
          </div>
        )}

        {/* No Model Warning */}
        {apiStatus === 'no_model' && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-xl flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-medium text-yellow-800">Model Yüklenmedi</p>
              <p className="text-sm text-yellow-600 mt-1">
                AI modeli henüz eğitilmedi. Önce modeli eğitip <code className="bg-yellow-100 px-1 rounded">
                saved_models/</code> klasörüne kaydedin.
              </p>
            </div>
          </div>
        )}

        {/* Tab Seçici */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-xl p-1 shadow-md inline-flex">
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === 'upload'
                  ? 'bg-green-600 text-white shadow-md'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Upload className="w-5 h-5" />
              Görüntü Yükle
            </button>
            <button
              onClick={() => setActiveTab('realtime')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === 'realtime'
                  ? 'bg-green-600 text-white shadow-md'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Camera className="w-5 h-5" />
              Canlı Kamera
            </button>
          </div>
        </div>

        {/* Realtime Tab */}
        {activeTab === 'realtime' && (
          <div className="max-w-3xl mx-auto">
            <RealtimeCamera />
            <div className="mt-6">
              <RecyclingGuide />
            </div>
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Column - Upload */}
            <div className="space-y-6">
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
                <h2 className="text-lg font-semibold text-gray-700 mb-4">
                  Görüntü Yükle
                </h2>
                <ImageUploader
                  onUpload={handleUpload}
                  isLoading={isLoading}
                  disabled={apiStatus !== 'ready'}
                />
              </div>

              {/* Error Message */}
              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                  <p className="text-red-700">{error}</p>
                </div>
              )}

              {/* Recycling Guide - Mobile */}
              <div className="lg:hidden">
                <RecyclingGuide />
              </div>
            </div>

            {/* Right Column - Result */}
            <div className="space-y-6">
              {result ? (
                <ClassificationResult result={result} />
              ) : (
                <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-100
                                text-center text-gray-500">
                  <Recycle className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <p className="text-lg font-medium">Sonuç bekleniyor</p>
                  <p className="text-sm mt-2">
                    Bir atık görüntüsü yükleyerek sınıflandırmayı başlatın
                  </p>
                </div>
              )}

              {/* Recycling Guide - Desktop */}
              <div className="hidden lg:block">
                <RecyclingGuide />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-100 mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="text-center text-sm text-gray-500">
            <p>BM5113 Bilgisayarla Görme Dersi Projesi</p>
            <p className="mt-1">
              EfficientNetV2-S Transfer Learning ile Atık Sınıflandırma
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
