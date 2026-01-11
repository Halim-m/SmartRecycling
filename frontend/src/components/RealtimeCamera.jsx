import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Camera, CameraOff, Loader2, RefreshCw } from 'lucide-react';
import { classifyImage } from '../services/api';

const RealtimeCamera = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [intervalId, setIntervalId] = useState(null);
  const [fps, setFps] = useState(2); // Saniyede kaç tahmin

  // Kamerayı başlat
  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 640, height: 480 }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      setError('Kamera erişimi reddedildi. Lütfen izin verin.');
      console.error('Kamera hatası:', err);
    }
  };

  // Kamerayı durdur
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
    setIsStreaming(false);
    setResult(null);
  }, [intervalId]);

  // Kare yakala ve sınıflandır
  const captureAndClassify = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Video boyutlarını al
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Kareyi canvas'a çiz
    ctx.drawImage(video, 0, 0);

    // Canvas'ı blob'a dönüştür
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      setIsProcessing(true);
      try {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        const response = await classifyImage(file);
        setResult(response);
        setError(null);
      } catch (err) {
        console.error('Sınıflandırma hatası:', err);
        // Hata durumunda sessizce devam et
      } finally {
        setIsProcessing(false);
      }
    }, 'image/jpeg', 0.8);
  }, [isProcessing]);

  // Otomatik sınıflandırma döngüsü
  const startAutoClassify = useCallback(() => {
    if (intervalId) return;

    const id = setInterval(() => {
      captureAndClassify();
    }, 1000 / fps);

    setIntervalId(id);
  }, [captureAndClassify, fps, intervalId]);

  const stopAutoClassify = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
  }, [intervalId]);

  // Component unmount olduğunda temizle
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Kutu rengi için stil
  const getBinColorStyle = (binColor) => {
    const colors = {
      'mavi': 'bg-blue-500',
      'yeşil': 'bg-green-500',
      'sarı': 'bg-yellow-500',
      'kahverengi': 'bg-amber-700',
      'siyah': 'bg-gray-800',
      'mor': 'bg-purple-500',
      'kırmızı': 'bg-red-500'
    };
    return colors[binColor] || 'bg-gray-500';
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <Camera className="w-6 h-6 text-green-600" />
          Canlı Sınıflandırma
        </h2>

        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-600">Hız:</label>
          <select
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm"
            disabled={isStreaming && intervalId}
          >
            <option value={1}>1 FPS</option>
            <option value={2}>2 FPS</option>
            <option value={3}>3 FPS</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
          {error}
        </div>
      )}

      <div className="relative">
        {/* Video ve Canvas */}
        <div className="relative aspect-video bg-gray-900 rounded-xl overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          <canvas ref={canvasRef} className="hidden" />

          {/* Overlay - Sonuç gösterimi */}
          {result && isStreaming && (
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white text-2xl font-bold">
                    {result.predicted_class_tr}
                  </p>
                  <p className="text-gray-300">
                    Güven: %{(result.confidence * 100).toFixed(1)}
                  </p>
                </div>
                <div className={`${getBinColorStyle(result.recycling_info?.bin_color)} px-4 py-2 rounded-lg`}>
                  <p className="text-white font-medium text-sm">
                    {result.recycling_info?.bin_color?.toUpperCase()} KUTU
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* İşleniyor göstergesi */}
          {isProcessing && (
            <div className="absolute top-4 right-4">
              <div className="bg-green-500 text-white px-3 py-1 rounded-full flex items-center gap-2 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" />
                Analiz
              </div>
            </div>
          )}

          {/* Kamera kapalı */}
          {!isStreaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
              <CameraOff className="w-16 h-16 mb-4" />
              <p>Kamera kapalı</p>
            </div>
          )}
        </div>

        {/* Kontrol butonları */}
        <div className="flex gap-3 mt-4">
          {!isStreaming ? (
            <button
              onClick={startCamera}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-xl font-medium flex items-center justify-center gap-2 transition-colors"
            >
              <Camera className="w-5 h-5" />
              Kamerayı Başlat
            </button>
          ) : (
            <>
              {!intervalId ? (
                <button
                  onClick={startAutoClassify}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-3 px-6 rounded-xl font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  <RefreshCw className="w-5 h-5" />
                  Otomatik Başlat
                </button>
              ) : (
                <button
                  onClick={stopAutoClassify}
                  className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white py-3 px-6 rounded-xl font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  <RefreshCw className="w-5 h-5" />
                  Durdur
                </button>
              )}
              <button
                onClick={captureAndClassify}
                disabled={isProcessing}
                className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white py-3 px-6 rounded-xl font-medium flex items-center justify-center gap-2 transition-colors"
              >
                Tek Çekim
              </button>
              <button
                onClick={stopCamera}
                className="bg-red-600 hover:bg-red-700 text-white py-3 px-6 rounded-xl font-medium flex items-center justify-center gap-2 transition-colors"
              >
                <CameraOff className="w-5 h-5" />
                Kapat
              </button>
            </>
          )}
        </div>
      </div>

      {/* Detaylı sonuç */}
      {result && (
        <div className="mt-6 p-4 bg-gray-50 rounded-xl">
          <h3 className="font-semibold text-gray-800 mb-3">Son Tahmin Detayları</h3>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Sınıf</p>
              <p className="font-medium">{result.predicted_class_tr}</p>
            </div>
            <div>
              <p className="text-gray-500">Güven</p>
              <p className="font-medium">%{(result.confidence * 100).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-500">Geri Dönüşüm</p>
              <p className="font-medium">
                {result.recycling_info?.recyclable ? '✅ Evet' : '❌ Hayır'}
              </p>
            </div>
            <div>
              <p className="text-gray-500">İşlem Süresi</p>
              <p className="font-medium">{result.processing_time_ms?.toFixed(0)} ms</p>
            </div>
          </div>

          {result.recycling_info?.instructions && (
            <div className="mt-3 p-3 bg-green-50 rounded-lg">
              <p className="text-green-800 text-sm">
                💡 {result.recycling_info.instructions}
              </p>
            </div>
          )}

          {/* Top 3 tahminler */}
          {result.top_k_predictions && (
            <div className="mt-4">
              <p className="text-gray-500 text-sm mb-2">Alternatif Tahminler:</p>
              <div className="space-y-2">
                {result.top_k_predictions.slice(0, 3).map((pred, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: `${pred.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-600 w-24">{pred.class_tr}</span>
                    <span className="text-sm font-medium w-12">
                      %{(pred.confidence * 100).toFixed(0)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RealtimeCamera;
