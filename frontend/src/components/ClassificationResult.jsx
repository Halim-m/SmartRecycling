import React from 'react';
import { CheckCircle, AlertCircle, Recycle, Trash2 } from 'lucide-react';

const binColors = {
  mavi: 'bg-blue-500',
  yesil: 'bg-green-500',
  yeşil: 'bg-green-500',
  sari: 'bg-yellow-500',
  sarı: 'bg-yellow-500',
  kahverengi: 'bg-amber-700',
  kirmizi: 'bg-red-500',
  kırmızı: 'bg-red-500',
  mor: 'bg-purple-500',
  siyah: 'bg-gray-800',
  bilinmiyor: 'bg-gray-400',
};

const ClassificationResult = ({ result }) => {
  if (!result) return null;

  const {
    predicted_class_tr,
    confidence,
    recycling_info,
    top_k_predictions,
    processing_time_ms
  } = result;

  const confidencePercent = (confidence * 100).toFixed(1);
  const binColorClass = binColors[recycling_info?.bin_color?.toLowerCase()] || 'bg-gray-400';

  return (
    <div className="animate-fadeIn space-y-6">
      {/* Ana Sonuç */}
      <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-700">Sınıflandırma Sonucu</h3>
          <span className="text-sm text-gray-500">{processing_time_ms}ms</span>
        </div>

        <div className="flex items-center gap-4 mb-6">
          <div className={`w-16 h-16 ${binColorClass} rounded-xl flex items-center justify-center`}>
            {recycling_info?.recyclable ? (
              <Recycle className="w-8 h-8 text-white" />
            ) : (
              <Trash2 className="w-8 h-8 text-white" />
            )}
          </div>

          <div>
            <h2 className="text-2xl font-bold text-gray-800">{predicted_class_tr}</h2>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 bg-gray-200 rounded-full h-2 w-32">
                <div
                  className="bg-primary-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${confidencePercent}%` }}
                />
              </div>
              <span className="text-sm font-medium text-gray-600">
                {confidencePercent}%
              </span>
            </div>
          </div>
        </div>

        {/* Geri Dönüşüm Bilgisi */}
        <div className="bg-gray-50 rounded-xl p-4">
          <div className="flex items-start gap-3">
            {recycling_info?.recyclable ? (
              <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-5 h-5 text-amber-500 mt-0.5 flex-shrink-0" />
            )}
            <div>
              <p className="font-medium text-gray-800">
                {recycling_info?.recyclable ? 'Geri Dönüştürülebilir' : 'Geri Dönüştürülemez'}
              </p>
              <p className="text-sm text-gray-600 mt-1">
                {recycling_info?.instructions}
              </p>
              {recycling_info?.environmental_impact && (
                <p className="text-xs text-gray-500 mt-2 italic">
                  {recycling_info.environmental_impact}
                </p>
              )}
            </div>
          </div>

          <div className="mt-4 flex items-center gap-2">
            <span className="text-sm text-gray-600">Kutu Rengi:</span>
            <div className={`w-6 h-6 ${binColorClass} rounded-full`} />
            <span className="text-sm font-medium capitalize">
              {recycling_info?.bin_color}
            </span>
          </div>
        </div>
      </div>

      {/* Alternatif Tahminler */}
      {top_k_predictions && top_k_predictions.length > 1 && (
        <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">
            Alternatif Tahminler
          </h3>
          <div className="space-y-3">
            {top_k_predictions.slice(1).map((pred, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-gray-600">{pred.class_tr}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gray-400 h-2 rounded-full"
                      style={{ width: `${(pred.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-500 w-12 text-right">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ClassificationResult;
