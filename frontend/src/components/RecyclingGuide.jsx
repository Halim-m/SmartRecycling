import React from 'react';
import { Recycle, Trash2, Info } from 'lucide-react';

const categories = [
  { name: 'Plastik', color: 'bg-yellow-500', bin: 'Sarı Kutu', recyclable: true },
  { name: 'Kağıt', color: 'bg-blue-500', bin: 'Mavi Kutu', recyclable: true },
  { name: 'Karton', color: 'bg-indigo-500', bin: 'Mavi Kutu', recyclable: true },
  { name: 'Cam', color: 'bg-green-500', bin: 'Yeşil Kutu', recyclable: true },
  { name: 'Metal', color: 'bg-orange-500', bin: 'Sarı Kutu', recyclable: true },
  { name: 'Organik', color: 'bg-amber-700', bin: 'Kahverengi Kutu', recyclable: true },
  { name: 'E-Atık', color: 'bg-red-500', bin: 'Özel Toplama', recyclable: true },
  { name: 'Tekstil', color: 'bg-purple-500', bin: 'Mor Kutu', recyclable: true },
  { name: 'Diğer', color: 'bg-gray-600', bin: 'Siyah Kutu', recyclable: false },
];

const RecyclingGuide = () => {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center">
          <Info className="w-5 h-5 text-primary-600" />
        </div>
        <h3 className="text-lg font-semibold text-gray-700">Geri Dönüşüm Rehberi</h3>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {categories.map((cat, index) => (
          <div
            key={index}
            className="flex flex-col items-center p-3 rounded-xl bg-gray-50
                       hover:bg-gray-100 transition-colors"
          >
            <div className={`w-10 h-10 ${cat.color} rounded-full flex items-center
                            justify-center mb-2`}>
              {cat.recyclable ? (
                <Recycle className="w-5 h-5 text-white" />
              ) : (
                <Trash2 className="w-5 h-5 text-white" />
              )}
            </div>
            <span className="text-sm font-medium text-gray-700">{cat.name}</span>
            <span className="text-xs text-gray-500">{cat.bin}</span>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-primary-50 rounded-xl">
        <p className="text-sm text-primary-800">
          <strong>Bilgi:</strong> Atıkları atmadan önce temizlemeyi ve
          kategorilerine göre ayırmayı unutmayın. Doğru geri dönüşüm
          çevremizi korumaya yardımcı olur!
        </p>
      </div>
    </div>
  );
};

export default RecyclingGuide;
