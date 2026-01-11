import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image, X } from 'lucide-react';

const ImageUploader = ({ onUpload, isLoading, disabled }) => {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
      onUpload(file);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp', '.bmp']
    },
    maxFiles: 1,
    disabled: disabled || isLoading,
  });

  const clearPreview = (e) => {
    e.stopPropagation();
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
  };

  return (
    <div
      {...getRootProps()}
      className={`
        relative border-2 border-dashed rounded-2xl p-8
        transition-all duration-300 cursor-pointer
        ${isDragActive
          ? 'border-primary-500 bg-primary-50'
          : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
        }
        ${isLoading ? 'opacity-50 cursor-wait' : ''}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input {...getInputProps()} />

      {preview ? (
        <div className="relative">
          <img
            src={preview}
            alt="Preview"
            className="max-h-72 mx-auto rounded-lg shadow-md object-contain"
          />
          {!isLoading && (
            <button
              onClick={clearPreview}
              className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full
                         hover:bg-red-600 transition-colors"
            >
              <X size={20} />
            </button>
          )}
        </div>
      ) : (
        <div className="text-center py-8">
          <div className="mx-auto w-16 h-16 mb-4 rounded-full bg-primary-100
                          flex items-center justify-center">
            {isDragActive ? (
              <Image className="w-8 h-8 text-primary-600" />
            ) : (
              <Upload className="w-8 h-8 text-primary-600" />
            )}
          </div>

          <p className="text-lg font-medium text-gray-700 mb-2">
            {isDragActive
              ? 'Görüntüyü buraya bırakın'
              : 'Atık görüntüsü yükleyin'
            }
          </p>

          <p className="text-sm text-gray-500">
            Sürükle-bırak veya tıklayarak seçin
          </p>

          <p className="text-xs text-gray-400 mt-2">
            JPEG, PNG, WebP (max 10MB)
          </p>
        </div>
      )}

      {isLoading && (
        <div className="absolute inset-0 bg-white/80 rounded-2xl
                        flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-4
                            border-primary-500 border-t-transparent mx-auto mb-3" />
            <p className="text-gray-600 font-medium">Analiz ediliyor...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
