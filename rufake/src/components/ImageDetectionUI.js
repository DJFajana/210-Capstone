import React, { useState } from 'react';

const ImageDetectionUI = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900">RUFake</h1>
          <p className="text-gray-600">Image Authentication System</p>
        </div>

        {/* Upload Card */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="mb-6">
            <h2 className="text-2xl font-semibold">Upload Image</h2>
            <p className="text-gray-600">Choose an image to analyze</p>
          </div>

          {/* Upload Area */}
          <div 
            className="border-2 border-dashed rounded-lg p-8 text-center hover:border-blue-500 cursor-pointer"
            onClick={() => document.getElementById('file-upload').click()}
          >
            {preview ? (
              <img 
                src={preview} 
                alt="Preview" 
                className="max-h-64 mx-auto rounded-lg"
              />
            ) : (
              <div>
                <p className="text-gray-600">Click to upload image</p>
                <p className="text-sm text-gray-400">JPG, PNG up to 10MB</p>
              </div>
            )}
            <input
              id="file-upload"
              type="file"
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageDetectionUI;