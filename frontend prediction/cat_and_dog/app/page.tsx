"use client";

import { useState, useRef, ChangeEvent } from 'react';

export default function Home() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<{ animal: string; confidence: number } | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
      setError('Please upload a JPEG or PNG image');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError('File size exceeds 5MB limit');
      return;
    }

    setError(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(reader.result as string);
      setPredictionResult(null); // Reset previous results
    };
    reader.readAsDataURL(file);
  };

  const handlePredict = async () => {
    if (!imagePreview) {
      setError('Please upload an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Get the file from the file input
      const fileInput = fileInputRef.current;
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        throw new Error('No file selected');
      }

      const file = fileInput.files[0];

      // Create form data to send to API
      const formData = new FormData();
      formData.append('image', file);

      // Call the prediction API
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      if (!data.success) {
        throw new Error(data.error || 'Prediction failed');
      }

      setPredictionResult({
        animal: data.prediction.animal,
        confidence: data.prediction.confidence
      });
    } catch (err: any) {
      setError(err.message || 'Prediction failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setImagePreview(null);
    setPredictionResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4 font-sans dark:from-gray-900 dark:to-gray-800">
      <main className="w-full max-w-4xl rounded-2xl bg-white p-6 shadow-xl dark:bg-gray-800 sm:p-8">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-800 dark:text-white sm:text-4xl">Cat & Dog Classifier</h1>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            Upload an image to predict whether it contains a cat or a dog
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          {/* Left Column - Upload Section */}
          <div className="space-y-6">
            <div className="rounded-xl border-2 border-dashed border-gray-300 bg-gray-50 p-6 text-center transition-all hover:border-indigo-400 dark:border-gray-600 dark:bg-gray-700">
              <div className="flex flex-col items-center justify-center space-y-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    <span className="font-medium text-indigo-600 dark:text-indigo-400">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">PNG, JPG (MAX. 5MB)</p>
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageUpload}
                  accept="image/jpeg,image/png"
                  className="hidden"
                  id="image-upload"
                />
                <label
                  htmlFor="image-upload"
                  className="cursor-pointer rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                >
                  Browse Files
                </label>
              </div>
            </div>

            {imagePreview && (
              <div className="rounded-xl bg-gray-50 p-4 dark:bg-gray-700">
                <h3 className="mb-3 text-lg font-medium text-gray-800 dark:text-white">Uploaded Image</h3>
                <div className="flex justify-center rounded-lg border border-gray-200 bg-white p-2 dark:border-gray-600 dark:bg-gray-600">
                  <img
                    src={imagePreview}
                    alt="Uploaded preview"
                    className="h-auto max-h-64 w-full rounded-lg object-contain"
                  />
                </div>
              </div>
            )}

            <div className="flex flex-col gap-3">
              <button
                onClick={handlePredict}
                disabled={isLoading || !imagePreview}
                className={`flex w-full justify-center rounded-lg px-4 py-3 text-sm font-medium text-white transition-colors ${!imagePreview || isLoading ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'} focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  'Predict'
                )}
              </button>

              {(imagePreview || predictionResult) && (
                <button
                  onClick={handleReset}
                  className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-700"
                >
                  Reset
                </button>
              )}
            </div>

            {error && (
              <div className="rounded-lg bg-red-50 p-4 text-sm text-red-800 dark:bg-red-900/30 dark:text-red-200">
                {error}
              </div>
            )}
          </div>

          {/* Right Column - Results Section */}
          <div className="space-y-6">
            <div className="rounded-xl bg-gray-50 p-6 dark:bg-gray-700">
              <h3 className="mb-4 text-lg font-medium text-gray-800 dark:text-white">Prediction Result</h3>

              {predictionResult ? (
                <div className="space-y-4">
                  <div className="rounded-lg bg-white p-6 text-center shadow-md dark:bg-gray-600">
                    <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-indigo-100 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-300">
                      {predictionResult.animal === 'Cat' ? (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4c1.615 0 3.116.59 4.249 1.636.343.32.38.853.1 1.2-.45.563-1.077.958-1.778 1.186V10h-5V8.022c-.7-.228-1.327-.623-1.778-1.186-.28-.347-.242-.88.1-1.2A5.483 5.483 0 0112 4zM9 12l2 2 4-4" />
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      )}
                    </div>
                    <h4 className="mb-2 text-xl font-bold text-gray-800 dark:text-white">
                      This is a {predictionResult.animal}!
                    </h4>
                    <div className="mt-4">
                      <div className="mb-1 flex justify-between text-sm text-gray-600 dark:text-gray-300">
                        <span>Confidence:</span>
                        <span className="font-medium">{predictionResult.confidence}%</span>
                      </div>
                      <div className="h-2.5 w-full rounded-full bg-gray-200 dark:bg-gray-600">
                        <div
                          className="h-2.5 rounded-full bg-green-500"
                          style={{ width: `${predictionResult.confidence}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 rounded-lg bg-blue-50 p-4 text-sm text-blue-800 dark:bg-blue-900/30 dark:text-blue-200">
                    <h5 className="font-medium">How it works:</h5>
                    <p className="mt-1">
                      Our AI model analyzes the image and identifies patterns characteristic of cats and dogs.
                      The confidence score indicates how certain the model is about its prediction.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center rounded-lg bg-white p-12 text-center dark:bg-gray-600">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <h4 className="mt-4 text-lg font-medium text-gray-800 dark:text-white">Ready for Prediction</h4>
                  <p className="mt-2 text-gray-600 dark:text-gray-300">
                    Upload an image and click "Predict" to see the results.
                  </p>
                </div>
              )}
            </div>

            <div className="rounded-xl bg-gray-50 p-6 dark:bg-gray-700">
              <h3 className="mb-4 text-lg font-medium text-gray-800 dark:text-white">About This System</h3>
              <div className="space-y-3 text-sm text-gray-600 dark:text-gray-300">
                <p>
                  This classifier uses a convolutional neural network trained on thousands of cat and dog images.
                </p>
                <p>
                  The model can identify various breeds and poses with high accuracy.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
