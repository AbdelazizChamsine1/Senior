import React, { useState, useEffect, useRef } from "react";

export default function FaceRecognitionUI() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const videoRef = useRef(null);

  // Access webcam
  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };
    startVideo();
  }, []);

  // Trigger recognition
  const handleRecognize = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://127.0.0.1:5000/recognize");
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ status: "error", message: "Failed to connect to backend." });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md bg-white shadow-2xl rounded-2xl p-6 text-center">
        <h1 className="text-2xl font-bold mb-4">🎥 Student Face Recognition</h1>

        <video ref={videoRef} autoPlay className="w-full rounded-lg shadow-md mb-4" />

        <button
          onClick={handleRecognize}
          disabled={loading}
          className={`w-full px-4 py-2 font-semibold rounded-lg ${
            loading ? "bg-gray-400" : "bg-blue-500 hover:bg-blue-600"
          } text-white`}
        >
          {loading ? "Detecting..." : "Recognize Face"}
        </button>

        <div className="mt-6">
          {result?.status === "success" && (
            <>
              <h2 className="text-xl font-semibold text-green-700">👋 Welcome, {result.name}!</h2>
              <p className="mt-2 italic text-green-600">"{result.quote}"</p>
            </>
          )}
          {result?.status === "unknown" && (
            <p className="text-red-500 font-semibold">Face not recognized.</p>
          )}
          {result?.status === "error" && (
            <p className="text-red-500 font-semibold">{result.message}</p>
          )}
        </div>
      </div>
    </div>
  );
}
