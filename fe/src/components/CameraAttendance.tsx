"use client";

import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, UserCheck, Loader2, Upload, X, Maximize, Minimize } from 'lucide-react';

interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface AttendanceResult {
  name: string;
  score: number;
  bbox: BBox;
}

export default function CameraAttendance() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [method, setMethod] = useState<'one_shot' | 'mean' | 'average_cosine'>('mean');
  const [numImages, setNumImages] = useState<number>(3);
  const [results, setResults] = useState<AttendanceResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [autoMode, setAutoMode] = useState(false);
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const processFrame = useCallback(async () => {
    let blob: Blob;
    
    if (uploadedImageSrc) {
      const res = await fetch(uploadedImageSrc);
      blob = await res.blob();
    } else {
      if (!webcamRef.current) return;
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;
      const resBase64 = await fetch(imageSrc);
      blob = await resBase64.blob();
    }

    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');
    formData.append('method', method);
    formData.append('num_images', numImages.toString());

    try {
      setIsLoading(true);
      const res = await fetch('http://localhost:8000/api/attendance', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      setResults(data.results || []);

      // Draw bounding boxes on canvas
      const canvas = canvasRef.current;
      if (canvas && data.results) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          data.results.forEach((res: AttendanceResult) => {
            const { x, y, w, h } = res.bbox;
            
            // Draw box
            ctx.strokeStyle = !res.name.startsWith("Unknown") ? '#22c55e' : '#ef4444'; // Green or Red
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            
            // Draw text
            ctx.fillStyle = !res.name.startsWith("Unknown") ? '#22c55e' : '#ef4444';
            ctx.font = '18px Arial';
            ctx.fillText(`${res.name} (${res.score.toFixed(2)})`, x, y - 5);
          });
        }
      }
    } catch (error) {
      console.error("Attendance failed:", error);
    } finally {
      setIsLoading(false);
    }
  }, [method, numImages, uploadedImageSrc]);

  // Draw bounding boxes when results change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    results.forEach((res) => {
      const { x, y, w, h } = res.bbox;
      
      // Draw box
      ctx.strokeStyle = !res.name.startsWith("Unknown") ? '#22c55e' : '#ef4444'; // Green or Red
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      
      // Draw label background
      ctx.fillStyle = !res.name.startsWith("Unknown") ? '#22c55e' : '#ef4444';
      const label = `${res.name} (${res.score.toFixed(2)})`;
      ctx.font = '16px Arial';
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(x, y > 20 ? y - 25 : y, textWidth + 10, 25);
      
      // Draw text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 5, y > 20 ? y - 7 : y + 17);
    });
  }, [results]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (autoMode) {
      interval = setInterval(() => {
        processFrame();
      }, 2000); // Process every 2 seconds in auto mode
    }
    return () => clearInterval(interval);
  }, [autoMode, processFrame]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const imageUrl = URL.createObjectURL(file);
    
    setUploadedImageSrc(imageUrl);
    setAutoMode(false);
    setResults([]);
    
    const img = new window.Image();
    img.onload = async () => {
      setDimensions({ width: img.width, height: img.height });

      const formData = new FormData();
      formData.append('image', file);
      formData.append('method', method);
      formData.append('num_images', numImages.toString());

      try {
        setIsLoading(true);
        const res = await fetch('http://localhost:8000/api/attendance', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) throw new Error('API Error');

        const data = await res.json();
        setResults(data.results || []);

      } catch (error) {
        console.error("Error processing uploaded image", error);
      } finally {
        setIsLoading(false);
        e.target.value = ''; 
      }
    };
    img.src = imageUrl;
  };

  const clearUploadedImage = () => {
    setUploadedImageSrc(null);
    setResults([]);
  };

  const containerClasses = isFullscreen 
    ? "fixed inset-4 z-50 rounded-xl overflow-hidden bg-black flex justify-center items-center shadow-2xl" 
    : "relative rounded-lg overflow-hidden bg-black flex justify-center items-center min-h-[480px]";

  return (
    <>
      <div className="flex flex-col md:flex-row gap-6 p-6 bg-slate-900 rounded-xl shadow-2xl border border-slate-700">
        <div className="flex-1 space-y-4 relative">
          <div className={containerClasses}>
            {uploadedImageSrc ? (
              <>
                <img src={uploadedImageSrc} alt="Uploaded" className="w-full h-full object-contain" />
                <button 
                  onClick={clearUploadedImage}
                  className="absolute top-4 right-4 bg-slate-900/80 hover:bg-red-600 text-white p-2 rounded-full transition z-20"
                  title="Đóng ảnh"
                >
                  <X className="w-5 h-5" />
                </button>
              </>
            ) : (
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={{ facingMode: "user" }}
                onUserMedia={(stream) => {
                  const track = stream.getVideoTracks()[0];
                  const settings = track.getSettings();
                  if (settings.width && settings.height) {
                    setDimensions({ width: settings.width, height: settings.height });
                  }
                }}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            )}
            <canvas
              ref={canvasRef}
              width={dimensions.width}
              height={dimensions.height}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
              style={{ objectFit: uploadedImageSrc ? 'contain' : 'cover' }}
            />
            
            <button 
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="absolute bottom-4 right-4 bg-slate-900/80 hover:bg-blue-600 text-white p-2 rounded-full transition z-20"
              title={isFullscreen ? "Thu nhỏ" : "Phóng to"}
            >
              {isFullscreen ? <Minimize className="w-5 h-5" /> : <Maximize className="w-5 h-5" />}
            </button>
          </div>
          
          <div className="flex flex-wrap items-center gap-4 shrink-0">
            <button 
              onClick={uploadedImageSrc ? clearUploadedImage : processFrame}
              disabled={isLoading || autoMode}
              className={`flex-1 font-semibold py-3 px-6 rounded-lg transition duration-200 flex items-center justify-center gap-2 disabled:opacity-50 ${
                uploadedImageSrc 
                  ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' 
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isLoading && !autoMode && !uploadedImageSrc ? <Loader2 className="animate-spin w-5 h-5" /> : <Camera className="w-5 h-5" />}
              {uploadedImageSrc ? "Trở Về Camera" : "Chụp Camera"}
            </button>
            
            <label className={`flex-1 bg-slate-700 hover:bg-slate-600 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 flex items-center justify-center gap-2 cursor-pointer ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}>
              {isLoading && uploadedImageSrc ? <Loader2 className="animate-spin w-5 h-5" /> : <Upload className="w-5 h-5" />}
              {uploadedImageSrc ? "Tải Ảnh Khác" : "Tải Ảnh Lên"}
              <input 
                type="file" 
                className="hidden" 
                accept="image/*"
                onChange={handleFileUpload}
                disabled={isLoading}
              />
            </label>

            {uploadedImageSrc ? (
              <button
                onClick={processFrame}
                disabled={isLoading}
                className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isLoading ? <Loader2 className="animate-spin w-5 h-5" /> : <UserCheck className="w-5 h-5" />}
                Nhận Diện Lại
              </button>
            ) : (
              <button
                onClick={() => setAutoMode(!autoMode)}
                className={`flex-1 px-6 py-3 rounded-lg font-semibold transition disabled:opacity-50 ${autoMode ? 'bg-red-600 hover:bg-red-700 text-white' : 'bg-slate-700 hover:bg-slate-600 text-slate-200'}`}
              >
                {autoMode ? 'Dừng Tự Động' : 'Tự Động (2s)'}
              </button>
            )}
          </div>
        </div>
        
        <div className="w-full md:w-96 flex flex-col gap-6">
          <div className="bg-slate-800 p-5 rounded-lg border border-slate-700 shrink-0">
            <h3 className="text-lg font-medium text-white mb-4">Cài Đặt Chế Độ</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Phương pháp nhận diện</label>
                <select 
                  value={method} 
                  onChange={(e) => {
                    const val = e.target.value as any;
                    setMethod(val);
                    if (val === 'one_shot') setNumImages(1);
                    else if (numImages === 1) setNumImages(3); // Auto switch back to multi if changing away from one_shot
                  }}
                  className="w-full bg-slate-700 text-white border-0 rounded-md p-2.5 focus:ring-2 focus:ring-blue-500"
                >
                  <option value="one_shot">One-shot (Tính Cosine với 1 ảnh)</option>
                  <option value="mean">Few-shot: Tính TB Embedding trước</option>
                  <option value="average_cosine">Few-shot: Tính TB Cosine sau cùng</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Số lượng ảnh đầu vào</label>
                <select 
                  value={numImages} 
                  onChange={(e) => setNumImages(Number(e.target.value))}
                  disabled={method === 'one_shot'}
                  className="w-full bg-slate-700 text-white border-0 rounded-md p-2.5 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {method === 'one_shot' ? (
                    <option value={1}>1 ảnh</option>
                  ) : (
                    <>
                      <option value={2}>2 ảnh</option>
                      <option value={3}>3 ảnh</option>
                    </>
                  )}
                </select>
              </div>

              <p className="text-xs text-slate-500 mt-2">
                * Ngưỡng: 0.57 (One-shot), 0.56 (TB Embedding), 0.50 (TB Cosine)
              </p>
            </div>
          </div>
          
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col h-[350px] overflow-hidden">
            <h3 className="text-lg font-medium text-white mb-3 flex items-center gap-2 shrink-0">
              <UserCheck className="w-5 h-5 text-green-400" />
              Có Mặt Gần Nhất
            </h3>
            
            <div className="flex-1 overflow-y-auto pr-1" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
              <style>{`.overflow-y-auto::-webkit-scrollbar { display: none; }`}</style>
              {results.length === 0 ? (
                <p className="text-sm text-slate-400 italic text-center py-6">Chưa có dữ liệu điểm danh</p>
              ) : (
                <ul className="space-y-2">
                  {results.filter(r => !r.name.startsWith("Unknown")).map((r, i) => (
                    <li key={i} className="flex justify-between items-center bg-slate-700/50 hover:bg-slate-700 py-2 px-3 rounded-md transition">
                      <span className="font-medium text-slate-200 text-sm">{r.name}</span>
                      <span className="text-xs font-mono bg-green-900/50 text-green-400 px-2 py-1 rounded">
                        {r.score.toFixed(3)}
                      </span>
                    </li>
                  ))}
                  
                  {(() => {
                    const unknowns = results.filter(r => r.name.startsWith("Unknown"));
                    if (unknowns.length === 0) return null;
                    
                    const groupedUnknowns = unknowns.reduce((acc, curr) => {
                      acc[curr.name] = (acc[curr.name] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>);
                    
                    return Object.entries(groupedUnknowns).map(([uName, count], i) => (
                      <li key={`u-${i}`} className="flex justify-between items-center bg-red-900/20 border border-red-900/50 py-2 px-3 rounded-md mt-2">
                        <span className="font-medium text-red-400 text-sm">
                          {uName === "Unknown" ? "Không nhận dạng được" : uName.replace("Unknown ", "Không nhận dạng được ")}
                        </span>
                        <span className="text-xs font-mono text-red-400">
                          {count} người
                        </span>
                      </li>
                    ));
                  })()}
                </ul>
              )}
            </div>
          </div>
        </div>
      </div>

      {isFullscreen && (
        <div 
          className="fixed inset-0 bg-slate-950/90 z-40 backdrop-blur-sm cursor-pointer"
          onClick={() => setIsFullscreen(false)}
        />
      )}
    </>
  );
}
