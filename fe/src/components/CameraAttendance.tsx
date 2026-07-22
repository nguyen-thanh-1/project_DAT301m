"use client";

import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, UserCheck, Loader2, Upload, X, Maximize, Minimize, Video, ShieldCheck, ShieldAlert } from 'lucide-react';
import { getApiBaseUrl } from '@/lib/api';

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
  cam?: string;
  avg_score?: number;
  score_cam1?: number;
  score_cam2?: number;
  is_real?: boolean;
  spoof_prob?: number;
}

export default function CameraAttendance() {
  const webcam1Ref = useRef<Webcam>(null);
  const canvas1Ref = useRef<HTMLCanvasElement>(null);
  const webcam2Ref = useRef<Webcam>(null);
  const canvas2Ref = useRef<HTMLCanvasElement>(null);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [cam1Id, setCam1Id] = useState<string>('');
  const [cam2Id, setCam2Id] = useState<string>('');
  const [isDualMode, setIsDualMode] = useState<boolean>(false); // Mặc định chạy 1 camera ban đầu
  const [cam2Error, setCam2Error] = useState<boolean>(false);

  const [method, setMethod] = useState<'one_shot' | 'mean' | 'average_cosine'>('mean');
  const [numImages, setNumImages] = useState<number>(3);
  const [enableAntiSpoof, setEnableAntiSpoof] = useState<boolean>(true); // Mặc định bật chế độ chống giả mạo
  
  const [results1, setResults1] = useState<AttendanceResult[]>([]);
  const [results2, setResults2] = useState<AttendanceResult[]>([]);
  const [dualCombinedResults, setDualCombinedResults] = useState<AttendanceResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dimensions1, setDimensions1] = useState({ width: 640, height: 480 });
  const [dimensions2, setDimensions2] = useState({ width: 640, height: 480 });
  const [autoMode, setAutoMode] = useState(false);
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Lấy danh sách các thiết bị camera đang kết nối
  const updateDevices = useCallback(async () => {
    try {
      const mediaDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = mediaDevices.filter(({ kind }) => kind === 'videoinput');
      setDevices(videoDevices);
      
      if (videoDevices.length > 0) {
        if (!cam1Id) setCam1Id(videoDevices[0].deviceId);
        if (!cam2Id) {
          // Nếu có từ 2 camera trở lên thì gán camera thứ 2 cho cam2, ngược lại dùng tạm camera 1
          if (videoDevices.length > 1) {
            setCam2Id(videoDevices[1].deviceId);
          } else {
            setCam2Id(videoDevices[0].deviceId);
          }
        }
      }
      if (videoDevices.length < 2) {
        setCam2Error(true);
      } else {
        setCam2Error(false);
      }
    } catch (err) {
      console.error("Error enumerating devices:", err);
      setCam2Error(true);
    }
  }, [cam1Id, cam2Id]);

  useEffect(() => {
    updateDevices();
  }, [updateDevices]);

  const handleUserMedia = useCallback((stream: MediaStream, isCam1: boolean) => {
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();
    if (settings.width && settings.height) {
      if (isCam1) setDimensions1({ width: settings.width, height: settings.height });
      else setDimensions2({ width: settings.width, height: settings.height });
    }
    // Cập nhật lại tên thiết bị sau khi người dùng cấp quyền
    updateDevices();
  }, [updateDevices]);

  const sendFrameToAPI = async (blob: Blob, camLabel: string): Promise<AttendanceResult[]> => {
    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');
    formData.append('method', method);
    formData.append('num_images', numImages.toString());
    formData.append('check_spoof', enableAntiSpoof ? 'true' : 'false');

    const res = await fetch(`${getApiBaseUrl()}/api/attendance`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    const resList = data.results || [];
    return resList.map((r: AttendanceResult) => ({ ...r, cam: camLabel }));
  };

  const processFrame = useCallback(async () => {
    try {
      setIsLoading(true);

      if (uploadedImageSrc) {
        const res = await fetch(uploadedImageSrc);
        const blob = await res.blob();
        const resList = await sendFrameToAPI(blob, 'Ảnh Tải Lên');
        setResults1(resList);
        setResults2([]);
        setDualCombinedResults([]);
      } else if (isDualMode && !cam2Error && devices.length >= 2 && webcam1Ref.current && webcam2Ref.current) {
        // Chế độ 2 camera: Gửi đồng thời 2 ảnh lên endpoint /api/attendance/dual để tính trung bình với threshold 0.5
        const imageSrc1 = webcam1Ref.current.getScreenshot();
        const imageSrc2 = webcam2Ref.current.getScreenshot();

        if (imageSrc1 && imageSrc2) {
          const [blob1, blob2] = await Promise.all([
            fetch(imageSrc1).then(r => r.blob()),
            fetch(imageSrc2).then(r => r.blob())
          ]);

          const formData = new FormData();
          formData.append('image1', blob1, 'cam1.jpg');
          formData.append('image2', blob2, 'cam2.jpg');
          formData.append('method', method);
          formData.append('num_images', numImages.toString());
          formData.append('threshold', '0.5');
          formData.append('check_spoof', enableAntiSpoof ? 'true' : 'false');

          const res = await fetch(`${getApiBaseUrl()}/api/attendance/dual`, {
            method: 'POST',
            body: formData,
          });
          if (!res.ok) throw new Error('API Dual Error');
          const data = await res.json();

          setResults1(data.results_cam1 || []);
          setResults2(data.results_cam2 || []);
          setDualCombinedResults(data.results || []);
        }
      } else {
        // Chế độ 1 camera (hoặc khi bật 2 camera nhưng không có camera thứ 2 -> tự động hoạt động như chế độ 1 camera)
        if (webcam1Ref.current) {
          const imageSrc1 = webcam1Ref.current.getScreenshot();
          if (imageSrc1) {
            const blob = await fetch(imageSrc1).then(r => r.blob());
            const resList = await sendFrameToAPI(blob, 'Cam 1 (Trái)');
            setResults1(resList);
            setResults2([]);
            setDualCombinedResults([]);
          }
        }
      }
    } catch (error) {
      console.error("Attendance failed:", error);
    } finally {
      setIsLoading(false);
    }
  }, [method, numImages, uploadedImageSrc, isDualMode, cam2Error, devices.length, enableAntiSpoof]);

  const drawBoundingBoxes = (canvas: HTMLCanvasElement | null, resList: AttendanceResult[]) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    resList.forEach((res) => {
      const { x, y, w, h } = res.bbox;
      
      const isSpoof = res.is_real === false || res.name.startsWith("Cảnh báo:");
      const isUnknown = res.name.startsWith("Unknown");

      // Draw box
      ctx.strokeStyle = isSpoof ? '#ef4444' : (!isUnknown ? '#22c55e' : '#eab308');
      ctx.lineWidth = isSpoof ? 4 : 3;
      ctx.strokeRect(x, y, w, h);
      
      // Draw label background and text
      let label = res.name;
      if (isSpoof) {
        label = `❌ GIẢ MẠO (${((1 - (res.spoof_prob ?? 0)) * 100).toFixed(0)}%)`;
      } else if (res.is_real === true && res.spoof_prob !== undefined) {
        const spoofTag = `🛡️ Thật (${(res.spoof_prob * 100).toFixed(0)}%)`;
        label = res.avg_score !== undefined 
          ? `${res.name} | ${spoofTag} (TB: ${res.avg_score.toFixed(2)})` 
          : `${res.name} | ${spoofTag} (${res.score.toFixed(2)})`;
      } else {
        label = res.avg_score !== undefined 
          ? `${res.name} (TB: ${res.avg_score.toFixed(2)})` 
          : `${res.name} (${res.score.toFixed(2)})`;
      }

      ctx.fillStyle = isSpoof ? '#ef4444' : (!isUnknown ? '#22c55e' : '#eab308');
      ctx.font = isSpoof ? 'bold 16px Arial' : '16px Arial';
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(x, y > 24 ? y - 28 : y, textWidth + 12, 28);
      
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 6, y > 24 ? y - 8 : y + 20);
    });
  };

  useEffect(() => {
    drawBoundingBoxes(canvas1Ref.current, results1);
  }, [results1, dimensions1]);

  useEffect(() => {
    drawBoundingBoxes(canvas2Ref.current, results2);
  }, [results2, dimensions2]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (autoMode) {
      interval = setInterval(() => {
        processFrame();
      }, 2000); // Tự động quét mỗi 2 giây
    }
    return () => clearInterval(interval);
  }, [autoMode, processFrame]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const imageUrl = URL.createObjectURL(file);
    
    setUploadedImageSrc(imageUrl);
    setAutoMode(false);
    setResults1([]);
    setResults2([]);
    setDualCombinedResults([]);
    
    const img = new window.Image();
    img.onload = async () => {
      setDimensions1({ width: img.width, height: img.height });

      try {
        setIsLoading(true);
        const formData = new FormData();
        formData.append('image', file);
        formData.append('method', method);
        formData.append('num_images', numImages.toString());
        formData.append('check_spoof', enableAntiSpoof ? 'true' : 'false');

        const res = await fetch(`${getApiBaseUrl()}/api/attendance`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) throw new Error('API Error');

        const data = await res.json();
        const resList = (data.results || []).map((r: AttendanceResult) => ({ ...r, cam: 'Ảnh Tải Lên' }));
        setResults1(resList);
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
    setResults1([]);
    setResults2([]);
    setDualCombinedResults([]);
  };

  const combinedResults = isDualMode && dualCombinedResults.length > 0 ? dualCombinedResults : [...results1, ...results2];

  const containerClasses = isFullscreen 
    ? "fixed inset-4 z-50 rounded-xl overflow-hidden bg-black flex flex-col justify-center items-center shadow-2xl p-4 gap-4 overflow-y-auto" 
    : "relative rounded-lg overflow-hidden flex flex-col gap-4";

  return (
    <>
      <div className="flex flex-col md:flex-row gap-6 p-6 bg-slate-900 rounded-xl shadow-2xl border border-slate-700">
        <div className="flex-1 space-y-4 relative">
          <div className={containerClasses}>
            {uploadedImageSrc ? (
              <div className="relative w-full h-[480px] bg-black rounded-lg overflow-hidden flex justify-center items-center border border-slate-700">
                <img src={uploadedImageSrc} alt="Uploaded" className="w-full h-full object-contain" />
                <canvas
                  ref={canvas1Ref}
                  width={dimensions1.width}
                  height={dimensions1.height}
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  style={{ objectFit: 'contain' }}
                />
                <button 
                  onClick={clearUploadedImage}
                  className="absolute top-4 right-4 bg-slate-900/80 hover:bg-red-600 text-white p-2 rounded-full transition z-20"
                  title="Đóng ảnh"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            ) : (
              <div className={`grid gap-4 w-full ${isDualMode ? 'grid-cols-1 xl:grid-cols-2' : 'grid-cols-1'}`}>
                {/* Camera 1 Box */}
                <div className="flex flex-col bg-slate-800/80 rounded-lg border border-slate-700 overflow-hidden shadow-lg">
                  <div className="flex items-center justify-between px-3 py-2 bg-slate-800 border-b border-slate-700">
                    <span className="text-xs font-semibold text-blue-400 flex items-center gap-1.5">
                      <Video className="w-4 h-4" /> Camera 1 (Góc Trái / Máy)
                    </span>
                    {devices.length > 0 && (
                      <select
                        value={cam1Id}
                        onChange={(e) => setCam1Id(e.target.value)}
                        className="bg-slate-900 text-slate-200 text-xs px-2 py-1 rounded border border-slate-600 focus:outline-none focus:border-blue-500 max-w-[170px] truncate"
                      >
                        {devices.map((d, idx) => (
                          <option key={d.deviceId || idx} value={d.deviceId}>
                            {d.label || `Camera ${idx + 1}`}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>
                  <div className="relative bg-black flex justify-center items-center min-h-[340px] aspect-video">
                    <Webcam
                      ref={webcam1Ref}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{ deviceId: cam1Id ? { exact: cam1Id } : undefined }}
                      onUserMedia={(stream) => handleUserMedia(stream, true)}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                    <canvas
                      ref={canvas1Ref}
                      width={dimensions1.width}
                      height={dimensions1.height}
                      className="absolute top-0 left-0 w-full h-full pointer-events-none"
                      style={{ objectFit: 'cover' }}
                    />
                  </div>
                </div>

                {/* Camera 2 Box - Only shown when in Dual Mode */}
                {isDualMode && (
                  <div className="flex flex-col bg-slate-800/80 rounded-lg border border-slate-700 overflow-hidden shadow-lg">
                    <div className="flex items-center justify-between px-3 py-2 bg-slate-800 border-b border-slate-700">
                      <span className="text-xs font-semibold text-purple-400 flex items-center gap-1.5">
                        <Video className="w-4 h-4" /> Camera 2 (Góc Phải / Webcam rời)
                      </span>
                      {devices.length >= 2 && !cam2Error && (
                        <select
                          value={cam2Id}
                          onChange={(e) => setCam2Id(e.target.value)}
                          className="bg-slate-900 text-slate-200 text-xs px-2 py-1 rounded border border-slate-600 focus:outline-none focus:border-purple-500 max-w-[170px] truncate"
                        >
                          {devices.map((d, idx) => (
                            <option key={d.deviceId || idx} value={d.deviceId}>
                              {d.label || `Camera ${idx + 1}`}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                    {devices.length < 2 || cam2Error ? (
                      <div className="relative bg-black/80 flex flex-col justify-center items-center min-h-[340px] aspect-video p-6 text-center border-t border-slate-700">
                        <Video className="w-12 h-12 mb-3 text-red-400/80 animate-pulse" />
                        <p className="font-semibold text-red-300 text-base mb-1.5">Không tìm thấy camera thứ 2</p>
                        <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
                          Vui lòng kết nối webcam rời qua cổng USB. Hệ thống đang tự động hoạt động như chế độ 1 camera.
                        </p>
                      </div>
                    ) : (
                      <div className="relative bg-black flex justify-center items-center min-h-[340px] aspect-video">
                        <Webcam
                          ref={webcam2Ref}
                          audio={false}
                          screenshotFormat="image/jpeg"
                          videoConstraints={{ deviceId: cam2Id ? { exact: cam2Id } : undefined }}
                          onUserMedia={(stream) => handleUserMedia(stream, false)}
                          onUserMediaError={() => setCam2Error(true)}
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        />
                        <canvas
                          ref={canvas2Ref}
                          width={dimensions2.width}
                          height={dimensions2.height}
                          className="absolute top-0 left-0 w-full h-full pointer-events-none"
                          style={{ objectFit: 'cover' }}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            
            <button 
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="absolute bottom-4 right-4 bg-slate-900/80 hover:bg-blue-600 text-white p-2 rounded-full transition z-20 shadow-lg border border-slate-700"
              title={isFullscreen ? "Thu nhỏ" : "Phóng to"}
            >
              {isFullscreen ? <Minimize className="w-5 h-5" /> : <Maximize className="w-5 h-5" />}
            </button>
          </div>
          
          {/* Controls Bar */}
          <div className="flex flex-wrap items-center gap-3 shrink-0 pt-2">
            <button 
              onClick={uploadedImageSrc ? clearUploadedImage : processFrame}
              disabled={isLoading || autoMode}
              className={`flex-1 font-semibold py-3 px-5 rounded-lg transition duration-200 flex items-center justify-center gap-2 disabled:opacity-50 shadow-md ${
                uploadedImageSrc 
                  ? 'bg-slate-700 hover:bg-slate-600 text-slate-200' 
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isLoading && !autoMode && !uploadedImageSrc ? <Loader2 className="animate-spin w-5 h-5" /> : <Camera className="w-5 h-5" />}
              {uploadedImageSrc ? "Trở Về Camera" : (isDualMode ? "Chụp Cả 2 Camera" : "Chụp Camera")}
            </button>
            
            <label className={`flex-1 bg-slate-700 hover:bg-slate-600 text-white font-semibold py-3 px-5 rounded-lg transition duration-200 flex items-center justify-center gap-2 cursor-pointer shadow-md ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}>
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

            {!uploadedImageSrc && (
              <>
                <button
                  onClick={() => setAutoMode(!autoMode)}
                  className={`px-5 py-3 rounded-lg font-semibold transition disabled:opacity-50 shadow-md ${autoMode ? 'bg-red-600 hover:bg-red-700 text-white animate-pulse' : 'bg-slate-700 hover:bg-slate-600 text-slate-200'}`}
                >
                  {autoMode ? '⏹ Dừng Tự Động' : '⚡ Tự Động (2s)'}
                </button>

                <button
                  onClick={() => setIsDualMode(!isDualMode)}
                  className={`px-5 py-3 rounded-lg font-semibold transition flex items-center justify-center gap-2 shadow-md border ${
                    isDualMode 
                      ? 'bg-purple-600 hover:bg-purple-700 text-white border-purple-500' 
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-600'
                  }`}
                  title="Chuyển đổi giữa chế độ 1 camera và 2 camera song song"
                >
                  <span>{isDualMode ? '🔴 2 Camera (Song Song)' : '⚪ 1 Camera'}</span>
                </button>

                <button
                  onClick={() => setEnableAntiSpoof(!enableAntiSpoof)}
                  className={`px-5 py-3 rounded-lg font-semibold transition flex items-center justify-center gap-2 shadow-md border ${
                    enableAntiSpoof 
                      ? 'bg-emerald-600 hover:bg-emerald-700 text-white border-emerald-500 shadow-emerald-900/30' 
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-400 border-slate-600'
                  }`}
                  title="Chống giả mạo bằng ảnh/màn hình trước khi nhận diện"
                >
                  {enableAntiSpoof ? <ShieldCheck className="w-5 h-5" /> : <ShieldAlert className="w-5 h-5" />}
                  <span>{enableAntiSpoof ? '🛡️ Anti-Face: BẬT' : '🛡️ Anti-Face: TẮT'}</span>
                </button>
              </>
            )}
          </div>
        </div>
        
        {/* Right Sidebar */}
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
                    else if (numImages === 1) setNumImages(3);
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

              <div className="pt-3 border-t border-slate-700">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm font-medium text-slate-300 flex items-center gap-1.5">
                    {enableAntiSpoof ? <ShieldCheck className="w-4 h-4 text-emerald-400" /> : <ShieldAlert className="w-4 h-4 text-slate-400" />}
                    Chống Giả Mạo (Anti-Face)
                  </span>
                  <input 
                    type="checkbox" 
                    checked={enableAntiSpoof} 
                    onChange={(e) => setEnableAntiSpoof(e.target.checked)}
                    className="w-4 h-4 text-emerald-600 bg-slate-700 border-slate-600 rounded focus:ring-emerald-500 focus:ring-2"
                  />
                </label>
                <p className="text-xs text-slate-500 mt-1">
                  Kiểm tra giả mạo (ảnh chụp, màn hình) trước khi xác nhận và định danh.
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col h-[380px] overflow-hidden">
            <h3 className="text-lg font-medium text-white mb-3 flex items-center justify-between shrink-0">
              <span className="flex items-center gap-2">
                <UserCheck className="w-5 h-5 text-green-400" />
                Có Mặt Gần Nhất
              </span>
              <span className="text-xs font-normal bg-slate-700 text-slate-300 px-2 py-0.5 rounded">
                {combinedResults.filter(r => !r.name.startsWith("Unknown")).length} lượt
              </span>
            </h3>
            
            <div className="flex-1 overflow-y-auto pr-1" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
              <style>{`.overflow-y-auto::-webkit-scrollbar { display: none; }`}</style>
              {combinedResults.length === 0 ? (
                <p className="text-sm text-slate-400 italic text-center py-6">Chưa có dữ liệu điểm danh</p>
              ) : (
                <ul className="space-y-2">
                  {combinedResults.filter(r => !r.name.startsWith("Unknown") && !r.name.startsWith("Cảnh báo:") && r.is_real !== false).map((r, i) => (
                    <li key={i} className="flex justify-between items-center bg-slate-700/50 hover:bg-slate-700 py-2 px-3 rounded-md transition border-l-4 border-green-500">
                      <div className="flex flex-col">
                        <span className="font-medium text-slate-200 text-sm flex items-center gap-1.5">
                          {r.name}
                          {r.is_real === true && r.spoof_prob !== undefined && (
                            <span className="text-[10px] bg-emerald-900/70 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-600/50">
                              🛡️ Thật {(r.spoof_prob * 100).toFixed(0)}%
                            </span>
                          )}
                        </span>
                        {r.score_cam1 !== undefined && r.score_cam2 !== undefined ? (
                          <span className="text-[11px] text-purple-300">
                            ⚡ TB 2 Cam: Cam1 ({r.score_cam1.toFixed(2)}) + Cam2 ({r.score_cam2.toFixed(2)})
                          </span>
                        ) : (
                          r.cam && <span className="text-[11px] text-slate-400">{r.cam}</span>
                        )}
                      </div>
                      <span className="text-xs font-mono bg-green-900/50 text-green-400 px-2 py-1 rounded">
                        {r.score.toFixed(3)}
                      </span>
                    </li>
                  ))}
                  
                  {(() => {
                    const spoofs = combinedResults.filter(r => r.is_real === false || r.name.startsWith("Cảnh báo:"));
                    if (spoofs.length === 0) return null;
                    
                    return spoofs.map((s, i) => (
                      <li key={`spoof-${i}`} className="flex justify-between items-center bg-red-950/40 border border-red-600/60 py-2 px-3 rounded-md mt-2 border-l-4 border-red-500 animate-pulse">
                        <div className="flex flex-col">
                          <span className="font-bold text-red-400 text-sm flex items-center gap-1.5">
                            ❌ PHÁT HIỆN GIẢ MẠO
                          </span>
                          <span className="text-[11px] text-red-300/80">
                            {s.cam || 'Cam 1'} | Độ tin cậy giả: {((1 - (s.spoof_prob ?? 0)) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <span className="text-xs font-mono bg-red-900/80 text-white px-2 py-1 rounded font-semibold">
                          BỊ CHẶN
                        </span>
                      </li>
                    ));
                  })()}

                  {(() => {
                    const unknowns = combinedResults.filter(r => r.name.startsWith("Unknown") && r.is_real !== false);
                    if (unknowns.length === 0) return null;
                    
                    const groupedUnknowns = unknowns.reduce((acc, curr) => {
                      const key = `${curr.name}___${curr.cam || ''}`;
                      acc[key] = (acc[key] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>);
                    
                    return Object.entries(groupedUnknowns).map(([uKey, count], i) => {
                      const [uName, uCam] = uKey.split('___');
                      return (
                        <li key={`u-${i}`} className="flex justify-between items-center bg-yellow-900/20 border border-yellow-700/50 py-2 px-3 rounded-md mt-2 border-l-4 border-yellow-500">
                          <div className="flex flex-col">
                            <span className="font-medium text-yellow-400 text-sm">
                              {uName === "Unknown" ? "Không nhận dạng được" : uName.replace("Unknown ", "Không nhận dạng được ")}
                            </span>
                            {uCam && <span className="text-[11px] text-yellow-300/70">{uCam}</span>}
                          </div>
                          <span className="text-xs font-mono text-yellow-400">
                            {count} người
                          </span>
                        </li>
                      );
                    });
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
