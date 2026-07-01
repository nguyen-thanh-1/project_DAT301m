"use client";

import React, { useState } from 'react';
import AnchorManager from '@/components/AnchorManager';
import CameraAttendance from '@/components/CameraAttendance';
import { Camera, Users } from 'lucide-react';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'attendance' | 'anchor'>('attendance');

  return (
    <main className="min-h-screen bg-slate-950 text-slate-300 font-sans p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex flex-col items-center justify-center text-center space-y-4 pt-8 pb-4">
          <div className="inline-block p-3 bg-blue-900/30 rounded-2xl border border-blue-500/30 mb-2">
            <Camera className="w-10 h-10 text-blue-400" />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 tracking-tight">
            Smart Attendance
          </h1>
          <p className="text-slate-400 max-w-xl text-sm md:text-base">
            Hệ thống điểm danh nhận diện khuôn mặt song song, sử dụng iResNet50. Nhanh chóng, chính xác và hiện đại.
          </p>
        </header>

        {/* Tabs */}
        <div className="flex justify-center">
          <div className="inline-flex bg-slate-900 p-1.5 rounded-xl border border-slate-800 shadow-inner">
            <button
              onClick={() => setActiveTab('attendance')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all duration-300 ${
                activeTab === 'attendance' 
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/50' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              <Camera className="w-4 h-4" />
              Điểm Danh
            </button>
            <button
              onClick={() => setActiveTab('anchor')}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all duration-300 ${
                activeTab === 'anchor' 
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-900/50' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              <Users className="w-4 h-4" />
              Quản Lý Dữ Liệu
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          {activeTab === 'attendance' ? <CameraAttendance /> : <AnchorManager />}
        </div>
        
      </div>
    </main>
  );
}
