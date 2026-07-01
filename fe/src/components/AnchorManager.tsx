"use client";

import React, { useState, useEffect } from 'react';
import { Upload, Users, Image as ImageIcon, CheckCircle, AlertCircle, Loader2, Trash2 } from 'lucide-react';

interface Anchor {
  id: string;
  name: string;
  num_images: number;
  image_urls?: string[];
}

export default function AnchorManager() {
  const [name, setName] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [anchors, setAnchors] = useState<Anchor[]>([]);
  const [status, setStatus] = useState<{type: 'success'|'error'|'', msg: string}>({type: '', msg: ''});
  const [isLoading, setIsLoading] = useState(false);

  const fetchAnchors = async (retries = 5) => {
    try {
      const res = await fetch('http://localhost:8000/api/anchors');
      if (!res.ok) throw new Error("API not ready");
      const data = await res.json();
      setAnchors(data.anchors || []);
    } catch (e) {
      if (retries > 0) {
        setTimeout(() => fetchAnchors(retries - 1), 3000);
      } else {
        console.error("Failed to fetch anchors: Server might be down");
      }
    }
  };

  useEffect(() => {
    fetchAnchors();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selected = Array.from(e.target.files);
      if (files.length + selected.length > 3) {
        setStatus({ type: 'error', msg: 'Chỉ được phép tải lên tối đa 3 ảnh (tối đa Triple-shot).' });
        return;
      }
      setFiles(prev => [...prev, ...selected]);
      setStatus({ type: '', msg: '' });
      e.target.value = ''; // Reset input
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Bạn có chắc chắn muốn xóa dữ liệu điểm danh của ${name}?`)) return;
    
    setIsLoading(true);
    setStatus({ type: '', msg: '' });
    
    try {
      const res = await fetch(`http://localhost:8000/api/anchors/${id}`, {
        method: 'DELETE',
      });
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Không thể xóa');
      }
      
      setStatus({ type: 'success', msg: `Đã xóa thành công ${name}` });
      fetchAnchors();
    } catch (error: any) {
      setStatus({ type: 'error', msg: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteImage = async (userId: string, imageIndex: number) => {
    setIsLoading(true);
    setStatus({ type: '', msg: '' });
    
    try {
      const res = await fetch(`http://localhost:8000/api/anchors/${userId}/images/${imageIndex}`, {
        method: 'DELETE',
      });
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Không thể xóa ảnh');
      }
      
      setStatus({ type: 'success', msg: `Đã xóa ảnh thành công` });
      fetchAnchors();
    } catch (error: any) {
      setStatus({ type: 'error', msg: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddImage = async (e: React.ChangeEvent<HTMLInputElement>, userId: string) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    
    setIsLoading(true);
    setStatus({ type: '', msg: '' });
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
      const res = await fetch(`http://localhost:8000/api/anchors/${userId}/images`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Không thể thêm ảnh');
      }
      
      setStatus({ type: 'success', msg: `Đã thêm ảnh thành công` });
      fetchAnchors();
    } catch (error: any) {
      setStatus({ type: 'error', msg: error.message });
    } finally {
      setIsLoading(false);
      e.target.value = ''; // Reset input
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || files.length === 0) {
      setStatus({ type: 'error', msg: 'Vui lòng nhập tên và chọn ít nhất 1 ảnh.' });
      return;
    }
    
    setIsLoading(true);
    setStatus({ type: '', msg: '' });
    
    const formData = new FormData();
    formData.append('name', name);
    files.forEach(f => formData.append('images', f));

    try {
      const res = await fetch('http://localhost:8000/api/anchors/upload', {
        method: 'POST',
        body: formData,
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || 'Lỗi khi upload.');
      }
      
      setStatus({ type: 'success', msg: `Đăng ký thành công ${data.name}!` });
      setName('');
      setFiles([]);
      // Reset file input
      const fileInput = document.getElementById('file-upload') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
      
      fetchAnchors();
    } catch (error: any) {
      setStatus({ type: 'error', msg: error.message });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col md:flex-row gap-6">
      {/* Upload Section */}
      <div className="flex-1 bg-slate-900 p-6 rounded-xl shadow-xl border border-slate-700">
        <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
          <Upload className="w-5 h-5 text-blue-400" />
          Đăng Ký Người Mới
        </h2>
        
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Họ và Tên</label>
            <input 
              type="text" 
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="VD: Nguyễn Văn A"
              className="w-full bg-slate-800 border border-slate-600 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Ảnh mẫu (Anchor) 
              <span className="text-slate-500 text-xs ml-2 font-normal">
                1 ảnh (One-shot), 2 ảnh (Double-shot), 3 ảnh (Triple-shot)
              </span>
            </label>
            
            <div className="flex flex-wrap gap-4 mb-2">
              {files.map((f, i) => (
                <div key={i} className="relative group">
                  <img 
                    src={URL.createObjectURL(f)} 
                    alt="Preview" 
                    className="w-24 h-24 object-cover rounded-lg border-2 border-blue-500 shadow-md"
                  />
                  <button 
                    type="button" 
                    onClick={() => removeFile(i)} 
                    className="absolute -top-2 -right-2 bg-red-500 hover:bg-red-600 rounded-full w-6 h-6 flex items-center justify-center text-white text-xs font-bold shadow-lg transition"
                  >
                    ×
                  </button>
                </div>
              ))}
              
              {files.length < 3 && (
                <label className="flex flex-col items-center justify-center w-24 h-24 border-2 border-slate-600 border-dashed rounded-lg cursor-pointer bg-slate-800 hover:bg-slate-750 hover:border-blue-400 transition">
                  <ImageIcon className="w-6 h-6 text-slate-400 mb-1" />
                  <span className="text-xs text-slate-300 font-medium">Thêm ảnh</span>
                  <input 
                    type="file" 
                    className="hidden" 
                    multiple 
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                </label>
              )}
            </div>
            {files.length === 0 && (
              <p className="text-xs text-slate-500 mt-2">Vui lòng chọn ít nhất 1 ảnh để đăng ký.</p>
            )}
          </div>

          {status.msg && (
            <div className={`p-3 rounded-lg flex items-start gap-2 text-sm ${status.type === 'error' ? 'bg-red-900/30 text-red-400 border border-red-900/50' : 'bg-green-900/30 text-green-400 border border-green-900/50'}`}>
              {status.type === 'error' ? <AlertCircle className="w-4 h-4 mt-0.5" /> : <CheckCircle className="w-4 h-4 mt-0.5" />}
              {status.msg}
            </div>
          )}
          
          <button 
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition flex justify-center items-center gap-2"
          >
            {isLoading && <Loader2 className="animate-spin w-4 h-4" />}
            Xác Nhận Đăng Ký
          </button>
        </form>
      </div>

      {/* List Section */}
      <div className="flex-1 bg-slate-900 p-6 rounded-xl shadow-xl border border-slate-700 max-h-[500px] flex flex-col">
        <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
          <Users className="w-5 h-5 text-purple-400" />
          Danh Sách Đã Đăng Ký ({anchors.length})
        </h2>
        
        <div className="flex-1 overflow-y-auto pr-2">
          {anchors.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500">
              <Users className="w-12 h-12 mb-3 opacity-20" />
              <p>Chưa có ai được đăng ký.</p>
            </div>
          ) : (
            <ul className="space-y-3">
              {anchors.map(a => (
                <li key={a.id} className="bg-slate-800 p-4 rounded-lg flex flex-col gap-2 border border-slate-750">
                  <div className="flex justify-between items-center w-full">
                    <span className="font-medium text-slate-200">{a.name}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-xs bg-slate-700 text-slate-300 px-2 py-1 rounded">
                        {a.num_images} ảnh
                      </span>
                      <button 
                        onClick={() => handleDelete(a.id, a.name)} 
                        disabled={isLoading}
                        className="text-red-400 hover:text-red-300 transition p-1 disabled:opacity-50" 
                        title="Xóa"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  {a.image_urls && (
                    <div className="flex gap-2 mt-2">
                      {a.image_urls.map((url, idx) => (
                        <div key={idx} className="relative group">
                          <img 
                            src={`http://localhost:8000${url}`} 
                            alt={`${a.name} ${idx + 1}`} 
                            className="w-16 h-16 object-cover rounded-md border border-slate-600"
                          />
                          <button 
                            type="button"
                            onClick={() => handleDeleteImage(a.id, idx)}
                            disabled={isLoading}
                            className="absolute -top-1 -right-1 bg-red-500 hover:bg-red-600 rounded-full w-5 h-5 flex items-center justify-center text-white text-[10px] font-bold shadow opacity-0 group-hover:opacity-100 transition disabled:opacity-0"
                            title="Xóa ảnh này"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                      {a.image_urls.length < 3 && (
                        <label className="flex flex-col items-center justify-center w-16 h-16 border border-slate-600 border-dashed rounded-md cursor-pointer bg-slate-800 hover:bg-slate-750 transition text-slate-400 hover:text-blue-400" title="Thêm ảnh">
                          <span className="text-2xl leading-none font-light mb-1">+</span>
                          <input 
                            type="file" 
                            className="hidden" 
                            accept="image/*"
                            onChange={(e) => handleAddImage(e, a.id)}
                            disabled={isLoading}
                          />
                        </label>
                      )}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
