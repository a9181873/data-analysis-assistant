import React, { useState, useRef } from 'react';
import { UploadCloud, FileType, CheckCircle } from 'lucide-react';
import './UploadSection.css';

const UploadSection = ({ onUploadSuccess }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStats, setUploadStats] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const onBtnClick = () => {
    inputRef.current.click();
  };

  const handleFile = async (file) => {
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    // Hardcode a session ID for demo purposes; normally comes from Auth or UUID generator
    const sessionId = "demo-session-id-123";

    try {
      const response = await fetch(`http://localhost:8000/api/data/upload?session_id=${sessionId}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
         throw new Error('Upload failed');
      }

      const data = await response.json();
      setUploadStats(data);
      onUploadSuccess({ sessionId, summary: data });
    } catch (err) {
      console.error(err);
      alert("上傳失敗，請檢查後端是否啟動與連線正常");
    } finally {
      setUploading(false);
    }
  };

  if (uploadStats) {
      return (
         <div className="glass-panel animate-fade-in" style={{ padding: '2rem', textAlign: 'center' }}>
            <CheckCircle className="upload-icon" style={{color: 'var(--accent-success)'}} />
            <h2 className="gradient-text">資料載入成功</h2>
            <p>已成功讀取 {uploadStats.message.split('成功載入 ')[1]}</p>
            
            <div className="stats-grid">
               <div className="stat-card glass-panel">
                 <div className="stat-value">{uploadStats.rows.toLocaleString()}</div>
                 <div className="stat-label">總行數 (Rows)</div>
               </div>
               <div className="stat-card glass-panel">
                 <div className="stat-value">{uploadStats.columns_count}</div>
                 <div className="stat-label">特徵欄位數 (Columns)</div>
               </div>
            </div>
         </div>
      );
  }

  return (
    <div className="animate-fade-in">
        <h2 style={{ marginBottom: '1.5rem' }}>上傳您的資料集</h2>
        <div 
          className={`upload-container ${dragActive ? "drag-active" : ""}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={onBtnClick}
        >
        
        {uploading ? (
            <div className="flex-center" style={{ flexDirection: 'column' }}>
                <div className="loader"></div>
                <h3>正在解析與推論資料結構...</h3>
                <p className="upload-subtitle">請稍候，這可能需要幾秒鐘</p>
            </div>
        ) : (
            <>
                <UploadCloud className="upload-icon" strokeWidth={1.5} />
                <h3 className="upload-title">點擊或拖曳檔案至此區域</h3>
                <p className="upload-subtitle">支援 CSV, TXT, Excel, SAS 等常見資料格式 (最大 200MB)</p>
                
                <input
                    ref={inputRef}
                    type="file"
                    className="upload-input"
                    onChange={handleChange}
                    accept=".csv,.txt,.xlsx,.xls,.sas7bdat"
                />
                
                <button className="btn btn-secondary">
                    <FileType size={18} /> 選擇檔案
                </button>
            </>
        )}
        </div>
    </div>
  );
};

export default UploadSection;
