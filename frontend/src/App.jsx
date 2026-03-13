import React, { useState } from 'react';
import { Activity, Database, Sparkles, BarChart2 } from 'lucide-react';
import UploadSection from './components/UploadSection';
import AICopilot from './components/AICopilot';
import './App.css';

function App() {
  const [sessionData, setSessionData] = useState(null);

  const handleUploadSuccess = (data) => {
    setSessionData(data);
  };

  return (
    <div className="app-container">
      <nav className="navbar glass-panel">
        <div className="nav-brand">
          <Activity className="brand-icon" />
          <span className="brand-text">Data<span className="gradient-text">Assis</span></span>
        </div>
        <div className="nav-links">
          <a href="#" className="nav-link active"><Database size={18} /> 資料總覽</a>
          <a href="#" className="nav-link"><BarChart2 size={18} /> 視覺分析</a>
          <a href="#" className="nav-link"><Sparkles size={18} /> 機器學習</a>
        </div>
      </nav>

      <main className="container main-content">
        <header className="page-header animate-fade-in">
          <h1 className="page-title">數據智能工作站</h1>
          <p className="page-subtitle">無縫整合資料清理、探索式分析與 AI 副駕駛，開啟您的數據洞察之旅。</p>
        </header>

        <section className="section-upload">
          <UploadSection onUploadSuccess={handleUploadSuccess} />
        </section>

        {sessionData && (
          <section className="section-dashboard animate-fade-in glass-panel">
            <div className="dashboard-header flex-between">
              <h3>資料結構預覽</h3>
              <span className="badge">{sessionData.summary.columns_count} 個特徵</span>
            </div>

            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    {sessionData.summary.columns.map((col, idx) => (
                      <th key={idx}>
                        {col.field}
                        <span className="type-badge">{col.type.replace('dtype(\'', '').replace('\')', '')}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sessionData.summary.preview.map((row, idx) => (
                    <tr key={idx}>
                      {sessionData.summary.columns.map((col, cIdx) => (
                        <td key={cIdx}>{String(row[col.field])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      <AICopilot sessionId={sessionData?.sessionId} />
    </div>
  );
}

export default App;
