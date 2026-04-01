import React from 'react';

const MLMetricsCard = ({ accuracy = 94, metrics = [
  { label: "精確度", value: "0.92" },
  { label: "召回率", value: "0.91" },
  { label: "F1 分數", value: "0.93" }
] }) => {
  // SVG 圓形計算
  const radius = 74;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (accuracy / 100) * circumference;

  return (
    <div className="flex items-center justify-center bg-transparent py-4 font-sans">
      {/* 主要卡片容器 - 典雅毛玻璃 */}
      <div className="relative w-[560px] p-10 rounded-[32px] bg-white/5 backdrop-blur-3xl border border-white/10 shadow-[0_32px_64px_-12px_rgba(0,0,0,0.5)] overflow-hidden">
        
        {/* 背景裝飾微光 (可選) */}
        <div className="absolute -top-24 -right-24 w-64 h-64 bg-blue-500/10 rounded-full blur-[80px] pointer-events-none" />
        <div className="absolute -bottom-24 -left-24 w-64 h-64 bg-purple-500/10 rounded-full blur-[80px] pointer-events-none" />

        {/* 標題 */}
        <h2 className="text-center text-slate-300 text-2xl font-light tracking-[0.3em] mb-12">
          模型性能指標
        </h2>

        {/* 中央進度圓環區塊 */}
        <div className="relative flex flex-col items-center justify-center mb-14">
          <div className="relative w-48 h-48 flex items-center justify-center">
            {/* SVG 雙圓環 */}
            <svg className="absolute w-full h-full -rotate-90 scale-110">
              {/* 外圈裝飾細環 */}
              <circle cx="96" cy="96" r="88" fill="none" stroke="white" strokeWidth="0.5" strokeOpacity="0.1" />
              {/* 內圈底色環 */}
              <circle cx="96" cy="96" r={radius} fill="none" stroke="white" strokeWidth="1" strokeOpacity="0.05" />
              {/* 動態進度條 */}
              <circle
                cx="96"
                cy="96"
                r={radius}
                fill="none"
                stroke="rgba(255, 255, 255, 0.8)"
                strokeWidth="2.5"
                strokeDasharray={circumference}
                style={{ strokeDashoffset: offset, transition: 'stroke-dashoffset 1.5s ease-out' }}
                strokeLinecap="round"
                className="drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]"
              />
            </svg>
            
            {/* 圓環中心數值 */}
            <div className="text-center z-10">
              <div className="text-6xl font-bold text-white tracking-tighter">{accuracy}%</div>
              <div className="text-sm text-slate-400 mt-3 tracking-[0.4em] font-light">準確率</div>
            </div>
          </div>
        </div>

        {/* 底部三個子指標卡 */}
        <div className="grid grid-cols-3 gap-4 relative z-10">
          {metrics.map((m, i) => (
            <div key={i} className="bg-white/[0.03] border border-white/5 rounded-2xl p-4 transition-all hover:bg-white/[0.06] hover:scale-105 duration-300">
              <div className="text-xs text-slate-500 mb-2 tracking-widest">{m.label}</div>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-semibold text-white/90">{m.value}</span>
                {/* 裝飾性趨勢線 SVG */}
                <svg width="32" height="16" viewBox="0 0 32 16" className="text-slate-500 opacity-40">
                  <path d="M0 12 L8 4 L16 10 L32 2" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeJoin="round" />
                </svg>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MLMetricsCard;
