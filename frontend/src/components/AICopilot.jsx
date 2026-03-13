import React, { useState, useRef, useEffect } from 'react';
import { Bot, User, Send, Minimize2, MessageSquare, Loader } from 'lucide-react';
import './AICopilot.css';

const AICopilot = ({ sessionId }) => {
    const [collapsed, setCollapsed] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'bot', content: '您好！我是您的數據分析副駕駛 (Copilot)。您可以問我任何關於目前載入資料的問題，例如：「幫我計算各部門薪資的平均值」。' }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (!collapsed) {
            scrollToBottom();
        }
    }, [messages, isTyping, collapsed]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim() || !sessionId) return;

        const userMsg = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsTyping(true);

        try {
            const response = await fetch('http://localhost:8000/api/ai/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, message: userMsg }),
            });

            if (!response.ok) {
                throw new Error('伺服器未回應或發生錯誤');
            }

            const data = await response.json();
            setMessages(prev => [...prev, { role: 'bot', content: data.response }]);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'error', content: `連線錯誤: ${err.message}` }]);
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <div className={`ai-copilot ${collapsed ? 'collapsed' : ''}`} onClick={() => collapsed && setCollapsed(false)}>
            <div className="chat-header">
                <div className="header-title">
                    {collapsed ? <MessageSquare size={24} className="bot-icon" /> : <Bot size={20} className="bot-icon" />}
                    {!collapsed && <span>AI Copilot</span>}
                </div>
                {!collapsed && (
                    <button className="collapse-btn" onClick={(e) => { e.stopPropagation(); setCollapsed(true); }}>
                        <Minimize2 size={16} />
                    </button>
                )}
            </div>

            <div className="chat-messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.content.split('\n').map((line, i) => (
                            <p key={i}>{line}</p>
                        ))}
                    </div>
                ))}
                {isTyping && (
                    <div className="message bot">
                        <div className="typing-indicator">
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                            <div className="typing-dot"></div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <form className="chat-input-area" onSubmit={handleSend}>
                <div className="input-wrapper">
                    <input
                        type="text"
                        className="chat-input"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={sessionId ? "請問需要什麼樣的數據洞察？" : "請先在上傳區載入資料..."}
                        disabled={!sessionId || isTyping}
                    />
                    <button
                        type="submit"
                        className="send-btn"
                        disabled={!input.trim() || !sessionId || isTyping}
                        aria-label="Send message"
                    >
                        {isTyping ? <Loader size={16} className="loader" style={{ margin: 0, borderTopColor: 'white', width: 16, height: 16, borderWidth: 2 }} /> : <Send size={16} style={{ transform: 'translateX(-1px)' }} />}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default AICopilot;
