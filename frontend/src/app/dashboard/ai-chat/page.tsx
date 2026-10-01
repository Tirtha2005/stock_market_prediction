"use client";

import { useState, useRef, useEffect } from "react";
import { MessageSquare, Send, Bot, User, Sparkles, TrendingUp, ShieldAlert, Newspaper, BarChart2 } from "lucide-react";
import { useAppStore } from "@/store/appStore";

export default function AIChatPage() {
  const { ticker, stockName } = useAppStore();
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: `Hello! I am your RAG-powered financial assistant. I have access to real-time market data, technical indicators, and semantic news analysis. How can I help you analyze ${stockName} (${ticker}) today?`
    }
  ]);

  useEffect(() => {
    // Reset welcome message when ticker changes
    setMessages([
      {
        role: "assistant",
        content: `Hello! I am your RAG-powered financial assistant. I have access to real-time market data, technical indicators, and semantic news analysis. How can I help you analyze ${stockName} (${ticker}) today?`
      }
    ]);
  }, [ticker, stockName]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;
    
    const userMsg = text.trim();
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setQuery("");
    setLoading(true);
    
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const cleanBase = baseUrl.endsWith('/api') ? baseUrl : `${baseUrl}/api`;
      
      const res = await fetch(`${cleanBase}/chat?ticker=${ticker}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg, ticker: ticker })
      });
      
      if (!res.ok) throw new Error("Chat request failed");
      
      const data = await res.json();
      const botResponse = data.response || data.reply || "No response generated.";
      
      setMessages(prev => [...prev, {
        role: "assistant", 
        content: botResponse
      }]);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        role: "assistant", 
        content: `Error: ${err.message}. Please verify the backend API server is running.`
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(query);
  };

  const quickPrompts = [
    { label: `Analyze ${stockName}`, query: `Give me a full stock analysis and forecast for ${stockName} (${ticker})`, icon: TrendingUp },
    { label: "RSI & MACD Indicators", query: `Explain RSI and MACD for ${ticker}`, icon: BarChart2 },
    { label: "Market Regime & Risk", query: `What is the current market regime and risk level for ${ticker}?`, icon: ShieldAlert },
    { label: "Latest Financial News", query: `Show latest financial news and headlines for ${stockName}`, icon: Newspaper },
  ];

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
          <MessageSquare className="w-6 h-6 text-primary" /> AI Chat Assistant
        </h1>
        <p className="text-secondary text-sm">Context-aware stock prediction & semantic RAG market intelligence</p>
      </div>

      <div className="glass-card flex-1 flex flex-col overflow-hidden relative">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 blur-[100px] pointer-events-none" />
        
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-4 max-w-3xl ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-surface-raised border border-border' : 'bg-primary/20 text-primary border border-primary/30'}`}>
                {msg.role === 'user' ? <User className="w-4 h-4 text-secondary" /> : <Bot className="w-5 h-5" />}
              </div>
              
              <div className={`p-4 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === 'user' 
                  ? 'bg-primary text-background rounded-tr-sm font-medium' 
                  : 'bg-surface-raised border border-border text-foreground rounded-tl-sm shadow-md'
              }`}>
                {msg.content}
                
                {msg.role === 'assistant' && i === messages.length - 1 && messages.length > 1 && !loading && (
                  <div className="mt-3 pt-3 border-t border-border/50 flex items-center gap-2 text-xs text-primary/80">
                    <Sparkles className="w-3 h-3" />
                    Context: Technical Indicators + LSTM Model + RAG Documents
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Quick Prompt Chips */}
          {messages.length === 1 && !loading && (
            <div className="pt-4">
              <p className="text-xs font-semibold text-secondary uppercase tracking-wider mb-3">Suggested Stock Market Queries:</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-2xl">
                {quickPrompts.map((p, idx) => {
                  const Icon = p.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => sendMessage(p.query)}
                      className="flex items-center gap-2 p-3 text-xs text-left text-foreground bg-surface-raised border border-border hover:border-primary/50 hover:bg-primary/10 rounded-xl transition-all shadow-sm"
                    >
                      <Icon className="w-4 h-4 text-primary shrink-0" />
                      <span className="font-medium">{p.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {loading && (
            <div className="flex gap-4 max-w-3xl">
              <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-primary/20 text-primary border border-primary/30">
                <Bot className="w-5 h-5" />
              </div>
              <div className="p-4 rounded-2xl text-sm leading-relaxed bg-surface-raised border border-border text-foreground rounded-tl-sm shadow-md flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.2s" }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.4s" }} />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="p-4 bg-surface border-t border-border">
          <form onSubmit={handleSend} className="relative">
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              placeholder={loading ? "Analyzing stock market data..." : `Ask about ${stockName} (${ticker}), RSI, MACD, or forecasts...`}
              className="w-full bg-surface-raised border border-border rounded-xl pl-4 pr-12 py-4 text-sm text-foreground focus:outline-none focus:border-primary/50 transition-colors shadow-inner disabled:opacity-50"
            />
            <button 
              type="submit"
              disabled={!query.trim() || loading}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-lg bg-primary text-background flex items-center justify-center hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="text-center mt-2 text-xs text-secondary/70">
            Powered by Time-Aware Hybrid RAG-LSTM Architecture
          </div>
        </div>
      </div>
    </div>
  );
}
