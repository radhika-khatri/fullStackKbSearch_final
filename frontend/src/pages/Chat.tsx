import React, { useState, useRef, useEffect } from 'react';
import ChatHistory from '../pages/ChatHistory';
import '../pages/Chat.css';
import { connectWebSocketChat, configureWhatsApp, sendWhatsAppMessage, createChatSession, storeChatMessage, getChatHistory, askStaticChat, chatWithImage, submitMessageFeedback, getFeedbackAnalytics } from '../services/api';

function generateSummary(messages) {
  const userMsg = messages.find(m => m.sender === 'user');
  const botMsg = messages.slice().reverse().find(m => m.sender === 'bot');
  if (!userMsg || !botMsg) return 'New Chat';
  return `${userMsg.text.slice(0, 40)} → ${botMsg.text.slice(0, 40)}`;
}

interface Message {
  sender: 'user' | 'bot';
  text: string;
  image?: string;
  id?: string;
  feedback?: 'up' | 'down' | null;
}

export default function Chat() {
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState<Message[]>([{ sender: 'bot', text: "Welcome to AppGallop! ✨" }]);
  const [input, setInput] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<{messages: any[], summary: string}[]>([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [messageFeedback, setMessageFeedback] = useState<{[key: string]: 'up' | 'down'}>({});
  const [feedbackToast, setFeedbackToast] = useState<{show: boolean, message: string, type: 'success' | 'error'}>({show: false, message: '', type: 'success'});
  const [feedbackStats, setFeedbackStats] = useState<{satisfaction_rate: number, total_feedback: number}>({satisfaction_rate: 0, total_feedback: 0});
  const [showFeedbackComment, setShowFeedbackComment] = useState<{[key: string]: boolean}>({});
  const [feedbackComments, setFeedbackComments] = useState<{[key: string]: string}>({});
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const initSession = async () => {
      try {
        const res = await createChatSession('anonymous');
        setSessionId(res.data.session_id);
        
        // Try to get chat history with better error handling
        try {
          const histRes = await getChatHistory(res.data.session_id);
          if (histRes.data && histRes.data.length > 0) {
            setMessages(histRes.data.map((m: any) => ({ sender: m.sender, text: m.message })));
          }
        } catch (histErr) {
          console.warn('Failed to load chat history:', histErr);
          // Continue without history - not critical
        }
      } catch (err: any) {
        console.error('Failed to create session:', err);
        if (err.code === 'ERR_NETWORK' || err.message?.includes('CORS')) {
          setMessages([{ sender: 'bot', text: "⚠️ Connection issue detected. Please check if the backend server is running on http://localhost:8004" }]);
        } else {
          setSessionId(''); // Chat will still work without session
          setMessages([{ sender: 'bot', text: "Welcome to AppGallop! ✨ (Session creation failed, but you can still chat)" }]);
        }
      }
    }
    
    const loadFeedbackStats = async () => {
      try {
        const statsRes = await getFeedbackAnalytics();
        setFeedbackStats(statsRes.data);
      } catch (err) {
        console.warn('Failed to load feedback stats:', err);
      }
    };
    
    initSession();
    loadFeedbackStats();
    
    try {
      const ws = connectWebSocketChat();
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, { sender: 'bot', text: data.message }]);
      };
      return () => ws.close();
    } catch (err) {
      console.error('WebSocket connection failed:', err);
    }
  }, []);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;
    
    setIsLoading(true);
    const userMessage = input || '📷 Shared an image';
    
    // Add user message
    const userMsg: Message = {
      sender: 'user',
      text: userMessage,
      image: imagePreview || undefined
    };
    setMessages(prev => [...prev, userMsg]);
    
    const originalInput = input;
    setInput('');
    
    // Clear image after sending
    const originalImage = selectedImage;
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    // Show loading message
    const loadingMsg: Message = { sender: 'bot', text: "Let me analyze that for you..." };
    setMessages(prev => [...prev, loadingMsg]);
    
    try {
      // Store user message if session exists
      if (sessionId) {
        await storeChatMessage(sessionId, 'user', userMessage);
      }
      
      let res;
      if (originalImage) {
        // Send image with optional text
        res = await chatWithImage(originalInput, originalImage);
      } else {
        // Send text only
        res = await askStaticChat(originalInput);
      }
      
      const answer = (res.data as any)?.answer || "Sorry, I couldn't process your request.";
      
      // Store bot response if session exists
      if (sessionId) {
        await storeChatMessage(sessionId, 'bot', answer);
      }
      
      setMessages(prev => {
        const msgs = prev.slice(0, -1); // Remove loading message
        const newMsgs = [...msgs, { sender: 'bot' as const, text: answer }];
        setHistory(h => [{ messages: newMsgs, summary: generateSummary(newMsgs) }, ...h]);
        return newMsgs;
      });
    } catch (err: any) {
      console.error('Chat error:', err);
      
      let errorMessage = "Sorry, there was an error processing your request.";
      
      // Provide more specific error messages
      if (err.code === 'NETWORK_ERROR' || err.message?.includes('CORS')) {
        errorMessage = "Connection error. Please check if the server is running.";
      } else if (err.response?.status === 500) {
        errorMessage = "Server error. Please try again in a moment.";
      } else if (err.response?.status === 413) {
        errorMessage = "Image too large. Please try a smaller image.";
      } else if (err.message?.includes('timeout')) {
        errorMessage = "Request timed out. Please try again with a smaller image.";
      } else if (originalImage && err.response?.status >= 400) {
        errorMessage = "Error processing image. Please try a different image or text only.";
      }
      
      setMessages(prev => {
        const msgs = prev.slice(0, -1); // Remove loading message
        const newMsgs = [...msgs, { sender: 'bot' as const, text: errorMessage }];
        setHistory(h => [{ messages: newMsgs, summary: generateSummary(newMsgs) }, ...h]);
        return newMsgs;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
      }
      
      // Validate file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        alert('Image file too large. Maximum size is 10MB.');
        return;
      }
      
      setSelectedImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFeedback = async (messageIndex: number, feedbackType: 'up' | 'down') => {
    try {
      const messageId = `${sessionId}_${messageIndex}_${Date.now()}`;
      const comment = feedbackComments[messageIndex] || null;
      
      // Update local state immediately for better UX
      setMessageFeedback(prev => ({
        ...prev,
        [messageIndex]: feedbackType
      }));

      // Submit feedback to backend
      await submitMessageFeedback(messageId, feedbackType, sessionId, comment);
      
      // Refresh feedback stats
      try {
        const statsRes = await getFeedbackAnalytics();
        setFeedbackStats(statsRes.data);
      } catch (statsErr) {
        console.warn('Failed to update feedback stats:', statsErr);
      }
      
      // Show success toast
      setFeedbackToast({
        show: true, 
        message: feedbackType === 'up' ? 'Thanks for the positive feedback! 👍' : 'Thanks for the feedback, we\'ll improve! 👎',
        type: 'success'
      });
      
      // Hide feedback comment input and clear comment
      setShowFeedbackComment(prev => ({...prev, [messageIndex]: false}));
      setFeedbackComments(prev => ({...prev, [messageIndex]: ''}));
      
      // Hide toast after 3 seconds
      setTimeout(() => {
        setFeedbackToast({show: false, message: '', type: 'success'});
      }, 3000);
      
      console.log(`Feedback submitted: ${feedbackType} for message ${messageIndex}`);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      // Revert local state on error
      setMessageFeedback(prev => {
        const updated = { ...prev };
        delete updated[messageIndex];
        return updated;
      });
      
      // Show error toast
      setFeedbackToast({
        show: true, 
        message: 'Failed to submit feedback. Please try again.',
        type: 'error'
      });
      
      // Hide toast after 3 seconds
      setTimeout(() => {
        setFeedbackToast({show: false, message: '', type: 'success'});
      }, 3000);
    }
  };

  const toggleFeedbackComment = (messageIndex: number) => {
    setShowFeedbackComment(prev => ({
      ...prev,
      [messageIndex]: !prev[messageIndex]
    }));
  };

  const handleCommentChange = (messageIndex: number, comment: string) => {
    setFeedbackComments(prev => ({
      ...prev,
      [messageIndex]: comment
    }));
  };

  const handleNewChat = async () => {
    const userId = localStorage.getItem('userId') || 'guest';
    try {
      const res = await createChatSession(userId);
      setSessionId(res.data.session_id);
      setMessages([{ sender: 'bot', text: "New chat started! 👋" }]);
    } catch (err) {
      setSessionId('');
      setMessages([{ sender: 'bot', text: "New chat started! 👋" }]);
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sessionId]);

  return (
    <div className="chat-layout">
      {/* Feedback Toast */}
      {feedbackToast.show && (
        <div className={`feedback-toast ${feedbackToast.type}`}>
          {feedbackToast.message}
        </div>
      )}
      
      <div className={`chat-sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <ChatHistory 
          history={history} 
          onNewChat={handleNewChat} 
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      </div>
      <div className="chat-main">
        <div className="chat-header">
          <div className="header-main">
            <img src="/AppgallopLG.png" alt="AppGallop Logo" className="header-logo" />
            AppGallop Assistant
          </div>
          {feedbackStats.total_feedback > 0 && (
            <div className="feedback-stats">
              <span className="stats-label">Satisfaction: </span>
              <span className="stats-value">{feedbackStats.satisfaction_rate}%</span>
              <span className="stats-count">({feedbackStats.total_feedback} reviews)</span>
            </div>
          )}
        </div>
        <div className="chat-body">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.sender}`}>
              {msg.image && (
                <img 
                  src={msg.image} 
                  alt="Shared image" 
                  style={{ 
                    maxWidth: '300px', 
                    maxHeight: '300px', 
                    borderRadius: '8px', 
                    marginBottom: '8px',
                    display: 'block'
                  }} 
                />
              )}
              <div className="message-content">
                {msg.text}
                {isLoading && msg.text.includes("Let me analyze") && (
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                )}
              </div>
              {/* Add feedback buttons for bot messages only */}
              {msg.sender === 'bot' && !msg.text.includes("Let me analyze") && !msg.text.includes("Welcome to AppGallop") && (
                <div className="feedback-section">
                  <div className="feedback-buttons">
                    <button
                      className={`feedback-btn thumbs-up ${messageFeedback[i] === 'up' ? 'active' : ''}`}
                      onClick={() => handleFeedback(i, 'up')}
                      title="This was helpful"
                      disabled={messageFeedback[i] !== undefined}
                    >
                      👍
                    </button>
                    <button
                      className={`feedback-btn thumbs-down ${messageFeedback[i] === 'down' ? 'active' : ''}`}
                      onClick={() => handleFeedback(i, 'down')}
                      title="This was not helpful"
                      disabled={messageFeedback[i] !== undefined}
                    >
                      👎
                    </button>
                    {messageFeedback[i] === undefined && (
                      <button
                        className="feedback-btn comment-btn"
                        onClick={() => toggleFeedbackComment(i)}
                        title="Add a comment"
                      >
                        💬
                      </button>
                    )}
                  </div>
                  
                  {/* Comment input */}
                  {showFeedbackComment[i] && messageFeedback[i] === undefined && (
                    <div className="feedback-comment-section">
                      <textarea
                        className="feedback-comment-input"
                        placeholder="Add a comment about this response (optional)..."
                        value={feedbackComments[i] || ''}
                        onChange={(e) => handleCommentChange(i, e.target.value)}
                        rows={2}
                      />
                      <div className="comment-actions">
                        <button
                          className="comment-action-btn submit"
                          onClick={() => handleFeedback(i, 'up')}
                        >
                          Submit with 👍
                        </button>
                        <button
                          className="comment-action-btn submit"
                          onClick={() => handleFeedback(i, 'down')}
                        >
                          Submit with 👎
                        </button>
                        <button
                          className="comment-action-btn cancel"
                          onClick={() => toggleFeedbackComment(i)}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        
        {/* Image preview section */}
        {imagePreview && (
          <div className="image-preview-section">
            <div className="image-preview-container">
              <img 
                src={imagePreview} 
                alt="Preview" 
                className="image-preview"
              />
              <span className="image-preview-text">Image selected</span>
              <button 
                type="button"
                onClick={clearImage}
                className="image-preview-close"
              >
                ×
              </button>
            </div>
          </div>
        )}
        
        <form className="chat-input" onSubmit={handleSend}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            style={{ display: 'none' }}
          />
          <button 
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="image-upload-btn"
            disabled={isLoading}
            title="Upload image"
          >
            📷
          </button>
          <input
            type="text"
            placeholder={selectedImage ? "Ask about your image..." : "Send a message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend(e);
              }
            }}
          />
          <button 
            type="submit" 
            disabled={isLoading || (!input.trim() && !selectedImage)}
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}
