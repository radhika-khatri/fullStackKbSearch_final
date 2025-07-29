import React, { useState, useRef, useEffect, useCallback } from 'react';
import ChatHistory from '../pages/ChatHistory';
import VoiceAssistant from '../components/VoiceAssistant';
import '../pages/Chat.css';
import { connectWebSocketChat, createChatSession, storeChatMessage, getChatHistory, askStaticChat, chatWithImage, submitMessageFeedback, getFeedbackAnalytics, voiceChat } from '../services/api';

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
  const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);
  const [showRatingPopup, setShowRatingPopup] = useState(false);
  const [hasShownRatingPopup, setHasShownRatingPopup] = useState(false);
  const [inactivityTimer, setInactivityTimer] = useState<NodeJS.Timeout | null>(null);
  const [hasMic, setHasMic] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Microphone detection logic
  useEffect(() => {
    const checkMic = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const hasAudioInput = devices.some(device => device.kind === 'audioinput');
          setHasMic(hasAudioInput);
        } catch (e) {
          setHasMic(false);
        }
      } else {
        setHasMic(false);
      }
    };
    checkMic();
  }, []);

  const retryMic = () => {
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then(devices => {
        const hasAudioInput = devices.some(device => device.kind === 'audioinput');
        setHasMic(hasAudioInput);
      }).catch(() => setHasMic(false));
    } else {
      setHasMic(false);
    }
  };

  // Dummy handler for voice button if not already defined
  const handleVoiceClick = () => {
    // Implement your voice logic here or integrate with VoiceAssistant
    alert('Voice input not implemented in this demo.');
  };

  const resetInactivityTimer = useCallback(() => {
    // Clear existing timer using a ref to avoid dependency issues
    setInactivityTimer((prevTimer) => {
      if (prevTimer) {
        clearTimeout(prevTimer);
      }
      
      // Only set new timer if popup hasn't been shown yet
      if (!hasShownRatingPopup) {
        const newTimer = setTimeout(() => {
          setShowRatingPopup(true);
          setHasShownRatingPopup(true); // Mark as shown
        }, 60000); // 60 seconds
        
        return newTimer;
      }
      return null;
    });
  }, [hasShownRatingPopup]);

  useEffect(() => {
    const initSession = async () => {
      try {
        const res = await createChatSession('anonymous');
        setSessionId(res.data.session_id);
        
        // Check if rating popup has been shown for this session
        const ratingShown = localStorage.getItem(`rating_shown_${res.data.session_id}`);
        if (ratingShown === 'true') {
          setHasShownRatingPopup(true);
        }
        
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
    
    // Start inactivity timer
    resetInactivityTimer();
    
    // Add event listeners for user activity
    const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'];
    const handleUserActivity = () => {
      // Use callback to avoid direct dependency
      resetInactivityTimer();
    };
    
    activityEvents.forEach(event => {
      document.addEventListener(event, handleUserActivity, true);
    });
    
    // WebSocket connection
    let ws: WebSocket | null = null;
    try {
      ws = connectWebSocketChat();
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, { sender: 'bot', text: data.message }]);
      };
    } catch (err) {
      console.error('WebSocket connection failed:', err);
    }
    
    // Cleanup function
    return () => {
      activityEvents.forEach(event => {
        document.removeEventListener(event, handleUserActivity, true);
      });
      // Clean up any existing timer on unmount
      setInactivityTimer((prevTimer) => {
        if (prevTimer) {
          clearTimeout(prevTimer);
        }
        return null;
      });
      if (ws) {
        ws.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty dependency array - this should only run once on mount/unmount

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;
    
    // Reset inactivity timer when user sends a message
    resetInactivityTimer();
    
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

  const handleVoiceInput = async (text: string, audioBlob?: Blob) => {
    console.log('🎤 Voice input received:', text);
    
    try {
      if (!text.trim()) {
        console.warn('Empty voice input received');
        return;
      }
      
      // Add user voice message to chat
      const userMessage = { 
        sender: 'user' as const, 
        text: `🎤 ${text}`, // Add voice emoji to indicate voice input
        id: `voice_${Date.now()}`
      };
      setMessages(prev => [...prev, userMessage]);
      
      // Add loading message
      setMessages(prev => [...prev, { sender: 'bot' as const, text: '...' }]);
      setIsLoading(true);

      // Use voiceChat API for complete voice workflow, or fallback to regular chat
      let response;
      
      if (audioBlob) {
        try {
          console.log('🗣️ Using voice chat API...');
          response = await voiceChat(audioBlob, true, sessionId);
          
          if (response.data.success) {
            const answer = response.data.bot_response;
            console.log('✅ Voice chat successful:', answer);
            
            setMessages(prev => {
              const msgs = prev.slice(0, -1); // Remove loading message
              return [...msgs, { sender: 'bot' as const, text: answer }];
            });
          } else {
            throw new Error('Voice chat failed: ' + response.data.message);
          }
        } catch (voiceErr) {
          console.warn('Voice API failed, falling back to text chat:', voiceErr);
          // Fallback to regular text chat
          response = await askStaticChat(text);
          const answer = response.data.answer;
          
          setMessages(prev => {
            const msgs = prev.slice(0, -1); // Remove loading message
            return [...msgs, { sender: 'bot' as const, text: answer }];
          });
        }
      } else {
        // No audio blob, just use text
        response = await askStaticChat(text);
        const answer = response.data.answer;
        
        setMessages(prev => {
          const msgs = prev.slice(0, -1); // Remove loading message
          return [...msgs, { sender: 'bot' as const, text: answer }];
        });
      }

      // Reset inactivity timer
      resetInactivityTimer();
      
    } catch (error) {
      console.error('Voice input error:', error);
      
      setMessages(prev => {
        const msgs = prev.slice(0, -1); // Remove loading message
        return [...msgs, { 
          sender: 'bot' as const, 
          text: "Sorry, I couldn't process your voice input. Please try again or type your message." 
        }];
      });
    } finally {
      setIsLoading(false);
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

  const handleCopyMessage = async (messageText: string, messageIndex: number) => {
    try {
      await navigator.clipboard.writeText(messageText);
      setCopiedMessageIndex(messageIndex);
      
      // Reset copy indicator after 2 seconds
      setTimeout(() => {
        setCopiedMessageIndex(null);
      }, 2000);
      
      console.log('Message copied to clipboard');
    } catch (error) {
      console.error('Failed to copy message:', error);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = messageText;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      
      setCopiedMessageIndex(messageIndex);
      setTimeout(() => {
        setCopiedMessageIndex(null);
      }, 2000);
    }
  };

  const handleRatingSubmit = async (rating: number, comment?: string) => {
    try {
      // Submit rating as feedback
      const ratingId = `rating_${sessionId}_${Date.now()}`;
      const feedbackType = rating >= 4 ? 'up' : 'down';
      await submitMessageFeedback(ratingId, feedbackType, sessionId, comment);
      
      // Update stats
      try {
        const statsRes = await getFeedbackAnalytics();
        setFeedbackStats(statsRes.data);
      } catch (statsErr) {
        console.warn('Failed to update feedback stats:', statsErr);
      }
      
      // Close popup and mark as completed (don't show again)
      setShowRatingPopup(false);
      setHasShownRatingPopup(true);
      
      // Save to localStorage to persist across page refreshes
      if (sessionId) {
        localStorage.setItem(`rating_shown_${sessionId}`, 'true');
      }
      
      // Clear any existing timer since user has provided feedback
      if (inactivityTimer) {
        clearTimeout(inactivityTimer);
        setInactivityTimer(null);
      }
      
      // Show success toast
      setFeedbackToast({
        show: true,
        message: 'Thank you for your rating! 🌟',
        type: 'success'
      });
      
      setTimeout(() => {
        setFeedbackToast({show: false, message: '', type: 'success'});
      }, 3000);
      
    } catch (error) {
      console.error('Failed to submit rating:', error);
      setFeedbackToast({
        show: true,
        message: 'Failed to submit rating. Please try again.',
        type: 'error'
      });
      
      setTimeout(() => {
        setFeedbackToast({show: false, message: '', type: 'success'});
      }, 3000);
    }
  };

  const handleCloseRatingPopup = () => {
    // Close popup and mark as shown (don't show again)
    setShowRatingPopup(false);
    setHasShownRatingPopup(true);
    
    // Save to localStorage to persist across page refreshes
    if (sessionId) {
      localStorage.setItem(`rating_shown_${sessionId}`, 'true');
    }
    
    // Clear any existing timer
    if (inactivityTimer) {
      clearTimeout(inactivityTimer);
      setInactivityTimer(null);
    }
  };

  const handleNewChat = async () => {
    const userId = localStorage.getItem('userId') || 'guest';
    
    // Clear old session rating state
    if (sessionId) {
      localStorage.removeItem(`rating_shown_${sessionId}`);
    }
    
    try {
      const res = await createChatSession(userId);
      setSessionId(res.data.session_id);
      setMessages([{ sender: 'bot', text: "New chat started! 👋" }]);
      
      // Reset rating popup state for new session
      setHasShownRatingPopup(false);
      setShowRatingPopup(false);
      
      // Start fresh inactivity timer
      resetInactivityTimer();
      
    } catch (err) {
      setSessionId('');
      setMessages([{ sender: 'bot', text: "New chat started! 👋" }]);
      
      // Reset rating popup state even on error
      setHasShownRatingPopup(false);
      setShowRatingPopup(false);
      
      // Start fresh inactivity timer
      resetInactivityTimer();
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
                  alt="Shared content" 
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
                <div className="message-text-wrapper">
                  {msg.text}
                </div>
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
                    {/* Copy button moved to feedback section */}
                    <button
                      className={`copy-btn ${copiedMessageIndex === i ? 'copied' : ''}`}
                      onClick={() => handleCopyMessage(msg.text, i)}
                      title={copiedMessageIndex === i ? "Copied!" : "Copy message"}
                    >
                      {copiedMessageIndex === i ? '✓' : '📋'}
                    </button>
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
    title="Upload Image"
  >
    📷
  </button>

  <button
    type="button"
    className="voice-btn"
    onClick={handleVoiceClick} // your voice logic
    disabled={isLoading}
  >
    🎙️
  </button>

  <input
    type="text"
    className="chat-text-input"
    placeholder={selectedImage ? "Ask about your image..." : "Send a message or use voice..."}
    value={input}
    onChange={(e) => setInput(e.target.value)}
    disabled={isLoading}
    onKeyDown={(e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend(e);
      }
    }}
  />

  <button
    type="submit"
    className="send-btn"
    disabled={isLoading || (!input.trim() && !selectedImage)}
  >
    {isLoading ? 'Sending...' : 'Send'}
  </button>
</form>

{!hasMic && (
  <div className="voice-warning">
    No microphone devices found. Please connect a microphone. <br />
    <span onClick={retryMic} style={{ textDecoration: 'underline', cursor: 'pointer' }}>
      Click to retry detection
    </span>
  </div>
)}

      </div>
      
      {/* Rating Popup */}
      {showRatingPopup && (
        <div className="rating-popup-overlay">
          <div className="rating-popup">
            <div className="rating-popup-header">
              <h3>How was your experience?</h3>
              <button 
                className="close-popup-btn"
                onClick={handleCloseRatingPopup}
              >
                ×
              </button>
            </div>
            <div className="rating-popup-content">
              <p>Please rate your overall experience with AppGallop Assistant</p>
              <div className="star-rating">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    className="star-btn"
                    onClick={() => handleRatingSubmit(star)}
                  >
                    ⭐
                  </button>
                ))}
              </div>
              <div className="rating-actions">
                <button 
                  className="rating-action-btn later"
                  onClick={handleCloseRatingPopup}
                >
                  Maybe Later
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
