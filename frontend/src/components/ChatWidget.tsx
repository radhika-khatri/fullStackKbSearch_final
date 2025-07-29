import React, { useState, useRef, useEffect } from 'react';
import './ChatWidget.css';
import VoiceAssistant from './VoiceAssistant';
import { useWidgetConfig } from './WidgetConfigContext';
import { askStaticChat, submitMessageFeedback, voiceChat } from '../services/api';
import { processImageForQuote } from '../services/imageQuote';
import { useNavigate } from 'react-router-dom';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  image?: string;
  id?: string;
  feedback?: 'up' | 'down' | null;
}

const initialMessages: Message[] = [
  { 
    sender: 'bot', 
    text: 'Hi! I am AppGallop AI. How can I help you today?',
    id: 'initial_message'
  }
];

const ChatWidget: React.FC = () => {
  const navigate = useNavigate();

  const { config } = useWidgetConfig();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  // Generate unique message ID
  const generateMessageId = () => {
    return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  };

  // Handle copy message functionality
  const handleCopyMessage = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => {
        setCopiedMessageId(null);
      }, 2000);
    } catch (error) {
      console.error('Failed to copy message:', error);
    }
  };

  // Handle feedback submission
  const handleFeedback = async (messageId: string, feedbackType: 'up' | 'down') => {
    try {
      await submitMessageFeedback(
        messageId,
        feedbackType,
        'widget_session_' + Date.now(),
        null
      );

      // Update the message with feedback
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { ...msg, feedback: feedbackType }
          : msg
      ));

      console.log(`Feedback submitted: ${feedbackType} for message ${messageId}`);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  useEffect(() => {
    // Start with initial messages when widget opens
    if (open && messages.length === 0) {
      setMessages(initialMessages);
    }
  }, [open, messages.length]);

  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [messages, open]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;
    
    setIsLoading(true);
    
    // Add user message to chat
    const userMessage: Message = {
      sender: 'user',
      text: input || '📷 Shared an image',
      image: imagePreview || undefined,
      id: generateMessageId()
    };
    setMessages(prev => [...prev, userMessage]);
    
    const userInput = input;
    setInput('');
    
    // Clear image after sending
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    try {
      let response;
      
      if (selectedImage) {
        // Send image for quote extraction
        const data: any = await processImageForQuote(selectedImage);
        let botText = data.vision_analysis || 'Image processed.';
        let quote = data.quote;
        const botMessage: Message = {
          sender: 'bot',
          text: botText,
          id: generateMessageId(),
          feedback: null,
          // @ts-ignore
          quote: quote
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        // Send text only
        const response = await askStaticChat(userInput);
        const data = response.data;
        const botMessage: Message = {
          sender: 'bot',
          text: data.answer,
          id: generateMessageId(),
          feedback: null
        };
        setMessages(prev => [...prev, botMessage]);
      }
      
    } catch (err) {
      console.error('Error sending message:', err);
      const errorMessage: Message = {
        sender: 'bot',
        text: 'Sorry, I could not get a response from the server.',
        id: generateMessageId(),
        feedback: null
      };
      setMessages(prev => [...prev, errorMessage]);
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
    console.log('🎤 Widget voice input received:', text);
    
    try {
      if (!text.trim()) {
        console.warn('Empty voice input received');
        return;
      }
      
      // Add user voice message to chat
      const userMessage: Message = { 
        sender: 'user',
        text: `🎤 ${text}`, // Add voice emoji to indicate voice input
        id: generateMessageId()
      };
      setMessages(prev => [...prev, userMessage]);
      setIsLoading(true);

      // Use voiceChat API for complete voice workflow, or fallback to regular chat
      let response;
      
      if (audioBlob) {
        try {
          console.log('🗣️ Using voice chat API for widget...');
          response = await voiceChat(audioBlob, true, 'widget_session');
          
          if (response.data.success) {
            const answer = response.data.bot_response;
            console.log('✅ Widget voice chat successful:', answer);
            
            const botMessage: Message = {
              sender: 'bot',
              text: answer,
              id: generateMessageId(),
              feedback: null
            };
            setMessages(prev => [...prev, botMessage]);
          } else {
            throw new Error('Voice chat failed: ' + response.data.message);
          }
        } catch (voiceErr) {
          console.warn('Voice API failed, falling back to text chat:', voiceErr);
          // Fallback to regular text chat
          response = await askStaticChat(text);
          const answer = response.data.answer;
          
          const botMessage: Message = {
            sender: 'bot',
            text: answer,
            id: generateMessageId(),
            feedback: null
          };
          setMessages(prev => [...prev, botMessage]);
        }
      } else {
        // No audio blob, just use text
        response = await askStaticChat(text);
        const answer = response.data.answer;
        
        const botMessage: Message = {
          sender: 'bot',
          text: answer,
          id: generateMessageId(),
          feedback: null
        };
        setMessages(prev => [...prev, botMessage]);
      }
      
    } catch (error) {
      console.error('Widget voice input error:', error);
      
      const errorMessage: Message = {
        sender: 'bot',
        text: "Sorry, I couldn't process your voice input. Please try again or type your message.",
        id: generateMessageId(),
        feedback: null
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <button
        className="chat-widget-fab"
        onClick={() => setOpen(o => !o)}
        aria-label="Open chat"
        style={{
          background: config.widgetColor,
          position: 'fixed',
          zIndex: 9999,
          ...(config.widgetPosition === 'bottomRight' ? { right: 32, bottom: 32 } : {}),
          ...(config.widgetPosition === 'bottomLeft' ? { left: 32, bottom: 32 } : {}),
          ...(config.widgetPosition === 'topRight' ? { right: 32, top: 32 } : {}),
          ...(config.widgetPosition === 'topLeft' ? { left: 32, top: 32 } : {}),
        }}
      >
        {config.profileMascot ? (
          <img src={config.profileMascot} alt="Mascot" style={{ width: 32, height: 32, borderRadius: '50%' }} />
        ) : (
          <svg width="32" height="32" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="14" cy="14" r="14" fill="#fff" />
            <path d="M8 18V10C8 8.89543 8.89543 8 10 8H18C19.1046 8 20 8.89543 20 10V16C20 17.1046 19.1046 18 18 18H10L8 20V18Z" fill={config.widgetColor} />
          </svg>
        )}
      </button>
      {open && (
        <div
          className="chat-widget-modal"
          style={{
            fontFamily: config.widgetFont,
            position: 'fixed',
            zIndex: 10000,
            width: 350,
            height: 480,
            background: '#fff',
            borderRadius: 16,
            boxShadow: '0 8px 32px rgba(0,0,0,0.18)',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            // Consistent offset for all corners
            ...(config.widgetPosition === 'bottomRight' ? { right: 32, bottom: 32 } : {}),
            ...(config.widgetPosition === 'bottomLeft' ? { left: 32, bottom: 32 } : {}),
            ...(config.widgetPosition === 'topRight' ? { right: 32, top: 32 } : {}),
            ...(config.widgetPosition === 'topLeft' ? { left: 32, top: 32 } : {}),
          }}
        >
          <div
            className="chat-widget-header"
            style={{ background: config.widgetColor }}
          >
            {config.profileMascot && (
              <img src={config.profileMascot} alt="Mascot" style={{ width: 32, height: 32, borderRadius: '50%', marginRight: 8 }} />
            )}
            {config.widgetName}
            <button style={{ background: 'none', border: 'none', color: '#fff', fontSize: 20, cursor: 'pointer', marginLeft: 'auto' }} onClick={() => setOpen(false)}>&times;</button>
          </div>
          <div className="chat-widget-body" ref={bodyRef}>
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-widget-message ${msg.sender}`}>
                <div className="chat-widget-bubble">
                  {msg.image && (
                    <img 
                      src={msg.image} 
                      alt="Shared content" 
                      style={{ 
                        maxWidth: '200px', 
                        maxHeight: '200px', 
                        borderRadius: '8px', 
                        marginBottom: '8px',
                        display: 'block'
                      }} 
                    />
                  )}
                  <div>
            {msg.text}
            {/* Show View/Edit Quote button if quote is present */}
            {msg.sender === 'bot' && (msg as any).quote && (
              <button
                style={{ marginTop: 8, background: '#667eea', color: '#fff', border: 'none', borderRadius: 6, padding: '4px 12px', cursor: 'pointer' }}
                onClick={() => navigate('/quote-editor', { state: { quote: (msg as any).quote } })}
              >
                View/Edit Quote
              </button>
            )}
                  </div>
                </div>
                {/* Enhanced feedback section for bot messages - positioned below response */}
                {msg.sender === 'bot' && msg.id && (
                  <div className="widget-feedback-section">
                    <div className="widget-feedback-buttons">
                      <button
                        className={`widget-copy-btn ${copiedMessageId === msg.id ? 'copied' : ''}`}
                        onClick={() => handleCopyMessage(msg.text, msg.id!)}
                        title="Copy message"
                      >
                        {copiedMessageId === msg.id ? '✓' : '📋'}
                      </button>
                      <button
                        className={`widget-feedback-btn thumbs-up ${msg.feedback === 'up' ? 'active' : ''}`}
                        onClick={() => handleFeedback(msg.id!, 'up')}
                        title="This was helpful"
                        disabled={msg.feedback !== null}
                      >
                        👍
                      </button>
                      <button
                        className={`widget-feedback-btn thumbs-down ${msg.feedback === 'down' ? 'active' : ''}`}
                        onClick={() => handleFeedback(msg.id!, 'down')}
                        title="This was not helpful"
                        disabled={msg.feedback !== null}
                      >
                        👎
                      </button>
                      {msg.feedback && (
                        <span className="widget-feedback-thank-you">
                          Thanks for your feedback!
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="chat-widget-message bot">
                <div className="chat-widget-bubble">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Image preview section */}
          {imagePreview && (
            <div style={{ 
              padding: '10px', 
              backgroundColor: '#f5f5f5', 
              borderTop: '1px solid #e0e0e0',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              <img 
                src={imagePreview} 
                alt="Preview" 
                style={{ 
                  width: '50px', 
                  height: '50px', 
                  objectFit: 'cover', 
                  borderRadius: '4px' 
                }} 
              />
              <span style={{ flex: 1, fontSize: '14px', color: '#666' }}>
                Image selected
              </span>
              <button 
                type="button"
                onClick={clearImage}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#999',
                  cursor: 'pointer',
                  fontSize: '18px',
                  padding: '2px'
                }}
              >
                ×
              </button>
            </div>
          )}
          
          <form className="chat-widget-input" onSubmit={handleSend}>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={selectedImage ? "Ask about your image..." : "Type your message or use voice..."}
              autoFocus
              style={{ fontFamily: config.widgetFont }}
              disabled={isLoading}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={{ display: 'none' }}
            />
            <VoiceAssistant 
              onVoiceInput={handleVoiceInput}
              isLoading={isLoading}
              disabled={false}
            />
            <button 
              type="button"
              onClick={() => fileInputRef.current?.click()}
              aria-label="Upload image"
              style={{ 
                background: 'none', 
                border: 'none', 
                color: config.widgetColor,
                cursor: 'pointer',
                fontSize: '18px',
                padding: '8px'
              }}
              disabled={isLoading}
            >
              📷
            </button>
            <button 
              type="submit" 
              aria-label="Send" 
              style={{ background: config.widgetColor }}
              disabled={isLoading || (!input.trim() && !selectedImage)}
            >
              {isLoading ? '⏳' : '➤'}
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default ChatWidget;