// src/services/api.js
import axios from 'axios';

// Use absolute URL for backend API
const API_BASE = 'http://localhost:8004';

export const uploadFile = (formData) => {
  return axios.post(`${API_BASE}/api/files/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const askQuestion = (question, fileId) => {
  return axios.post(`${API_BASE}/api/qa/ask`, { question, file_id: fileId });
};

export const generateToken = (userId) => {
  return axios.post(`${API_BASE}/api/auth/token`, { user_id: userId });
};

export const startTraining = () => {
  return axios.post(`${API_BASE}/api/training/start-training/`);
};

// Knowledge Base API functions
export const getKBEntries = () => {
  return axios.get(`${API_BASE}/api/files/kb/entries`);
};

export const createKBEntry = (question, answer) => {
  return axios.post(`${API_BASE}/api/files/kb/entries`, { question, answer });
};

export const updateKBEntry = (id, question, answer) => {
  return axios.put(`${API_BASE}/api/files/kb/entries/${id}`, { question, answer });
};

export const deleteKBEntry = (id) => {
  return axios.delete(`${API_BASE}/api/files/kb/entries/${id}`);
};

// Conversations API functions
export const saveConversation = (conversation) => {
  return axios.post(`${API_BASE}/api/conversations`, conversation);
};

export const getUserConversations = (userId) => {
  return axios.get(`${API_BASE}/api/conversations/${userId}`);
};

// WebSocket Chat API
export const connectWebSocketChat = () => {
  return new WebSocket(`${API_BASE}/ws/chat`);
};

// WhatsApp Configuration API
export const configureWhatsApp = (config) => {
  return axios.post(`${API_BASE}/whatsapp/configure-whatsapp`, config);
};

// WhatsApp Message Sending API
export const sendWhatsAppMessage = (message) => {
  return axios.post(`${API_BASE}/whatsapp/send-message`, message);
};

// Authentication API functions
export const login = (email, password, role) => {
  return axios.post(`${API_BASE}/api/auth/login`, {
    email,
    password,
    role,
  });
};

// Chat session API functions
export const createChatSession = (userId) => {
  return axios.post(`${API_BASE}/api/chat/session`, { user_id: userId });
};

export const storeChatMessage = (sessionId, sender, message) => {
  return axios.post(`${API_BASE}/api/chat/message`, { session_id: sessionId, sender, message });
};

export const getChatHistory = (sessionId) => {
  return axios.get(`${API_BASE}/api/chat/history/${sessionId}`);
};

export const askStaticChat = (question) => {
  return axios.post(`${API_BASE}/api/qa/static-chat`, { question });
};

// Image processing API functions
export const processImage = (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  return axios.post(`${API_BASE}/api/image/process-image`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const analyzeImageWithQuestion = (imageFile, question) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  if (question) {
    formData.append('question', question);
  }
  
  return axios.post(`${API_BASE}/api/image/analyze-image-with-question`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const chatWithImage = (question, imageFile) => {
  const formData = new FormData();
  if (question) {
    formData.append('question', question);
  }
  if (imageFile) {
    formData.append('image', imageFile);
  }
  
  return axios.post(`${API_BASE}/api/qa/chat-multimodal`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const checkImageProcessingHealth = () => {
  return axios.get(`${API_BASE}/api/image/health`);
};

// Feedback API functions
export const submitMessageFeedback = (messageId, feedbackType, sessionId = null, comment = null) => {
  return axios.post(`${API_BASE}/api/feedback/message-feedback`, {
    message_id: messageId,
    feedback_type: feedbackType,
    session_id: sessionId,
    comment: comment
  });
};

export const getMessageFeedback = (messageId) => {
  return axios.get(`${API_BASE}/api/feedback/message-feedback/${messageId}`);
};

export const getFeedbackAnalytics = () => {
  return axios.get(`${API_BASE}/api/feedback/feedback-analytics`);
};

// Voice Assistant API functions
export const speechToText = (audioBlob, saveAudio = false) => {
  const formData = new FormData();
  formData.append('audio_file', audioBlob, 'voice_input.wav');
  formData.append('save_audio', saveAudio);
  
  return axios.post(`${API_BASE}/api/voice/speech-to-text`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000 // 30 second timeout for voice processing
  });
};

export const voiceChat = (audioBlob, saveAudio = true, sessionId = 'anonymous') => {
  const formData = new FormData();
  formData.append('audio_file', audioBlob, 'voice_chat.wav');
  formData.append('save_audio', saveAudio);
  formData.append('session_id', sessionId);
  
  return axios.post(`${API_BASE}/api/voice/voice-chat`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000 // 30 second timeout for voice processing
  });
};

export const checkVoiceHealth = () => {
  return axios.get(`${API_BASE}/api/voice/health`);
};

export const listVoiceFiles = () => {
  return axios.get(`${API_BASE}/api/voice/files`);
};
