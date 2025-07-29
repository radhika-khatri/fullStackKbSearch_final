import React, { useState } from "react";
import { 
  BiMessageRounded, BiUser, BiBarChart, BiCog, BiCheckCircle, 
  BiErrorCircle, BiPhone, BiGlobe, BiPowerOff, BiShield, BiTime, BiTrendingUp,
  BiSend, BiDownload, BiUpload, BiBell, BiShow, BiChat, BiVideoRecording,
  BiFile, BiCalendar, BiStar, BiHeart, BiShareAlt, BiFilter, BiSearch,
  BiDotsVertical, BiPlus, BiEdit, BiTrash, BiCopy, BiArchive, BiBookmark,
  BiFlag, BiQrScan, BiLink, BiBot, BiHeadphone, BiVolumeFull, BiVolumeMute,
  BiPlayCircle, BiPauseCircle, BiRefresh, BiChevronRight, BiChevronLeft,
  BiChevronDown, BiChevronUp, BiMicrophone, BiImage, BiPaperclip, BiSmile,
  BiLike, BiDislike, BiUserCheck, BiUserX, BiUserPlus, BiUserMinus,
  BiPulse, BiLineChart, BiDollarCircle, BiCrosshair, BiTrophy, BiTrendingDown,
  BiLayer, BiData, BiServer, BiCloud, BiWifi, BiWifiOff, BiLock, BiLockOpen,
  BiKey, BiShieldAlt, BiShieldX, BiError, BiInfoCircle, BiCheck, BiX, BiHelpCircle,
  BiPlay, BiPause, BiStop
} from "react-icons/bi";

import "./Integrations.css";
import axios from 'axios';

const Integrations = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [autoReply, setAutoReply] = useState(true);
  const [businessHours, setBusinessHours] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");
  const [templates, setTemplates] = useState([
    { id: 1, name: "Welcome Message", content: "Hi! Welcome to our business. How can we help you today?", status: "active" },
    { id: 2, name: "Thank You", content: "Thank you for contacting us! We'll get back to you soon.", status: "active" },
    { id: 3, name: "Business Hours", content: "We're currently closed. Our business hours are 9 AM - 6 PM.", status: "inactive" }
  ]);
  const [newTemplate, setNewTemplate] = useState({ name: "", content: "" });
  const [isLoading, setIsLoading] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");

  const showToast = (message: string, type: 'success' | 'error' = 'success') => {
    // Simple toast notification implementation
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
      toast.remove();
    }, 3000);
  };

  const handleConnect = async () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsConnected(true);
      setIsLoading(false);
      showToast("Connected Successfully! Your WhatsApp Business account is now connected.");
    }, 2000);
  };

  const handleDisconnect = () => {
    setIsConnected(false);
    showToast("Disconnected. WhatsApp Business account has been disconnected.", 'error');
  };

  const addTemplate = () => {
    if (newTemplate.name && newTemplate.content) {
      setTemplates([...templates, {
        id: templates.length + 1,
        name: newTemplate.name,
        content: newTemplate.content,
        status: "active"
      }]);
      setNewTemplate({ name: "", content: "" });
      showToast("Template Added. New message template has been created successfully.");
    }
  };

  const deleteTemplate = (id: number) => {
    setTemplates(templates.filter(t => t.id !== id));
    showToast("Template Deleted. Message template has been removed.", 'error');
  };

  const renderOverviewTab = () => (
    <div className="tab-content">
      <div className="content-grid">
        {/* Connection Status */}
        <div className="integration-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiPhone />
              </div>
              Connection Status
            </div>
            <p className="card-description">
              Current status of your WhatsApp Business integration
            </p>
          </div>
          <div className="card-content">
            {!isConnected ? (
              <div className="connection-setup">
                <div className="setup-icon">
                  <BiPhone />
                </div>
                <h3>Connect Your WhatsApp</h3>
                <p>Start by connecting your WhatsApp Business account to enable powerful messaging features and automation</p>
                <button 
                  onClick={handleConnect} 
                  disabled={isLoading}
                  className="btn btn-primary btn-large"
                >
                  {isLoading ? (
                    <>
                      <BiRefresh className="btn-icon spinning" />
                      Connecting...
                    </>
                  ) : (
                    <>
                      <BiMessageRounded className="btn-icon" />
                      Connect WhatsApp Business
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="connection-success">
                <div className="success-banner">
                  <div className="success-icon">
                    <BiCheckCircle />
                  </div>
                  <div>
                    <p className="success-title">Connected Successfully</p>
                    <p className="success-subtitle">WhatsApp Business is ready for use</p>
                  </div>
                </div>
                <div className="connection-details">
                  <div className="detail-row">
                    <span>Business Name:</span>
                    <span>Your Business</span>
                  </div>
                  <div className="detail-row">
                    <span>Phone Number:</span>
                    <span>+1 (555) 123-4567</span>
                  </div>
                  <div className="detail-row">
                    <span>Account Status:</span>
                    <span className="status-badge active">
                      <BiPulse />
                      Active
                    </span>
                  </div>
                  <div className="detail-row">
                    <span>API Version:</span>
                    <span>v16.0</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Features */}
        <div className="integration-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiPowerOff />
              </div>
              Integration Features
            </div>
            <p className="card-description">
              Powerful features to enhance your WhatsApp communication
            </p>
          </div>
          <div className="card-content">
            <div className="features-list">
              {[
                { icon: BiBot, title: "AI Auto-Reply", desc: "Intelligent responses using AI", available: true },
                { icon: BiBarChart, title: "Advanced Analytics", desc: "Real-time performance insights", available: true },
                { icon: BiShield, title: "Enterprise Security", desc: "End-to-end encrypted messaging", available: true },
                { icon: BiGlobe, title: "Multi-language Support", desc: "Communicate in 50+ languages", available: true },
                { icon: BiQrScan, title: "QR Code Generation", desc: "Generate WhatsApp QR codes", available: false },
                { icon: BiHeadphone, title: "Voice Messages", desc: "Send and receive voice notes", available: false }
              ].map((feature, index) => (
                <div key={index} className={`feature-item ${feature.available ? 'available' : 'unavailable'}`}>
                  <div className="feature-icon">
                    <feature.icon />
                  </div>
                  <div className="feature-content">
                    <div className="feature-title">
                      {feature.title}
                      <span className={`feature-badge ${feature.available ? 'active' : 'coming-soon'}`}>
                        {feature.available ? 'Active' : 'Coming Soon'}
                      </span>
                    </div>
                    <p className="feature-description">{feature.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="integration-card">
        <div className="card-header">
          <div className="card-title">
            <div className="icon-wrapper primary">
              <BiPowerOff />
            </div>
            Quick Actions
          </div>
          <p className="card-description">
            Common tasks and shortcuts for your WhatsApp integration
          </p>
        </div>
        <div className="card-content">
          <div className="quick-actions">
            {[
              { icon: BiSend, title: "Send Broadcast", desc: "Send message to multiple contacts" },
              { icon: BiUser, title: "Manage Contacts", desc: "Organize your contact list" },
              { icon: BiFile, title: "Message Templates", desc: "Create reusable templates" },
              { icon: BiBarChart, title: "View Analytics", desc: "Check performance metrics" }
            ].map((action, index) => (
              <button 
                key={index}
                className="quick-action-btn"
                disabled={!isConnected}
              >
                <action.icon className="action-icon" />
                <div className="action-content">
                  <div className="action-title">{action.title}</div>
                  <div className="action-description">{action.desc}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

// --- WhatsApp Setup Tab with Backend Integration ---
const [waConfig, setWaConfig] = useState({
  client_id: '',
  phone_id: '',
  token: '',
  verify_token: ''
});
const [waConfigLoading, setWaConfigLoading] = useState(false);
const [waConfigSuccess, setWaConfigSuccess] = useState(false);
const [waConfigError, setWaConfigError] = useState('');

const handleWaConfigChange = (e) => {
  setWaConfig({ ...waConfig, [e.target.name]: e.target.value });
};

// Generate 32-character random token
const generateRandomToken = () => {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < 32; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
};

const generateClientId = () => {
  const newClientId = generateRandomToken();
  setWaConfig({ ...waConfig, client_id: newClientId });
  showToast('Client ID generated successfully!', 'success');
};

const generateVerifyToken = () => {
  const newVerifyToken = generateRandomToken();
  setWaConfig({ ...waConfig, verify_token: newVerifyToken });
  showToast('Verify Token generated successfully!', 'success');
};

const copyToClipboard = async (text: string, fieldName: string) => {
  try {
    await navigator.clipboard.writeText(text);
    showToast(`${fieldName} copied to clipboard!`, 'success');
  } catch (err) {
    showToast('Failed to copy to clipboard', 'error');
  }
};

const handleWaConfigSubmit = async (e) => {
  e.preventDefault();
  setWaConfigLoading(true);
  setWaConfigError('');
  setWaConfigSuccess(false);
  try {
    const res = await axios.post('/api/routes/whatsapp/configure-whatsapp', waConfig);
    setWaConfigSuccess(true);
    showToast('WhatsApp configuration saved!', 'success');
    setIsConnected(true);
  } catch (err) {
    setWaConfigError('Failed to save WhatsApp config.');
    showToast('Failed to save WhatsApp config.', 'error');
  } finally {
    setWaConfigLoading(false);
  }
};

const renderSetupTab = () => (
  <div className="tab-content">
    <div className="integration-card">
      <div className="card-header">
        <div className="card-title">
          <div className="icon-wrapper primary">
            <BiCog />
          </div>
          WhatsApp Business Setup
        </div>
        <p className="card-description">
          Follow these steps to connect your WhatsApp Business account
        </p>
      </div>
      <div className="card-content">
        <div className="setup-steps">
          {/* Step 1: Create Account */}
          <div className="setup-step">
            <div className={`step-number ${isConnected ? 'completed' : 'pending'}`}>{isConnected ? <BiCheckCircle /> : 1}</div>
            <div className="step-content">
              <div className="step-header">
                <h3>Create WhatsApp Business Account</h3>
                {isConnected && <span className="completed-badge">Completed</span>}
              </div>
              <p>Register your business with WhatsApp Business API platform</p>
            </div>
          </div>
          {/* Step 2: Configure API Credentials */}
          <div className="setup-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <div className="step-header">
                <h3>Configure API Credentials</h3>
              </div>
              <p>Enter your API credentials and business information</p>
              <form className="credentials-form" onSubmit={handleWaConfigSubmit} autoComplete="off">
                <div className="form-row">
                  <div className="form-group">
                    <label>Client ID</label>
                    <div className="input-with-button">
                      <input name="client_id" type="text" placeholder="Enter your Client ID" value={waConfig.client_id} onChange={handleWaConfigChange} required />
                      <button type="button" className="btn btn-generate" onClick={generateClientId} title="Generate Random Client ID">
                        <BiRefresh />
                        Generate
                      </button>
                      {waConfig.client_id && (
                        <button type="button" className="btn btn-copy" onClick={() => copyToClipboard(waConfig.client_id, 'Client ID')} title="Copy Client ID">
                          <BiCopy />
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="form-group">
                    <label>Phone ID</label>
                    <input name="phone_id" type="text" placeholder="Enter your Phone ID" value={waConfig.phone_id} onChange={handleWaConfigChange} required />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>API Token</label>
                    <input name="token" type="password" placeholder="Enter your API Token" value={waConfig.token} onChange={handleWaConfigChange} required autoComplete="current-password" />
                  </div>
                  <div className="form-group">
                    <label>Verify Token</label>
                    <div className="input-with-button">
                      <input name="verify_token" type="text" placeholder="Enter your Verify Token" value={waConfig.verify_token} onChange={handleWaConfigChange} required />
                      <button type="button" className="btn btn-generate" onClick={generateVerifyToken} title="Generate Random Verify Token">
                        <BiRefresh />
                        Generate
                      </button>
                      {waConfig.verify_token && (
                        <button type="button" className="btn btn-copy" onClick={() => copyToClipboard(waConfig.verify_token, 'Verify Token')} title="Copy Verify Token">
                          <BiCopy />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                {waConfigError && <div className="form-error">{waConfigError}</div>}
                {waConfigSuccess && <div className="form-success">Configuration saved!</div>}
                <button className="btn btn-primary" type="submit" disabled={waConfigLoading}>
                  {waConfigLoading ? 'Saving...' : 'Save Configuration'}
                </button>
              </form>
            </div>
          </div>
          {/* Step 3: Test Connection */}
          <div className="setup-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <div className="step-header">
                <h3>Test Connection</h3>
              </div>
              <p>Verify that your integration is working correctly</p>
              {isConnected && <span className="completed-badge">Connected</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
);
//end renderSetupTab

  const renderTemplatesTab = () => (
    <div className="tab-content">
      <div className="content-grid">
        {/* Template Management */}
        <div className="integration-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiFile />
              </div>
              Message Templates
            </div>
            <p className="card-description">
              Manage your reusable message templates
            </p>
          </div>
          <div className="card-content">
            <div className="templates-list">
              {templates.map((template) => (
                <div key={template.id} className="template-item">
                  <div className="template-header">
                    <h4>{template.name}</h4>
                    <div className="template-actions">
                      <span className={`status-badge ${template.status}`}>
                        {template.status}
                      </span>
                      <button className="btn-icon" onClick={() => deleteTemplate(template.id)}>
                        <BiTrash />
                      </button>
                    </div>
                  </div>
                  <p className="template-content">{template.content}</p>
                  <div className="template-buttons">
                    <button className="btn btn-outline btn-small">
                      <BiEdit />
                      Edit
                    </button>
                    <button className="btn btn-outline btn-small">
                      <BiCopy />
                      Copy
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Add New Template */}
        <div className="integration-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiPlus />
              </div>
              Create New Template
            </div>
            <p className="card-description">
              Add a new message template for quick responses
            </p>
          </div>
          <div className="card-content">
            <div className="template-form">
              <div className="form-group">
                <label>Template Name</label>
                <input 
                  type="text"
                  placeholder="e.g., Welcome Message"
                  value={newTemplate.name}
                  onChange={(e) => setNewTemplate({...newTemplate, name: e.target.value})}
                />
              </div>
              <div className="form-group">
                <label>Message Content</label>
                <textarea 
                  placeholder="Enter your message template here..."
                  rows={4}
                  value={newTemplate.content}
                  onChange={(e) => setNewTemplate({...newTemplate, content: e.target.value})}
                />
              </div>
              <div className="form-group">
                <label>Template Category</label>
                <select>
                  <option value="">Select a category</option>
                  <option value="welcome">Welcome Messages</option>
                  <option value="support">Customer Support</option>
                  <option value="marketing">Marketing</option>
                  <option value="transactional">Transactional</option>
                </select>
              </div>
              <button 
                onClick={addTemplate}
                className="btn btn-primary"
                disabled={!newTemplate.name || !newTemplate.content}
              >
                <BiPlus className="btn-icon" />
                Add Template
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSettingsTab = () => (
    <div className="tab-content">
      <div className="content-grid">
        {/* Automation Settings */}
        <div className="integration-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiBot />
              </div>
              Automation Settings
            </div>
            <p className="card-description">
              Configure automated responses and behaviors
            </p>
          </div>
          <div className="card-content">
            <div className="settings-list">
              <div className="setting-item">
                <div className="setting-content">
                  <div className="setting-title">Auto-Reply Messages</div>
                  <p className="setting-description">
                    Automatically respond to incoming messages
                  </p>
                </div>
                <label className="switch">
                  <input 
                    type="checkbox" 
                    checked={autoReply} 
                    onChange={(e) => setAutoReply(e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>

              <div className="setting-item">
                <div className="setting-content">
                  <div className="setting-title">Business Hours Only</div>
                  <p className="setting-description">
                    Only send auto-replies during business hours
                  </p>
                </div>
                <label className="switch">
                  <input 
                    type="checkbox" 
                    checked={businessHours} 
                    onChange={(e) => setBusinessHours(e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>

              <div className="setting-item">
                <div className="setting-content">
                  <div className="setting-title">AI-Powered Responses</div>
                  <p className="setting-description">
                    Use AI to generate contextual responses
                  </p>
                </div>
                <label className="switch">
                  <input type="checkbox" />
                  <span className="slider"></span>
                </label>
              </div>
            </div>

            <div className="form-group">
              <label>Welcome Message</label>
              <textarea 
                placeholder="Hi! Thanks for contacting us. How can we help you today?"
                rows={3}
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Business Hours Start</label>
                <input type="time" defaultValue="09:00" />
              </div>
              <div className="form-group">
                <label>Business Hours End</label>
                <input type="time" defaultValue="18:00" />
              </div>
            </div>
          </div>
        </div>

        {/* Notification Settings */}
        <div className="integration-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiBell />
              </div>
              Notification Settings
            </div>
            <p className="card-description">
              Control when and how you receive notifications
            </p>
          </div>
          <div className="card-content">
            <div className="settings-list">
              <div className="setting-item">
                <div className="setting-content">
                  <div className="setting-title">Email Notifications</div>
                  <p className="setting-description">
                    Receive email alerts for new messages
                  </p>
                </div>
                <label className="switch">
                  <input 
                    type="checkbox" 
                    checked={notifications} 
                    onChange={(e) => setNotifications(e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>

              <div className="setting-item">
                <div className="setting-content">
                  <div className="setting-title">Push Notifications</div>
                  <p className="setting-description">
                    Receive browser push notifications
                  </p>
                </div>
                <label className="switch">
                  <input type="checkbox" />
                  <span className="slider"></span>
                </label>
              </div>

              <div className="setting-item">
                <div className="setting-content">
                  <div className="setting-title">SMS Notifications</div>
                  <p className="setting-description">
                    Receive SMS alerts for urgent messages
                  </p>
                </div>
                <label className="switch">
                  <input type="checkbox" />
                  <span className="slider"></span>
                </label>
              </div>
            </div>

            <div className="form-group">
              <label>Notification Email</label>
              <input 
                type="email"
                placeholder="admin@yourbusiness.com"
              />
            </div>

            <div className="form-group">
              <label>Notification Frequency</label>
              <select>
                <option value="instant">Instant</option>
                <option value="hourly">Hourly</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAnalyticsTab = () => (
    <div className="tab-content">
      <div className="content-grid">
        {/* Performance Metrics */}
        <div className="integration-card analytics-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiLineChart />
              </div>
              Performance Analytics
            </div>
            <p className="card-description">
              Detailed insights into your WhatsApp messaging performance
            </p>
          </div>
          <div className="card-content">
            <div className="analytics-grid">
              <div className="analytics-item">
                <p className="analytics-label">Messages Sent Today</p>
                <p className="analytics-value">147</p>
                <div className="analytics-change positive">
                  <BiTrendingUp />
                  <span>+23% vs yesterday</span>
                </div>
              </div>
              <div className="analytics-item">
                <p className="analytics-label">Response Rate</p>
                <p className="analytics-value">94.2%</p>
                <div className="analytics-change positive">
                  <BiTrendingUp />
                  <span>+5.2% vs last week</span>
                </div>
              </div>
            </div>
            
            <div className="progress-items">
              <div className="progress-item">
                <div className="progress-header">
                  <span>Message Delivery Rate</span>
                  <span>98.5%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{width: '98.5%'}}></div>
                </div>
              </div>

              <div className="progress-item">
                <div className="progress-header">
                  <span>Customer Satisfaction</span>
                  <span>4.8/5.0</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{width: '96%'}}></div>
                </div>
              </div>

              <div className="progress-item">
                <div className="progress-header">
                  <span>Template Usage</span>
                  <span>78%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{width: '78%'}}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="integration-card">
          <div className="card-header">
            <div className="card-title">
              <div className="icon-wrapper primary">
                <BiCrosshair />
              </div>
              Quick Stats
            </div>
          </div>
          <div className="card-content">
            <div className="quick-stats">
              {[
                { label: "Peak Hour", value: "2:00 PM", icon: BiTime },
                { label: "Top Template", value: "Welcome Msg", icon: BiStar },
                { label: "Avg Session", value: "8.2 min", icon: BiPulse },
                { label: "Conversion Rate", value: "12.4%", icon: BiTrendingUp },
                { label: "Total Revenue", value: "$3,247", icon: BiDollarCircle },
                { label: "Customer Rating", value: "4.8★", icon: BiTrophy }
              ].map((stat, index) => (
                <div key={index} className="stat-item">
                  <div className="stat-icon">
                    <stat.icon />
                  </div>
                  <div className="stat-content">
                    <span className="stat-label">{stat.label}</span>
                    <span className="stat-value">{stat.value}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="integration-card">
        <div className="card-header">
          <div className="card-title">
            <div className="icon-wrapper primary">
              <BiPulse />
            </div>
            Recent Activity
          </div>
          <p className="card-description">
            Latest interactions and system events
          </p>
        </div>
        <div className="card-content">
          <div className="activity-list">
            {[
              { type: "message", user: "John Doe", action: "sent a message", time: "2 min ago", status: "delivered" },
              { type: "template", user: "System", action: "used Welcome template", time: "5 min ago", status: "success" },
              { type: "contact", user: "Jane Smith", action: "started conversation", time: "12 min ago", status: "active" },
              { type: "error", user: "System", action: "failed to deliver message", time: "1 hour ago", status: "error" },
              { type: "analytics", user: "System", action: "generated daily report", time: "2 hours ago", status: "completed" }
            ].map((activity, index) => (
              <div key={index} className="activity-item">
                <div className={`activity-icon ${activity.status}`}>
                  {activity.type === 'message' && <BiMessageRounded />}
                  {activity.type === 'template' && <BiFile />}
                  {activity.type === 'contact' && <BiUser />}
                  {activity.type === 'error' && <BiErrorCircle />}
                  {activity.type === 'analytics' && <BiBarChart />}
                </div>
                <div className="activity-content">
                  <p>
                    <strong>{activity.user}</strong> {activity.action}
                  </p>
                  <p className="activity-time">{activity.time}</p>
                </div>
                <span className={`status-badge ${activity.status}`}>
                  {activity.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="settings-container">
      <div className="settings-header">
        <h1>Integrations</h1>
        <p>Connect and manage your WhatsApp Business communications</p>
        {saveMessage && (
          <div className="save-message">
            <BiCheck />
            {saveMessage}
          </div>
        )}
      </div>

      <div className="settings-layout">
        <div className="settings-tabs">
          {[
            { id: 'overview', label: 'Overview', icon: <BiMessageRounded /> },
            { id: 'setup', label: 'Setup', icon: <BiCog /> },
            { id: 'templates', label: 'Templates', icon: <BiFile /> },
            { id: 'settings', label: 'Settings', icon: <BiShield /> },
            { id: 'analytics', label: 'Analytics', icon: <BiBarChart /> }
          ].map(tab => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        <div className="settings-content">
          {activeTab === 'overview' && renderOverviewTab()}
          {activeTab === 'setup' && renderSetupTab()}
          {activeTab === 'templates' && renderTemplatesTab()}
          {activeTab === 'settings' && renderSettingsTab()}
          {activeTab === 'analytics' && renderAnalyticsTab()}
        </div>
      </div>
    </div>
  );
};

export default Integrations;
