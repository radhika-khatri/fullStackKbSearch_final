import React, { useState, useEffect } from 'react';
import {
  BiUser, BiNotification, BiCog, BiShield,
  BiDownload, BiTrash, BiEdit, BiSave,
  BiX, BiCheck, BiEnvelope, BiPhone,
  BiAtom, // Using BiAtom for AI, or choose another relevant icon
  BiKey // <--- ADDED BiKey HERE
} from 'react-icons/bi';
import './Settings.css';

interface UserProfile {
  name: string;
  email: string;
  phone: string;
  role: string;
  avatar: string;
}

interface NotificationSettings {
  emailNotifications: boolean;
  pushNotifications: boolean;
  smsNotifications: boolean;
  campaignUpdates: boolean;
  systemAlerts: boolean;
}

interface SystemSettings {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  autoSave: boolean;
  dataRetention: number;
}

interface AIConfigurations {
  geminiApiKey: string;
  openAiApiKey: string; // Changed from mistralApiKey to openAiApiKey
  autoApprovalLimit: number;
  aiResponseSpeed: string;
  autoQuoteGeneration: boolean;
  smartPricing: boolean;
  apiProvider: string;
  widgetColor: string;
  widgetName: string;
  profileMascot: string | null; // Stores file path or base64
  widgetPosition: string;
  widgetFont: string;
}

export default function Settings() {
  const [activeTab, setActiveTab] = useState<'profile' | 'notifications' | 'system' | 'security' | 'aiConfig'>('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');

  const [userProfile, setUserProfile] = useState<UserProfile>({
    name: 'John Doe',
    email: 'john.doe@birdcorp.com',
    phone: '+1 (555) 123-4567',
    role: 'Administrator',
    avatar: '/logo192.png'
  });

  const [notifications, setNotifications] = useState<NotificationSettings>({
    emailNotifications: true,
    pushNotifications: true,
    smsNotifications: false,
    campaignUpdates: true,
    systemAlerts: true
  });

  const [systemSettings, setSystemSettings] = useState<SystemSettings>({
    theme: 'light',
    language: 'en',
    timezone: 'UTC-5',
    autoSave: true,
    dataRetention: 30
  });

  const [aiConfigurations, setAiConfigurations] = useState<AIConfigurations>({
    geminiApiKey: '',
    openAiApiKey: '', // Changed initial state
    autoApprovalLimit: 10000,
    aiResponseSpeed: 'fast',
    autoQuoteGeneration: true,
    smartPricing: true,
    apiProvider: 'openai',
    widgetColor: '#6A5ACD', // Example default purple
    widgetName: 'AI Assistant',
    profileMascot: '/mascot.png', // Example default mascot
    widgetPosition: 'bottomRight',
    widgetFont: 'Inter'
  });

  useEffect(() => {
    // Load saved theme
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme) {
      setSystemSettings(prev => ({
        ...prev,
        theme: savedTheme === 'true' ? 'dark' : 'light'
      }));
    }
  }, []);

  const handleSave = async () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setIsEditing(false); // Only applicable for profile tab editing
      setSaveMessage('Settings saved successfully!');

      // Apply theme changes
      if (systemSettings.theme === 'dark') {
        document.body.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'true');
      } else {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'false');
      }

      setTimeout(() => setSaveMessage(''), 3000);
    }, 1000);
  };

  const handleCancel = () => {
    setIsEditing(false);
    // Reset changes (in real app, reload from server or reset state to initial values)
    // For simplicity, we won't implement a full reset here.
  };

  // --- Updated function to simulate API key test ---
  const testApiKey = async (provider: 'gemini' | 'openai', apiKey: string) => {
    setLoading(true);
    console.log(`Attempting to test ${provider.toUpperCase()} API Key: ${apiKey}`);

    // Simulate an asynchronous API call
    setTimeout(() => {
      setLoading(false);
      let isValid = false;

      if (provider === 'gemini') {
        // Simple mock validation for Gemini: check if it starts with "AIza" (common for Google API keys)
        // In a real scenario, you'd make a call to a Gemini API endpoint.
        isValid = apiKey.startsWith('AIza') && apiKey.length > 30;
      } else if (provider === 'openai') {
        // Simple mock validation for OpenAI: check if it starts with "sk-"
        // In a real scenario, you'd make a call to an OpenAI API endpoint (e.g., /v1/models).
        isValid = apiKey.startsWith('sk-') && apiKey.length > 30;
      }

      if (isValid) {
        alert(`${provider.toUpperCase()} API Key is valid!`);
        console.log(`Test for ${provider.toUpperCase()} API Key successful.`);
      } else {
        alert(`${provider.toUpperCase()} API Key is invalid. Please check the key format.`);
        console.warn(`Test for ${provider.toUpperCase()} API Key failed.`);
      }
    }, 1500); // Simulate network delay
  };

  const tabItems = [
    { key: 'profile', label: 'Profile', icon: <BiUser /> },
    { key: 'notifications', label: 'Notifications', icon: <BiNotification /> },
    { key: 'system', label: 'System', icon: <BiCog /> },
    { key: 'security', label: 'Security', icon: <BiShield /> },
    { key: 'aiConfig', label: 'AI Configurations', icon: <BiAtom /> } // New tab
  ];

  const renderProfile = () => (
    <div className="settings-section">
      <div className="section-header">
        <h3>Profile Information</h3>
        <button
          className={`edit-button ${isEditing ? 'cancel' : 'edit'}`}
          onClick={() => isEditing ? handleCancel() : setIsEditing(true)}
        >
          {isEditing ? <BiX /> : <BiEdit />}
          {isEditing ? 'Cancel' : 'Edit'}
        </button>
      </div>

      <div className="profile-content">
        <div className="avatar-section">
          <img src={userProfile.avatar} alt="Avatar" className="user-avatar" />
          {isEditing && (
            <button className="change-avatar-btn">Change Photo</button>
          )}
        </div>

        <div className="profile-fields">
          <div className="field-group">
            <label>Full Name</label>
            <input
              type="text"
              value={userProfile.name}
              disabled={!isEditing}
              onChange={(e) => setUserProfile(prev => ({ ...prev, name: e.target.value }))}
            />
          </div>

          <div className="field-group">
            <label>Email Address</label>
            <input
              type="email"
              value={userProfile.email}
              disabled={!isEditing}
              onChange={(e) => setUserProfile(prev => ({ ...prev, email: e.target.value }))}
            />
          </div>

          <div className="field-group">
            <label>Phone Number</label>
            <input
              type="tel"
              value={userProfile.phone}
              disabled={!isEditing}
              onChange={(e) => setUserProfile(prev => ({ ...prev, phone: e.target.value }))}
            />
          </div>

          <div className="field-group">
            <label>Role</label>
            <input
              type="text"
              value={userProfile.role}
              disabled
            />
          </div>
        </div>
      </div>

      {isEditing && (
        <div className="action-buttons">
          <button className="save-button" onClick={handleSave} disabled={loading}>
            <BiSave />
            {loading ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      )}
    </div>
  );

  const renderNotifications = () => (
    <div className="settings-section">
      <div className="section-header">
        <h3>Notification Preferences</h3>
      </div>

      <div className="notification-settings">
        <div className="notification-group">
          <h4>Communication</h4>

          <div className="toggle-item">
            <div className="toggle-info">
              <BiEnvelope className="toggle-icon" />
              <div>
                <span>Email Notifications</span>
                <p>Receive updates and alerts via email</p>
              </div>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={notifications.emailNotifications}
                onChange={(e) => setNotifications(prev => ({
                  ...prev,
                  emailNotifications: e.target.checked
                }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="toggle-item">
            <div className="toggle-info">
              <BiNotification className="toggle-icon" />
              <div>
                <span>Push Notifications</span>
                <p>Get real-time browser notifications</p>
              </div>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={notifications.pushNotifications}
                onChange={(e) => setNotifications(prev => ({
                  ...prev,
                  pushNotifications: e.target.checked
                }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="toggle-item">
            <div className="toggle-info">
              <BiPhone className="toggle-icon" />
              <div>
                <span>SMS Notifications</span>
                <p>Receive critical alerts via SMS</p>
              </div>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={notifications.smsNotifications}
                onChange={(e) => setNotifications(prev => ({
                  ...prev,
                  smsNotifications: e.target.checked
                }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>
        </div>

        <div className="notification-group">
          <h4>Content Updates</h4>

          <div className="toggle-item">
            <div className="toggle-info">
              <div>
                <span>Campaign Updates</span>
                <p>Notifications about campaign status changes</p>
              </div>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={notifications.campaignUpdates}
                onChange={(e) => setNotifications(prev => ({
                  ...prev,
                  campaignUpdates: e.target.checked
                }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="toggle-item">
            <div className="toggle-info">
              <div>
                <span>System Alerts</span>
                <p>Important system maintenance and updates</p>
              </div>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={notifications.systemAlerts}
                onChange={(e) => setNotifications(prev => ({
                  ...prev,
                  systemAlerts: e.target.checked
                }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>
        </div>
      </div>

      <div className="action-buttons">
        <button className="save-button" onClick={handleSave} disabled={loading}>
          <BiSave />
          {loading ? 'Saving...' : 'Save Preferences'}
        </button>
      </div>
    </div>
  );

  const renderSystem = () => (
    <div className="settings-section">
      <div className="section-header">
        <h3>System Preferences</h3>
      </div>

      <div className="system-settings">
        <div className="setting-item">
          <div className="setting-info">
            <span>Theme</span>
            <p>Choose your preferred appearance</p>
          </div>
          <select
            value={systemSettings.theme}
            onChange={(e) => setSystemSettings(prev => ({
              ...prev,
              theme: e.target.value as 'light' | 'dark' | 'auto'
            }))}
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
            <option value="auto">Auto</option>
          </select>
        </div>

        <div className="setting-item">
          <div className="setting-info">
            <span>Language</span>
            <p>Select your preferred language</p>
          </div>
          <select
            value={systemSettings.language}
            onChange={(e) => setSystemSettings(prev => ({
              ...prev,
              language: e.target.value
            }))}
          >
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
          </select>
        </div>

        <div className="setting-item">
          <div className="setting-info">
            <span>Timezone</span>
            <p>Your local timezone for scheduling</p>
          </div>
          <select
            value={systemSettings.timezone}
            onChange={(e) => setSystemSettings(prev => ({
              ...prev,
              timezone: e.target.value
            }))}
          >
            <option value="UTC-8">Pacific Time (UTC-8)</option>
            <option value="UTC-5">Eastern Time (UTC-5)</option>
            <option value="UTC+0">GMT (UTC+0)</option>
            <option value="UTC+1">Central European Time (UTC+1)</option>
          </select>
        </div>

        <div className="toggle-item">
          <div className="toggle-info">
            <div>
              <span>Auto-save</span>
              <p>Automatically save changes while editing</p>
            </div>
          </div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={systemSettings.autoSave}
              onChange={(e) => setSystemSettings(prev => ({
                ...prev,
                autoSave: e.target.checked
              }))}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>

        <div className="setting-item">
          <div className="setting-info">
            <span>Data Retention</span>
            <p>How long to keep chat history (days)</p>
          </div>
          <input
            type="number"
            min="1"
            max="365"
            value={systemSettings.dataRetention}
            onChange={(e) => setSystemSettings(prev => ({
              ...prev,
              dataRetention: parseInt(e.target.value)
            }))}
          />
        </div>
      </div>

      <div className="action-buttons">
        <button className="save-button" onClick={handleSave} disabled={loading}>
          <BiSave />
          {loading ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );

  const renderSecurity = () => (
    <div className="settings-section">
      <div className="section-header">
        <h3>Security & Privacy</h3>
      </div>

      <div className="security-settings">
        <div className="security-item">
          <div className="security-info">
            <BiKey className="security-icon" /> {/* This is where BiKey is used */}
            <div>
              <span>Change Password</span>
              <p>Update your account password</p>
            </div>
          </div>
          <button className="action-button">Change</button>
        </div>

        <div className="security-item">
          <div className="security-info">
            <BiShield className="security-icon" />
            <div>
              <span>Two-Factor Authentication</span>
              <p>Add an extra layer of security</p>
            </div>
          </div>
          <button className="action-button primary">Enable</button>
        </div>

        <div className="security-item">
          <div className="security-info">
            <BiDownload className="security-icon" />
            <div>
              <span>Download Data</span>
              <p>Export your account data</p>
            </div>
          </div>
          <button className="action-button">Download</button>
        </div>

        <div className="security-item danger">
          <div className="security-info">
            <BiTrash className="security-icon" />
            <div>
              <span>Delete Account</span>
              <p>Permanently delete your account and all data</p>
            </div>
          </div>
          <button className="action-button danger">Delete</button>
        </div>
      </div>
    </div>
  );

  const renderAIConfigurations = () => (
    <div className="settings-section">
      <div className="section-header">
        <h3>AI Platform Configurations</h3>
      </div>

      <div className="ai-config-settings">
        {/* Gemini API Key */}
        <div className="setting-item api-key-item">
          <div className="setting-info">
            <span>Gemini API Key</span>
            <p>Enter your Gemini API key</p>
          </div>
          <div className="api-key-input-group">
            <input
              type="text"
              placeholder="Enter your Gemini API key"
              value={aiConfigurations.geminiApiKey}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, geminiApiKey: e.target.value }))}
            />
            <button className="test-button" onClick={() => testApiKey('gemini', aiConfigurations.geminiApiKey)} disabled={loading}>
              Test
            </button>
          </div>
        </div>

        {/* OpenAI API Key */}
        <div className="setting-item api-key-item">
          <div className="setting-info">
            <span>OpenAI API Key</span>
            <p>Enter your OpenAI API key</p>
          </div>
          <div className="api-key-input-group">
            <input
              type="text"
              placeholder="Enter your OpenAI API key"
              value={aiConfigurations.openAiApiKey}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, openAiApiKey: e.target.value }))}
            />
            <button className="test-button" onClick={() => testApiKey('openai', aiConfigurations.openAiApiKey)} disabled={loading}>
              Test
            </button>
          </div>
        </div>

        {/* Auto Approval Limit */}
        <div className="setting-item">
          <div className="setting-info">
            <span>Auto Approval Limit</span>
            <p>Maximum amount for auto-approval ($)</p>
          </div>
          <input
            type="number"
            value={aiConfigurations.autoApprovalLimit}
            onChange={(e) => setAiConfigurations(prev => ({ ...prev, autoApprovalLimit: parseInt(e.target.value) }))}
          />
        </div>

        {/* AI Response Speed */}
        <div className="setting-item">
          <div className="setting-info">
            <span>AI Response Speed</span>
          </div>
          <select
            value={aiConfigurations.aiResponseSpeed}
            onChange={(e) => setAiConfigurations(prev => ({ ...prev, aiResponseSpeed: e.target.value }))}
          >
            <option value="fast">Fast (1-2s)</option>
            <option value="normal">Normal (3-5s)</option>
            <option value="slow">Slow (5-10s)</option>
          </select>
        </div>

        {/* Auto Quote Generation */}
        <div className="toggle-item">
          <div className="toggle-info">
            <div>
              <span>Auto Quote Generation</span>
              <p>Enable AI to automatically generate quotes</p>
            </div>
          </div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={aiConfigurations.autoQuoteGeneration}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, autoQuoteGeneration: e.target.checked }))}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>

        {/* Smart Pricing */}
        <div className="toggle-item">
          <div className="toggle-info">
            <div>
              <span>Smart Pricing</span>
              <p>Use AI for dynamic pricing recommendations</p>
            </div>
          </div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={aiConfigurations.smartPricing}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, smartPricing: e.target.checked }))}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>

        {/* API Provider - Full Width */}
        <div className="setting-item api-provider-item">
          <div className="setting-info">
            <span>API Provider</span>
            <p>Select your preferred AI service provider</p>
          </div>
          <select
            value={aiConfigurations.apiProvider}
            onChange={(e) => setAiConfigurations(prev => ({ ...prev, apiProvider: e.target.value }))}
          >
            <option value="openai">OpenAI (ChatGPT)</option>
            <option value="gemini">Google Gemini</option>
            <option value="mistral">Mistral AI</option>
          </select>
        </div>
      </div>

      {/* Widget Customization Section */}
      <div className="widget-section">
        <h3 style={{ 
          fontSize: '1.25rem', 
          marginBottom: '1.5rem', 
          color: '#333', 
          borderBottom: '2px solid #007bff', 
          paddingBottom: '0.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <BiAtom style={{ color: '#007bff' }} />
          Widget Customization
        </h3>

        <div className="widget-customization-grid">
          <div className="setting-item">
            <div className="setting-info">
              <span>Widget Color</span>
              <p>Pick your brand color</p>
            </div>
            {/* Simple color input, a more advanced picker would be a separate component */}
            <input
              type="color"
              value={aiConfigurations.widgetColor}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, widgetColor: e.target.value }))}
            />
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <span>Widget Name</span>
            </div>
            <input
              type="text"
              value={aiConfigurations.widgetName}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, widgetName: e.target.value }))}
            />
          </div>

          <div className="setting-item file-upload-item">
            <div className="setting-info">
              <span>Profile Mascot (Upload Image)</span>
            </div>
            <div className="file-upload-control">
              <input
                type="file"
                id="mascot-upload"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setAiConfigurations(prev => ({ ...prev, profileMascot: URL.createObjectURL(e.target.files[0]) }));
                  }
                }}
              />
              <label htmlFor="mascot-upload" className="choose-file-button">Choose File</label>
              <span className="file-name">{aiConfigurations.profileMascot ? 'File chosen' : 'No file chosen'}</span>
              {aiConfigurations.profileMascot && (
                <img src={aiConfigurations.profileMascot} alt="Mascot" className="mascot-preview" />
              )}
            </div>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <span>Widget Position</span>
            </div>
            <select
              value={aiConfigurations.widgetPosition}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, widgetPosition: e.target.value }))}
            >
              <option value="bottomRight">Bottom Right</option>
              <option value="bottomLeft">Bottom Left</option>
              <option value="topRight">Top Right</option>
              <option value="topLeft">Top Left</option>
            </select>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <span>Widget Font</span>
            </div>
            <select
              value={aiConfigurations.widgetFont}
              onChange={(e) => setAiConfigurations(prev => ({ ...prev, widgetFont: e.target.value }))}
            >
              <option value="Inter">Inter</option>
              <option value="Arial">Arial</option>
              <option value="Roboto">Roboto</option>
              <option value="Open Sans">Open Sans</option>
            </select>
          </div>
        </div>
      </div>

      <div className="action-buttons full-width-save">
        <button className="save-button" onClick={handleSave} disabled={loading}>
          <BiSave />
          {loading ? 'Saving...' : 'Save Widget Configuration'}
        </button>
        <button
          className="activate-button"
          style={{ marginLeft: 16, background: '#4CAF50', color: '#fff', padding: '0.5rem 1.5rem', borderRadius: 6, border: 'none', fontWeight: 600, cursor: 'pointer' }}
          onClick={async () => {
            setLoading(true);
            try {
              // Replace with actual user ID logic
              const userId = window.localStorage.getItem('userId') || '1';
              const res = await fetch(`/api/widget-config/${userId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(aiConfigurations),
              });
              if (!res.ok) throw new Error('Failed to activate widget config');
              setSaveMessage('Widget configuration activated!');
            } catch (err) {
              setSaveMessage('Failed to activate widget config');
            } finally {
              setLoading(false);
            }
          }}
          disabled={loading}
        >
          Activate
        </button>
      </div>
    </div>
  );


  const renderContent = () => {
    switch (activeTab) {
      case 'profile':
        return renderProfile();
      case 'notifications':
        return renderNotifications();
      case 'system':
        return renderSystem();
      case 'security':
        return renderSecurity();
      case 'aiConfig': // New case for AI Configurations
        return renderAIConfigurations();
      default:
        return renderProfile();
    }
  };

  return (
    <div className="settings-container">
      <div className="settings-header">
        <h1>Settings</h1>
        <p>Manage your account preferences and system settings</p>
        {saveMessage && (
          <div className="save-message">
            <BiCheck />
            {saveMessage}
          </div>
        )}
      </div>

      <div className="settings-layout">
        <div className="settings-tabs">
          {tabItems.map((tab) => (
            <button
              key={tab.key}
              className={`tab-button ${activeTab === tab.key ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.key as any)}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        <div className="settings-content">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}