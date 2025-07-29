import React, { useState, useEffect } from 'react';
import {
  BiUser, BiNotification, BiSun, BiMoon, BiShield, 
  BiGlobe, BiDownload, BiTrash, BiEdit, BiSave,
  BiX, BiCheck, BiCog, BiKey, BiEnvelope, BiPhone
} from 'react-icons/bi';
import './Settings.css';

export default function Settings() {
  const [activeTab, setActiveTab] = useState('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');

  const [userProfile, setUserProfile] = useState({
    name: 'John Doe',
    email: 'john.doe@birdcorp.com',
    phone: '+1 (555) 123-4567',
    role: 'Administrator',
    avatar: '/logo192.png'
  });

  const [notifications, setNotifications] = useState({
    emailNotifications: true,
    pushNotifications: true,
    smsNotifications: false,
    campaignUpdates: true,
    systemAlerts: true
  });

  const [systemSettings, setSystemSettings] = useState({
    theme: 'light',
    language: 'en',
    timezone: 'UTC-5',
    autoSave: true,
    dataRetention: 30
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
      setIsEditing(false);
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
    // Reset changes (in real app, reload from server)
  };

  const tabItems = [
    { key: 'profile', label: 'Profile', icon: React.createElement(BiUser) },
    { key: 'notifications', label: 'Notifications', icon: React.createElement(BiNotification) },
    { key: 'system', label: 'System', icon: React.createElement(BiCog) },
    { key: 'security', label: 'Security', icon: React.createElement(BiShield) }
  ];

  const renderProfile = () => (
    <div className="settings-section">
      <div className="section-header">
        <h3>Profile Information</h3>
        <button
          className={`edit-button ${isEditing ? 'cancel' : 'edit'}`}
          onClick={() => isEditing ? handleCancel() : setIsEditing(true)}
        >
          {isEditing ? React.createElement(BiX) : React.createElement(BiEdit)}
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
            {React.createElement(BiSave)}
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
              {React.createElement(BiEnvelope, { className: "toggle-icon" })}
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
              {React.createElement(BiNotification, { className: "toggle-icon" })}
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
              {React.createElement(BiPhone, { className: "toggle-icon" })}
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
          {React.createElement(BiSave)}
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
              theme: e.target.value 
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
          {React.createElement(BiSave)}
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
            {React.createElement(BiKey, { className: "security-icon" })}
            <div>
              <span>Change Password</span>
              <p>Update your account password</p>
            </div>
          </div>
          <button className="action-button">Change</button>
        </div>

        <div className="security-item">
          <div className="security-info">
            {React.createElement(BiShield, { className: "security-icon" })}
            <div>
              <span>Two-Factor Authentication</span>
              <p>Add an extra layer of security</p>
            </div>
          </div>
          <button className="action-button primary">Enable</button>
        </div>

        <div className="security-item">
          <div className="security-info">
            {React.createElement(BiDownload, { className: "security-icon" })}
            <div>
              <span>Download Data</span>
              <p>Export your account data</p>
            </div>
          </div>
          <button className="action-button">Download</button>
        </div>

        <div className="security-item danger">
          <div className="security-info">
            {React.createElement(BiTrash, { className: "security-icon" })}
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
            {React.createElement(BiCheck)}
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
              onClick={() => setActiveTab(tab.key)}
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
