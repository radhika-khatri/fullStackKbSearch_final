// UserDashboard.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWidgetConfig, WidgetFont, WidgetPosition } from '../components/WidgetConfigContext';
import { uploadFile } from '../services/api';
import './UserDashboard.css'; // Import the CSS file

// Dashboard stats
const stats = [
  { label: 'Total Messages', value: 1250, icon: 'bi-chat-dots-fill' },
  { label: 'Sessions', value: 320, icon: 'bi-people-fill' },
  { label: 'Avg Response', value: '1.2s', icon: 'bi-stopwatch-fill' },
  { label: 'Satisfaction', value: '4.2/5', icon: 'bi-star-fill' },
];

const tabs = [
  { label: 'Widget Settings', icon: 'bi-gear-fill' },
  { label: 'Knowledge Base', icon: 'bi-journal-richtext' },
  { label: 'Analytics', icon: 'bi-graph-up' },
  { label: 'Installation', icon: 'bi-cloud-arrow-down-fill' },
];

interface WidgetConfig {
  enabled: boolean;
  widgetColor: string;
  widgetPosition: WidgetPosition;
  widgetFont: WidgetFont;
  profileMascot: string | null;
  widgetName: string;
}

const UserDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const { config, setConfig } = useWidgetConfig();
  const [activeTab, setActiveTab] = useState(0);
  const [copied, setCopied] = useState(false);
  const [mascotFile, setMascotFile] = useState<File | null>(null);
  const [mascotPreview, setMascotPreview] = useState<string | null>(config.profileMascot);

  useEffect(() => {
    if (config.profileMascot && config.profileMascot !== mascotPreview) {
      setMascotPreview(config.profileMascot);
    }
  }, [config.profileMascot]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    try {
      await uploadFile(file);
      alert('File uploaded successfully!');
      setFile(null);
    } catch (err) {
      alert('Upload failed.');
    }
  };

  const installInstructions = [
    'Copy the installation script below.',
    'Paste it just before the </body> tag of your website.',
    'Save and deploy your website.',
    'The AI widget will appear on your site automatically.'
  ];

  let installScript = `<script>
  window.AppGallopWidget = {
  "enabled": true,
  "name": "AI Assistant",
  "greeting": "Hello! How can I help you today?",
  "color": "${config.widgetColor}",
  "position": "${config.widgetPosition.replace('bottom', 'bottom-').replace('top', 'top-').toLowerCase()}",
  "font": "${config.widgetFont}",
  "mascot": "${config.profileMascot}",
  "hasKnowledgeBase": false,
  "apiUrl": "http://127.0.0.1:5001"
};
</script>
<script src="http://127.0.0.1:5001/widget.js"></script>`;

  const handleCopy = () => {
    navigator.clipboard.writeText(installScript);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleMascotChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setMascotFile(file);
      const reader = new FileReader();
      reader.onload = ev => setMascotPreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleSaveConfig = () => {
    setConfig(cfg => ({
      ...cfg,
      profileMascot: mascotPreview || cfg.profileMascot,
    }));
    alert('Widget configuration saved!');
  };

  const handleLogout = () => {
    // Clear any authentication tokens or user data
    localStorage.removeItem("authToken");
    localStorage.removeItem("userRole");
    localStorage.removeItem("userData");
    
    // Navigate to login page
    navigate("/");
  };

  const handleWidgetPositionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPosition = e.target.value as WidgetPosition;
    setConfig(cfg => {
      const updatedConfig = { ...cfg, widgetPosition: newPosition };
      updateInstallScript(updatedConfig);
      return updatedConfig;
    });
  };

  const updateInstallScript = (config: WidgetConfig) => {
    installScript = `<script>
    window.AppGallopWidget = {
    "enabled": true,
    "name": "AI Assistant",
    "greeting": "Hello! How can I help you today?",
    "color": "${config.widgetColor}",
    "position": "${config.widgetPosition.replace('bottom', 'bottom-').replace('top', 'top-').toLowerCase()}",
    "font": "${config.widgetFont}",
    "mascot": "${config.profileMascot}",
    "hasKnowledgeBase": false,
    "apiUrl": "http://127.0.0.1:5001"
  };
  </script>
  <script src="http://127.0.0.1:5001/widget.js"></script>`;
  };

  const renderWidgetSettings = () => (
    <div className="dashboard-card widget-settings">
      <h2 className="card-title">Widget Settings</h2>
      <div className="settings-grid">
        <div className="setting-item">
          <label className="setting-label">Widget Status</label>
          <button
            onClick={() => setConfig(cfg => ({ ...cfg, enabled: !cfg.enabled }))}
            className={`status-button ${config.enabled ? 'active' : 'inactive'}`}
          >
            <i className={`bi ${config.enabled ? 'bi-toggle-on' : 'bi-toggle-off'}`}></i>
            {config.enabled ? 'Deactivate' : 'Activate'}
          </button>
        </div>

        <div className="setting-item">
          <label className="setting-label">Widget Color</label>
          <input
            type="color"
            value={config.widgetColor}
            onChange={e => setConfig(cfg => ({ ...cfg, widgetColor: e.target.value }))}
            className="color-input"
          />
        </div>

        <div className="setting-item">
          <label className="setting-label">Widget Name</label>
          <input
            type="text"
            value={config.widgetName}
            onChange={e => setConfig(cfg => ({ ...cfg, widgetName: e.target.value }))}
            className="text-input"
          />
        </div>

        <div className="setting-item">
          <label className="setting-label">Widget Font</label>
          <select
            value={config.widgetFont}
            onChange={e => setConfig(cfg => ({ ...cfg, widgetFont: e.target.value as WidgetFont }))}
            className="select-input"
          >
            {['Inter', 'Roboto', 'Montserrat', 'Lato', 'Poppins', 'Open Sans', 'Nunito', 'Oswald', 'Raleway', 'Merriweather'].map(font => (
              <option key={font} value={font}>{font}</option>
            ))}
          </select>
        </div>

        <div className="setting-item">
          <label className="setting-label">Widget Position</label>
          <select
            value={config.widgetPosition}
            onChange={handleWidgetPositionChange}
            className="select-input"
          >
            <option value="bottomRight">Bottom Right</option>
            <option value="bottomLeft">Bottom Left</option>
            <option value="topRight">Top Right</option>
            <option value="topLeft">Top Left</option>
          </select>
        </div>

        <div className="setting-item">
          <label className="setting-label">Profile Mascot (Upload Image)</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleMascotChange}
            className="file-input"
          />
          {mascotPreview && (
            <img src={mascotPreview} alt="Mascot Preview" className="mascot-preview" />
          )}
        </div>
      </div>
      <button onClick={handleSaveConfig} className="save-config-button">
        Save Widget Configuration
      </button>
    </div>
  );

  const renderKnowledgeBase = () => (
    <div className="dashboard-card knowledge-base">
      <h2 className="card-title">
        <i className="bi bi-database-fill me-3"></i>
        Knowledge Base Management
      </h2>

      <div className="upload-section">
        <i className="bi bi-upload upload-icon"></i>
        <h3 className="upload-title">Upload Any Data File</h3>
        <p className="upload-description">
          Upload any file format - PDF, Excel, Word, CSV, JSON, TXT, and more!<br />
          Our AI will automatically extract and structure the data for your chatbot.
        </p>

        <div className="file-type-icons">
          {[
            { label: 'PDF', icon: 'bi-file-earmark-pdf-fill', color: '#e53935' },
            { label: 'Excel', icon: 'bi-file-earmark-excel-fill', color: '#2e7d32' },
            { label: 'Word', icon: 'bi-file-earmark-word-fill', color: '#1565c0' },
            { label: 'CSV', icon: 'bi-file-earmark-spreadsheet-fill', color: '#fbc02d' },
            { label: 'JSON', icon: 'bi-file-earmark-code-fill', color: '#00acc1' },
            { label: 'TXT', icon: 'bi-file-earmark-text-fill', color: '#616161' },
            { label: 'Images', icon: 'bi-image-fill', color: '#7b1fa2' },
            { label: 'More', icon: 'bi-plus-circle', color: '#424242' },
          ].map((item, i) => (
            <div key={i} className="file-type-item">
              <i className={`bi ${item.icon}`} style={{ color: item.color }}></i>
              <span>{item.label}</span>
            </div>
          ))}
        </div>

        <input
          type="file"
          id="kb-upload"
          className="hidden-file-input"
          onChange={handleFileChange}
        />
        <button
          className="upload-button"
          onClick={() => document.getElementById('kb-upload')?.click()}
        >
          <i className="bi bi-cloud-arrow-up-fill"></i> Select File to Upload
        </button>

        <p className="upload-info">
          Max file size: 50MB • All formats supported • Data is processed securely
        </p>

        {file && (
          <div className="selected-file-info">
            Selected File: <span>{file.name}</span>
            <button
              onClick={handleUpload}
              className="upload-now-button"
            >
              Upload Now
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderAnalytics = () => (
    <div className="dashboard-card analytics-section">
      <h2 className="card-title">Analytics</h2>
      <div className="stats-grid">
        {stats.map((stat, idx) => (
          <div key={idx} className="stat-card">
            <i className={`bi ${stat.icon} stat-icon`}></i>
            <div className="stat-label">{stat.label}</div>
            <div className="stat-value">{stat.value}</div>
          </div>
        ))}
      </div>

      <div className="chart-section">
        <h3 className="chart-title">Message Activity (Last 7 Days)</h3>
        <div className="bar-chart">
          {[5, 8, 3, 10, 6, 12, 7].map((val, idx) => (
            <div key={idx} className="bar-container" title={`${val} messages on ${['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][idx]}`}>
              <div className="bar" style={{ height: `${val * 12}px` }}></div>
              <span className="bar-label">{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][idx]}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="sessions-table-section">
        <h3 className="table-title">Recent Sessions</h3>
        <table className="sessions-table">
          <thead>
            <tr>
              <th>Session ID</th>
              <th>User</th>
              <th>Messages</th>
              <th>Satisfaction</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            {[
              { id: '#1001', user: 'John Doe', messages: 5, satisfaction: '4/5', date: '2025-07-10', satColor: 'satisfaction-high' },
              { id: '#1002', user: 'Jane Smith', messages: 3, satisfaction: '5/5', date: '2025-07-09', satColor: 'satisfaction-perfect' },
              { id: '#1003', user: 'Alex Lee', messages: 2, satisfaction: '3/5', date: '2025-07-08', satColor: 'satisfaction-medium' },
              { id: '#1004', user: 'Sarah Chen', messages: 7, satisfaction: '4.5/5', date: '2025-07-07', satColor: 'satisfaction-high' },
              { id: '#1005', user: 'David Kim', messages: 4, satisfaction: '5/5', date: '2025-07-06', satColor: 'satisfaction-perfect' },
              { id: '#1006', user: 'Emily White', messages: 6, satisfaction: '4/5', date: '2025-07-05', satColor: 'satisfaction-high' },
            ].map((session, idx) => (
              <tr key={idx}>
                <td>{session.id}</td>
                <td>{session.user}</td>
                <td>{session.messages}</td>
                <td className={session.satColor}>{session.satisfaction}</td>
                <td>{session.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const renderInstallation = () => (
    <div className="dashboard-card installation-section">
      <h2 className="card-title">Install Widget</h2>
      <div className="installation-guide">
        <h3 className="guide-title">Installation Instructions</h3>
        <div className="guide-content">
          <p className="guide-intro"><b>Step-by-Step Installation Guide</b></p>
          <ol className="instruction-list">
            {installInstructions.map((step, idx) => (
              <li key={idx}>{step}</li>
            ))}
          </ol>
          <p className="guide-note">
            Note: You can install the widget on any website platform (HTML, WordPress, Shopify, React, etc.) by following these steps. For advanced integration, see the developer documentation.
          </p>
        </div>
        <h3 className="guide-title">Installation Script</h3>
        <div className="script-block">
          <pre>{installScript}</pre>
        </div>
        <button onClick={handleCopy} className={`copy-button ${copied ? 'copied' : ''}`}>
          <i className={`bi ${copied ? 'bi-check-lg' : 'bi-clipboard'}`}></i>
          {copied ? 'Copied!' : 'Copy Script'}
        </button>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 0:
        return renderWidgetSettings();
      case 1:
        return renderKnowledgeBase();
      case 2:
        return renderAnalytics();
      case 3:
        return renderInstallation();
      default:
        return null;
    }
  };

  return (
    <div className="user-dashboard-container">
      <div className="dashboard-header-wrapper">
        <div className="dashboard-header">
          <h1 className="dashboard-title">AppGallop Dashboard</h1>
          <button className="logout-button" title="Logout" onClick={handleLogout}>
            <i className="bi bi-box-arrow-right"></i>
            Logout
          </button>
        </div>
      </div>

      <div className="dashboard-content-wrapper">
        <p className="dashboard-subtitle">Configure and monitor your AI assistant with ease.</p>

        {/* Stats Cards */}
        <div className="stats-cards-grid">
          {stats.map((stat, idx) => (
            <div key={idx} className="stat-card-large">
              <div className="stat-card-label">
                <i className={`bi ${stat.icon}`}></i>
                {stat.label}
              </div>
              <div className="stat-card-value">{stat.value}</div>
            </div>
          ))}
        </div>

        {/* Tabs with icons */}
        <div className="tabs-container">
          {tabs.map((tab, idx) => (
            <button
              key={tab.label}
              onClick={() => setActiveTab(idx)}
              className={`tab-button ${activeTab === idx ? 'active' : ''}`}
              title={tab.label}
            >
              <i className={`bi ${tab.icon}`}></i>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Main Content Card */}
        <div className="main-content-card">
          {renderTabContent()}
        </div>
      </div>

      {/* Subtle background shapes for glassmorphism effect */}
      <div className="background-shapes">
        <svg width="100%" height="100%" viewBox="0 0 1400 900" fill="none" xmlns="http://www.w3.org/2000/svg" className="svg-background">
          <circle cx="1100" cy="150" r="200" fill="#6366f1" opacity="0.06" />
          <circle cx="300" cy="800" r="250" fill="#1976d2" opacity="0.05" />
          <rect x="50" y="50" width="150" height="150" rx="30" fill="#6366f1" opacity="0.03" transform="rotate(25 50 50)" />
          <rect x="1000" y="600" width="180" height="180" rx="40" fill="#1976d2" opacity="0.04" transform="rotate(-40 1000 600)" />
        </svg>
      </div>
    </div>
  );
};

export default UserDashboard;