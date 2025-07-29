// src/pages/Campaigns.tsx
import React, { useState, type JSX } from 'react';
import './campaigns.css';
import ViewToggle from '../components/ViewToggle';
import { exportToCSV, exportToPDF } from '../utils/exportUtils';
import {
  BarChart3, MailOpen, Coins, BarChart, DollarSign,
  Users, Settings, ArrowUpRight
} from 'lucide-react';

interface Stat {
  label: string;
  value: string;
  sub: string;
  icon: React.ReactNode; // or even `any` but like… 😬
  bar?: number;
}

interface Campaign {
  name: string;
  status: string;
  date: string;
  openRate: string;
  clickRate: string;
}

export default function Campaigns() {
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');

  const stats: Stat[] = [
    { label: 'Open Rate', value: '85.8%', sub: '1,846 recipients', icon: <MailOpen />, bar: 85.8 },
    { label: 'Click Rate', value: '59.1%', sub: '1,846 recipients', icon: <BarChart3 />, bar: 59.1 },
    { label: 'Placed Order Rate', value: '21.4%', sub: '1,846 recipients', icon: <Coins />, bar: 21.4 },
    { label: 'Revenue', value: '$12,894.36', sub: 'AOV: $1,485 · Per recip.: $5.37', icon: <DollarSign /> },
  ];

  // Generate more realistic engagement data with better distribution
  const engagementData: number[] = Array.from({ length: 29 }, (_, i) => {
    const baseValue = 40 + Math.sin(i * 0.5) * 20;
    const randomVariation = Math.random() * 30;
    return Math.max(20, Math.floor(baseValue + randomVariation));
  });
  
  const colors = ['#4f46e5', '#7c3aed', '#06b6d4', '#10b981', '#f59e0b'];

  const recentCampaigns: Campaign[] = [
    { name: 'Festive Fiesta Frenzy', status: 'Sent', date: 'Sep 15, 2025', openRate: '89%', clickRate: '64%' },
    { name: 'Winter Wonderland Bonanza', status: 'Draft', date: 'Sep 25, 2025', openRate: '-', clickRate: '-' },
    { name: 'Autumn Adventure Awaits', status: 'Scheduled', date: 'Oct 05, 2025', openRate: '-', clickRate: '-' },
  ];

  const toggleView = () => setViewMode(viewMode === 'chart' ? 'table' : 'chart');

  return (
    <div className="campaigns-wrapper">
      <div className="campaigns-header">
        <div>
          <h1 className="campaigns-title">📊 Campaign Analytics</h1>
          <div className="last-updated">Updated: July 18, 2025 at 10:48 AM</div>
        </div>
        <div className="campaigns-controls">
          <button className="export-btn" onClick={() => exportToCSV(recentCampaigns, 'campaigns')}>
            Export CSV
          </button>
          <button className="export-btn" onClick={() => exportToPDF('campaigns-section', 'campaigns')}>
            Export PDF
          </button>
          <ViewToggle currentView={viewMode} onToggle={toggleView} />
        </div>
      </div>

      <div className="campaigns-stats">
        {stats.map((stat, i) => (
          <div className="stat-card" key={i}>
            <div className="icon-box">{stat.icon}</div>
            <div className="stat-content">
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
              <div className="stat-sub">{stat.sub}</div>
              {stat.bar !== undefined && (
                <div className="progress-bar">
                  <div className="progress" style={{ width: `${stat.bar}%` }}></div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div id="campaigns-section">
        {viewMode === 'chart' ? (
          <div className="engagement-chart">
            <div className="chart-header">
              <BarChart />
              <span>Engagement Over Time</span>
            </div>
            <div className="bar-container">
              {engagementData.map((val, i) => (
                <div
                  key={i}
                  className="bar-wrapper"
                  title={`Day ${i + 1}: ${val}% engagement`}
                >
                  <div
                    className="bar"
                    style={{
                      height: `${Math.max(val * 3, 60)}px`,
                      background: `linear-gradient(135deg, ${colors[i % colors.length]}, ${colors[(i + 1) % colors.length]})`
                    }}
                  />
                  <span className="bar-label">{i + 1}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="campaigns-summary">
            <h2 className="summary-title">📋 Recent Campaigns</h2>
            <table className="campaigns-table">
              <thead>
                <tr>
                  <th>Campaign Name</th>
                  <th>Status</th>
                  <th>Launch Date</th>
                  <th>Open Rate</th>
                  <th>Click Rate</th>
                </tr>
              </thead>
              <tbody>
                {recentCampaigns.map((c, i) => (
                  <tr key={i}>
                    <td><strong>{c.name}</strong></td>
                    <td><span className={`status-tag ${c.status.toLowerCase()}`}>{c.status}</span></td>
                    <td>{c.date}</td>
                    <td><strong>{c.openRate}</strong></td>
                    <td><strong>{c.clickRate}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="campaign-actions">
        <div className="action-tile">
          <Users />
          <div>
            <strong>Audience Segments</strong>
            <p>View and manage your audience lists.</p>
          </div>
          <ArrowUpRight />
        </div>
        <div className="action-tile">
          <Settings />
          <div>
            <strong>Campaign Settings</strong>
            <p>Customize your delivery preferences.</p>
          </div>
          <ArrowUpRight />
        </div>
      </div>
    </div>
  );
}