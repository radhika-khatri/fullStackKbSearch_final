// src/pages/Dashboard.tsx
import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  BiBarChartAlt2, BiUser, BiRocket, BiGroup, BiEnvelope,
  BiCog, BiCube, BiLayer, BiChevronRight, BiCheckCircle,
  BiErrorCircle, BiTimeFive, BiMenu, BiX, BiMoon, BiSun, BiLogOut
} from "react-icons/bi";

import ChartPanel from "./ChartPanel";
import "./Dashboard.css";
import InsightCards from "../components/InsightCards";
import { useAuth } from "../contexts/AuthContext";

const kpiData = [
  { label: "Campaigns", value: "25", icon: <BiRocket /> },
  { label: "Recipients", value: "72,910", icon: <BiUser /> },
  { label: "Sent", value: "56,910", icon: <BiEnvelope /> },
  { label: "Delivery rate", value: "74.5%", icon: <BiCheckCircle /> },
  { label: "Click through rate", value: "24.5%", icon: <BiBarChartAlt2 /> },
];

const summaryData = [
  { title: "Total revenue", value: "$512,375", change: "+2.5%" },
  { title: "Attributed revenue", value: "$486,375", change: "+10.5%" },
  { title: "Per recipient", value: "$121.36" },
  { title: "Email", value: "$61.28 / 46.84%" },
  { title: "SMS", value: "$15.21 / 12.33%" },
  { title: "WhatsApp", value: "$2.12 / 2.21%" },
];

const campaignRows = [
  { name: "Bird promotional offers", status: "Sent", type: "WhatsApp", delivers: "99.6%", revenue: "$10,235.11" },
  { name: "Festive Fiesta Frenzy", status: "Sent", type: "WhatsApp", delivers: "85.8%", revenue: "$12,894.36" },
  { name: "Celebrate with a Splash", status: "Failed", type: "Email", delivers: "—", revenue: "—" },
  { name: "Winter Wonderland Bonanza", status: "Draft", type: "Email", delivers: "—", revenue: "—" },
  { name: "Autumn Adventure Awaits", status: "Sent", type: "WhatsApp", delivers: "92.2%", revenue: "$2,645.99" },
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<"Overview" | "Analytics">("Overview");
  const [darkMode, setDarkMode] = useState(false);
  const navigate = useNavigate();
  const { logout } = useAuth();

  useEffect(() => {
    const storedTheme = localStorage.getItem("darkMode");
    setDarkMode(storedTheme === "true");
  }, []);

  useEffect(() => {
    document.body.className = darkMode ? "dark-mode" : "";
    localStorage.setItem("darkMode", String(darkMode));
  }, [darkMode]);

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  return (
    <div className="user-dashboard-container">
      <div className="dashboard-header-wrapper">
        <div className="dashboard-header">
          <div className="header-left">
            <h1 className="dashboard-title">AppGallop Dashboard</h1>            
          </div>
          <div className="header-right">
            {/* Tabs with icons - moved to header */}
            
            <button className="logout-button" title="Logout" onClick={handleLogout}>
              <BiLogOut />
              Logout
            </button>
          </div>
        </div>
      </div>

      <div className="dashboard-content-wrapper">

        <div className="content-header" style={{ marginBottom: "32px" }}>
          <h2 className="dashboard-subtitle">Monitor your campaigns and analytics with ease.</h2>
          
          <div className="tabs-container">
            <button
              onClick={() => setActiveTab("Overview")}
              className={`tab-button ${activeTab === "Overview" ? 'active' : ''}`}
              title="Overview"
            >
              <i className="bi bi-bar-chart"></i>
              Overview
            </button>
            <button
              onClick={() => setActiveTab("Analytics")}
              className={`tab-button ${activeTab === "Analytics" ? 'active' : ''}`}
              title="Analytics"
            >
              <i className="bi bi-graph-up"></i>
              Analytics
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="stats-cards-grid">
          {kpiData.map((kpi, idx) => (
            <div key={idx} className="stat-card-large">
              <div className="stat-card-label">
                {kpi.icon}
                {kpi.label}
              </div>
              <div className="stat-card-value">{kpi.value}</div>
            </div>
          ))}
        </div>

        {/* Main Content Card */}
        <div className="main-content-card">
          {activeTab === "Overview" && (
            <>
              <section className="insight-section">
                <InsightCards />
              </section>

              <section className="summary-section">
                <div className="insight-header">
                  <h2 className="insight-heading">Business Performance Summary</h2>
                </div>
                <div className="summary-grid insight-grid">
                  {summaryData.map((item) => (
                    <div className="insight-card" key={item.title}>
                      <div className="insight-details">
                        <h4>{item.title}</h4>
                        <p>{item.value}</p>
                        {item.change && <p className="summary-change">({item.change})</p>}
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="campaign-table">
                <h3>Top performing campaigns</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Status</th>
                      <th>Type</th>
                      <th>Delivers</th>
                      <th>Revenue</th>
                    </tr>
                  </thead>
                  <tbody>
                    {campaignRows.map((row) => (
                      <tr key={row.name}>
                        <td>{row.name}</td>
                        <td>
                          <span className={`status ${row.status.toLowerCase()}`}>
                            {row.status === "Sent" && <BiCheckCircle />}
                            {row.status === "Failed" && <BiErrorCircle />}
                            {row.status === "Draft" && <BiTimeFive />}
                            {row.status}
                          </span>
                        </td>
                        <td>{row.type}</td>
                        <td>{row.delivers}</td>
                        <td>{row.revenue}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            </>
          )}

          {activeTab === "Analytics" && (
            <section className="analytics-section">
              <h2>Analytics</h2>
              <ChartPanel />
            </section>
          )}
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
}

function KPI({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="kpi-card glass">
      <div className="kpi-icon">{icon}</div>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  );
}
