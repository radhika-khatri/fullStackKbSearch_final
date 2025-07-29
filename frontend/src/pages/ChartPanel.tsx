// src/pages/ChartPanel.tsx
import React, { useState } from 'react';
import {
  Chart as ChartJS,
  LineElement,
  ArcElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
} from 'chart.js';
import { Line, Pie } from 'react-chartjs-2';
import './ChartPanel.css';

ChartJS.register(
  LineElement,
  ArcElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
);

const revenueLineData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  datasets: [
    {
      label: 'Revenue',
      data: [12000, 19000, 30000, 25000, 32000, 37000],
      fill: true,
      backgroundColor: 'rgba(29, 161, 242, 0.2)',
      borderColor: '#1DA1F2',
      tension: 0.3,
    },
  ],
};

const campaignPieData = {
  labels: ['Email', 'SMS', 'WhatsApp'],
  datasets: [
    {
      label: 'Channels',
      data: [45, 25, 30],
      backgroundColor: ['#4db5ff', '#ff9f40', '#36a2eb'],
      borderColor: ['#fff', '#fff', '#fff'],
      borderWidth: 1,
    },
  ],
};

const audienceLineData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  datasets: [
    {
      label: 'Audience Growth',
      data: [800, 1200, 1800, 2400, 2900, 3600],
      fill: false,
      borderColor: '#00C49F',
      tension: 0.3,
    },
  ],
};

const clickRatePieData = {
  labels: ['Clicked', 'Not Clicked'],
  datasets: [
    {
      label: 'Click Rate',
      data: [60, 40],
      backgroundColor: ['#845EC2', '#D5D5D5'],
      borderColor: ['#fff', '#fff'],
      borderWidth: 1,
    },
  ],
};

export default function ChartPanel() {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          padding: 15,
          usePointStyle: true,
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        cornerRadius: 8,
        padding: 12,
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false,
        },
        ticks: {
          font: {
            size: 11,
          },
        },
      },
      y: {
        display: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          font: {
            size: 11,
          },
        },
      },
    },
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: {
          padding: 15,
          usePointStyle: true,
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        cornerRadius: 8,
        padding: 12,
      },
    },
  };

  return (
    <div className="charts-container">
      <div className="charts-header">
        <h2>Analytics Overview</h2>
        <p>Comprehensive view of your campaign performance</p>
      </div>
      
      <div className="chart-grid">
        <div className="chart-box">
          <div className="chart-header">
            <h4>Monthly Revenue</h4>
            <span className="chart-subtitle">Trending upward</span>
          </div>
          <div className="chart-content">
            <Line data={revenueLineData} options={chartOptions} />
          </div>
        </div>

        <div className="chart-box">
          <div className="chart-header">
            <h4>Campaign Distribution</h4>
            <span className="chart-subtitle">By channel</span>
          </div>
          <div className="chart-content">
            <Pie data={campaignPieData} options={pieOptions} />
          </div>
        </div>

        <div className="chart-box">
          <div className="chart-header">
            <h4>Audience Growth</h4>
            <span className="chart-subtitle">Monthly increase</span>
          </div>
          <div className="chart-content">
            <Line data={audienceLineData} options={chartOptions} />
          </div>
        </div>

        <div className="chart-box">
          <div className="chart-header">
            <h4>Click Rate</h4>
            <span className="chart-subtitle">Engagement ratio</span>
          </div>
          <div className="chart-content">
            <Pie data={clickRatePieData} options={pieOptions} />
          </div>
        </div>
      </div>
    </div>
  );
}
