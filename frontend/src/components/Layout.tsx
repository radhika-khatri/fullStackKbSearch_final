
import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';

export default function Layout({ children }) {
  const location = useLocation();
  // Exclude sidebar for user-dashboard and chatwidget routes (case-insensitive, supports subroutes)
  const path = location.pathname.toLowerCase();
  const hideSidebar =
    path.startsWith('/user-dashboard') ||
    path.startsWith('/chatwidget');

  // Sidebar state for open/collapse
  const [activeSidebar, setActiveSidebar] = useState('Dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {!hideSidebar && (
        <Sidebar
          activeSidebar={activeSidebar}
          setActiveSidebar={setActiveSidebar}
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
        />
      )}
      <main style={{ flex: 1 }}>{children}</main>
    </div>
  );
}
