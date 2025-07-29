import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import KnowledgeBase from './pages/KnowledgeBase';
import Login from './pages/Login';
import Register from './pages/Register';
import Settings from './pages/Settings';
import ChatWidget from './components/ChatWidget';
import { WidgetConfigProvider, useWidgetConfig } from './components/WidgetConfigContext';
import UserDashboard from './pages/UserDashboard';
import Campaigns from './pages/Campaigns';
import Journeys from './pages/Journeys';
import JourneyDetail from './pages/JourneyDetail';
import Integrations from './pages/Integrations';
import Layout from './components/Layout';
import TemplatePage from './pages/TemplatePage';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider } from './contexts/AuthContext';

import QuoteEditor from './pages/QuoteEditor';
import QuotePage from './pages/QuotePage';


function ChatWidgetConditional() {
  const { config } = useWidgetConfig();
  return config.enabled ? <ChatWidget /> : null;
}

export default function App() {
  return (
    <AuthProvider>
      <WidgetConfigProvider>
        <Router>
          <div className="App">
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              {/* Admin routes */}
              <Route path="/dashboard" element={<ProtectedRoute allowedRoles={['admin']}><Layout><Dashboard /></Layout></ProtectedRoute>} />
              <Route path="/chat" element={<ProtectedRoute allowedRoles={['admin']}><Layout><Chat /></Layout></ProtectedRoute>} />
              <Route path="/knowledge-base" element={<ProtectedRoute allowedRoles={['admin']}><Layout><KnowledgeBase /></Layout></ProtectedRoute>} />
              <Route path="/settings" element={<ProtectedRoute allowedRoles={['admin']}><Layout><Settings /></Layout></ProtectedRoute>} />
              <Route path="/campaigns" element={<ProtectedRoute allowedRoles={['admin']}><Layout><Campaigns /></Layout></ProtectedRoute>} />
              <Route path="/journeys" element={<ProtectedRoute allowedRoles={['admin']}><Layout><Journeys /></Layout></ProtectedRoute>} />
              <Route path="/journeys/:id" element={<ProtectedRoute allowedRoles={['admin']}><Layout><JourneyDetail /></Layout></ProtectedRoute>} />
              <Route path="/integrations" element={<ProtectedRoute allowedRoles={['admin']}><Layout><Integrations /></Layout></ProtectedRoute>} />
              <Route path="/template" element={<ProtectedRoute allowedRoles={['admin']}><Layout><TemplatePage /></Layout></ProtectedRoute>} />
              {/* User routes */}
              <Route path="/user-dashboard" element={<ProtectedRoute allowedRoles={['user', 'admin']}><Layout><UserDashboard /></Layout></ProtectedRoute>} />
              {/* Quote Editor route (no protection, for chat widget redirect) */}
              <Route path="/quote-editor" element={<QuoteEditor />} />
              {/* Quote Page route */}
              <Route path="/quotes" element={<QuotePage />} />
              {/* Default redirect */}
              <Route path="/" element={<Login />} />
            </Routes>
            <ChatWidgetConditional />
          </div>
        </Router>
      </WidgetConfigProvider>
    </AuthProvider>
  );
}
// ...existing code...
              {/* Default redirect */}

