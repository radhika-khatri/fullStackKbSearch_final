import React from 'react';
import { Navigate } from 'react-router-dom';

interface ProtectedRouteProps {
  children: React.ReactNode;
  allowedRoles?: string[];
  requireAuth?: boolean;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  allowedRoles = [], 
  requireAuth = true 
}) => {
  const token = localStorage.getItem('authToken');
  const userRole = localStorage.getItem('userRole');

  // Check if authentication is required
  if (requireAuth && !token) {
    return <Navigate to="/login" replace />;
  }

  // Check if user has the required role
  if (allowedRoles.length > 0 && !allowedRoles.includes(userRole || '')) {
    // Redirect based on user's role
    if (userRole === 'user') {
      return <Navigate to="/user-dashboard" replace />;
    } else if (userRole === 'admin') {
      return <Navigate to="/dashboard" replace />;
    }
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
