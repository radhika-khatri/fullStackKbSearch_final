import React, { useState } from 'react';
import './Login.css';
import { FaEnvelope, FaLock, FaBolt, FaChartLine, FaFish, FaShieldAlt, FaRocket } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [role, setRole] = useState('admin');
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email || !password || !role) {
      setError('Please enter email, password and role.');
      return;
    }

    setLoading(true);
    try {
      const success = await login(email, password, role);
      setLoading(false);

      if (success) {
        // Redirect based on role
        if (role === 'admin') {
          navigate('/dashboard');
        } else {
          navigate('/user-dashboard');
        }
      } else {
        setError('Login failed. Please check your credentials.');
      }
    } catch (err) {
      setError('An error occurred during login.');
      setLoading(false);
    }
  };

  const Logo = () => (
    <div className="login-logo">
      <img src="/AppgallopLG1.png" alt="AppGallop Logo" className="logo-image" />
    </div>
  );

  const WelcomeMessage = () => (
    <>
      <h2 className="login-title">Welcome to AppGallop</h2>
      <p className="login-subtext">Access your intelligent platform and manage your business operations with ease.</p>
    </>
  );

  const Benefits = () => (
    <div className="login-benefits">
      <div className="benefit-item">
        <FaRocket className="benefit-icon" />
        <div>
          <h4>Smart Solutions</h4>
          <p>Intelligent business tools powered by advanced technology</p>
        </div>
      </div>
      <div className="benefit-item">
        <FaShieldAlt className="benefit-icon" />
        <div>
          <h4>Secure Platform</h4>
          <p>Enterprise-grade security for your business data</p>
        </div>
      </div>
      <div className="benefit-item">
        <FaChartLine className="benefit-icon" />
        <div>
          <h4>Fast Performance</h4>
          <p>Get instant results with our optimized platform</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="login-cover">
      <div className="login-left-panel">
        <div className="brand-section">
          <Logo />
          <h1 className="brand-name">AppGallop</h1>
          <p className="brand-tagline">Intelligent business solutions at your fingertips</p>
        </div>
        <Benefits />
        <div className="decorative-elements">
          <div className="floating-shape shape-1"></div>
          <div className="floating-shape shape-2"></div>
          <div className="floating-shape shape-3"></div>
        </div>
      </div>

      <div className="login-right-panel">
        <div className="login-card">
          <WelcomeMessage />
          {error && <div className="login-error">{typeof error === 'string' ? error : 'An error occurred'}</div>}
          <form onSubmit={async (e) => {
            e.preventDefault();
            setLoading(true);
            try {
              const res = await fetch('http://localhost:8004/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, role }),
              });

              const data = await res.json();

              if (res.ok) {
                localStorage.setItem('authToken', data.access_token);
                localStorage.setItem('userRole', data.role);
                localStorage.setItem('userData', JSON.stringify(data.user));
                
                // Role-based redirection
                if (role === 'user') {
                  navigate('/user-dashboard');
                } else if (role === 'admin') {
                  navigate('/dashboard');
                } else {
                  navigate('/dashboard'); // Default fallback
                }
              } else {
                let errorMessage = 'Login failed.';
                if (data.detail) {
                  errorMessage = typeof data.detail === 'string' ? data.detail : data.detail.join(', ');
                }
                alert(`❌ ${errorMessage}`);
              }
            } catch (err) {
              alert('❌ An error occurred during login.');
            } finally {
              setLoading(false);
            }
          }} className="login-form">
            <div className="login-form-group">
              <label>Email Address</label>
              <div className="input-icon-wrapper">
                <FaEnvelope className="icon" />
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="Enter your email address"
                />
              </div>
            </div>

            <div className="login-form-group">
              <label>Password</label>
              <div className="input-icon-wrapper">
                <FaLock className="icon" />
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter your password"
                />
              </div>
            </div>

            <div className="role-selection">
              <span className="role-label">Login as:</span>
              <div className="role-options">
                <label className={`role-option ${role === 'admin' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="role"
                    value="admin"
                    checked={role === 'admin'}
                    onChange={() => setRole('admin')}
                  />
                  <span className="role-text">Admin</span>
                </label>
                <label className={`role-option ${role === 'user' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="role"
                    value="user"
                    checked={role === 'user'}
                    onChange={() => setRole('user')}
                  />
                  <span className="role-text">User</span>
                </label>
              </div>
            </div>

            <button className="login-btn" type="submit" disabled={loading}>
              {loading ? (
                <span className="loading-text">
                  <span className="spinner"></span>
                  Signing in...
                </span>
              ) : (
                'Sign In'
              )}
            </button>
          </form>
          
          <div className="login-footer">
            <p>Don't have an account? <a href="/register">Create one here</a></p>
          </div>
        </div>
      </div>
    </div>
  );
}
