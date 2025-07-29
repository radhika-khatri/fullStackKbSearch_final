import React, { useState } from 'react';
import './Register.css';
import { FaEnvelope, FaLock, FaUser } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

// You can use Boxicons or Bootstrap Icons by adding their CDN in public/index.html
// Example for Boxicons: <link href="https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css" rel="stylesheet" />
// Example for Bootstrap Icons: <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet" />

export default function Register() {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!name || !email || !password || !confirm) {
      setError('Please fill all fields.');
      return;
    }

    if (password !== confirm) {
      setError('Passwords do not match.');
      return;
    }

    try {
      const res = await fetch('http://localhost:8004/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (res.ok) {
        alert('✅ Registered! Now login.');
        navigate('/login');
      } else {
        // Handle different error response formats
        let errorMessage = 'Registration failed.';
        if (data.detail) {
          if (typeof data.detail === 'string') {
            errorMessage = data.detail;
          } else if (Array.isArray(data.detail) && data.detail.length > 0) {
            errorMessage = data.detail.join(', ');
          }
        }
        alert(`❌ ${errorMessage}`);
      }
    } catch (err) {
      alert('❌ An error occurred during registration.');
    }
  };

  const Logo = () => (
    <div className="login-logo">
      <img src="/AppgallopLG1.png" alt="AppGallop Logo" className="logo-image" />
    </div>
  );

  const WelcomeMessage = () => (
    <>
      <h2 className="login-title">Create your AppGallop account</h2>
      <p className="login-subtext">Register to access your dashboard and tools.</p>
    </>
  );

  const Benefits = () => (
    <ul className="login-benefits">
      <li><FaLock /> Secure, encrypted user access</li>
      <li><FaEnvelope /> Email-based account verification</li>
    </ul>
  );

  return (
    <div className="login-cover">
      <div className="login-left-panel">
        <Logo />
        <h1 className="brand-name">AppGallop</h1>
        <p className="brand-tagline">Powering smart conversations with AI.</p>
        <Benefits />
      </div>

      <div className="login-right-panel">
        <div className="login-card">
          <Logo />
          <WelcomeMessage />
          {error && <div className="login-error">{typeof error === 'string' ? error : 'An error occurred'}</div>}
          <form onSubmit={handleRegister} className="login-form">
            <div className="login-form-group">
              <label>Name</label>
              <div className="input-icon-wrapper">
                {/* Boxicons user icon */}
                <i className="bx bx-user icon" style={{ fontSize: 18 }}></i>
                <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Enter your name..." />
              </div>
            </div>
            <div className="login-form-group">
              <label>Email</label>
              <div className="input-icon-wrapper">
                {/* Boxicons envelope icon */}
                <i className="bx bx-envelope icon" style={{ fontSize: 18 }}></i>
                <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="Enter email..." />
              </div>
            </div>
            <div className="login-form-group">
              <label>Password</label>
              <div className="input-icon-wrapper">
                {/* Boxicons lock icon */}
                <i className="bx bx-lock icon" style={{ fontSize: 18 }}></i>
                <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter password..." />
              </div>
            </div>
            <div className="login-form-group">
              <label>Confirm Password</label>
              <div className="input-icon-wrapper">
                {/* Boxicons lock icon */}
                <i className="bx bx-lock icon" style={{ fontSize: 18 }}></i>
                <input type="password" value={confirm} onChange={e => setConfirm(e.target.value)} placeholder="Confirm password..." />
              </div>
            </div>
            <button className="login-btn" type="submit">Register</button>
          </form>
          <div className="login-footer">
            <a href="/login">Already have an account? Login</a>
          </div>
        </div>
      </div>
    </div>
  );
}
