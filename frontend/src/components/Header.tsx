import React from 'react';

const Header: React.FC = () => (
  <header style={{
    position: 'sticky',
    top: 0,
    zIndex: 10,
    background: 'rgba(255,255,255,0.85)',
    boxShadow: '0 2px 12px #6366f133',
    borderRadius: '0 0 32px 32px',
    padding: '24px 32px 12px 32px',
    marginBottom: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  }}>
    <h1 style={{
      fontSize: 40,
      fontWeight: 900,
      background: 'linear-gradient(90deg, #6366f1 0%, #1976d2 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      letterSpacing: 1.2,
      textShadow: '0 2px 8px #6366f133',
      margin: 0,
      fontFamily: 'Inter, Roboto, Montserrat, Lato, Poppins, Open Sans, Nunito, Oswald, Raleway, Merriweather',
    }}>Chatbot Dashboard</h1>
    <button
      style={{
        background: 'linear-gradient(90deg, #6366f1 0%, #1976d2 100%)',
        color: '#fff',
        border: 'none',
        borderRadius: 50,
        padding: '12px 24px',
        fontWeight: 700,
        fontSize: 18,
        boxShadow: '0 2px 8px #6366f133',
        cursor: 'pointer',
        transition: 'background 0.2s',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
      }}
      title="Get Help"
    >
      <i className="bi bi-question-circle-fill" style={{ fontSize: 22 }}></i>
      Help
    </button>
  </header>
);

export default Header;
