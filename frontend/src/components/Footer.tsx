import React from 'react';

const Footer: React.FC = () => (
  <footer style={{
    width: '100%',
    background: 'linear-gradient(90deg, #6366f1 0%, #1976d2 100%)',
    color: '#fff',
    padding: '32px 0 16px 0',
    textAlign: 'center',
    fontFamily: 'Inter, Roboto, Montserrat, Lato, Poppins, Open Sans, Nunito, Oswald, Raleway, Merriweather',
    fontWeight: 600,
    fontSize: 18,
    letterSpacing: 0.5,
    borderRadius: '32px 32px 0 0',
    boxShadow: '0 -2px 12px #6366f133',
    marginTop: 48,
    position: 'relative',
    zIndex: 2,
  }}>
    <div style={{ marginBottom: 12 }}>
      <span style={{ fontSize: 22, fontWeight: 700 }}>
        <i className="bi bi-stars" style={{ marginRight: 8 }}></i>
        Powered by AppGallop
      </span>
    </div>
    <div style={{ fontSize: 16, opacity: 0.85 }}>
      &copy; {new Date().getFullYear()} BirdCorp. All rights reserved.
    </div>
    <div style={{ marginTop: 8, fontSize: 14, opacity: 0.7 }}>
      <a href="https://birdcorp.com/privacy" style={{ color: '#fff', textDecoration: 'underline', marginRight: 16 }}>Privacy Policy</a>
      <a href="https://birdcorp.com/terms" style={{ color: '#fff', textDecoration: 'underline' }}>Terms of Service</a>
    </div>
  </footer>
);

export default Footer;
