import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="nav-content">
        <NavLink to="/" className="logo">
          🔐 StegoSystem
        </NavLink>
        <div className="nav-links">
          <NavLink 
            to="/embed" 
            className={location.pathname === '/embed' || location.pathname === '/' ? 'active' : ''}
          >
            Embed Secret
          </NavLink>
          <NavLink 
            to="/extract" 
            className={location.pathname === '/extract' ? 'active' : ''}
          >
            Extract Secret
          </NavLink>
          <NavLink 
            to="/stegnogan" 
            className={location.pathname === '/stegnogan' ? 'active' : ''}
          >
            {/* StegnoGAN++ */}
          </NavLink>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
