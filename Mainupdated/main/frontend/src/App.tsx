import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import EmbedPage from './components/EmbedPage';
import ExtractPage from './components/ExtractPage';

import Navbar from './components/Navbar';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main className="container">
          <Routes>
            <Route path="/" element={<EmbedPage />} />
            <Route path="/embed" element={<EmbedPage />} />
            <Route path="/extract" element={<ExtractPage />} />
            
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
