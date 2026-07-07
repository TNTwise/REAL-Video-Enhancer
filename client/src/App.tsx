import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import "./styles/App.css";
import { HomePage } from './pages/HomePage/HomePage';
import { DownloadPage } from './pages/DownloadPage/DownloadPage';
import RootLayout from './layouts/RootLayout';
import { getBackendUrl } from './lib/api';

function App() {
  useEffect(() => {
    getBackendUrl().then((url) => {
      console.log(`Backend URL: ${url}`);
    });
  }, []);

  return (
    <div className="layout">

      <main className="content">
        <Routes>
          <Route path="/" element={<RootLayout />} >
            <Route index element={<HomePage />} />
            <Route path="/home" element={<HomePage />} />
            <Route path="/download" element={<DownloadPage />} />
            <Route path="*" element={<div>404 - Page Not Found</div>} />
          </Route>
        </Routes>
      </main>
    </div>
  );
}

export default App;
