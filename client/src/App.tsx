import { Routes, Route } from 'react-router-dom';
import "./styles/App.css";
import {PageSelectorNavBar, PageSelectorNavItem} from "./components/PageSelectorNavBar/PageSelectorNavBar";
import homeIcon from './assets/icons/home.svg';
import settingsIcon from './assets/icons/settings.svg';
import downloadIcon from './assets/icons/download.svg';
import { HomePage } from './pages/HomePage/HomePage';
import { DownloadPage } from './pages/DownloadPage/DownloadPage';

function App() {
  return (
    <div className="layout">
      <PageSelectorNavBar >
        <PageSelectorNavItem icon={homeIcon} label="homePage" href="/home" active={true} />
        <PageSelectorNavItem icon={settingsIcon} label="settingsPage" href="/settings" active={false} />
        <PageSelectorNavItem icon={downloadIcon} label="downloadPage" href="/download" active={false} />
      </PageSelectorNavBar>

      <main className="content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/download" element={<DownloadPage />} />
          <Route path="*" element={<div>404 - Page Not Found</div>} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
