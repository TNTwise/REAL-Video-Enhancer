import { Outlet, useNavigate } from "react-router-dom";
import { Home, Cpu, Settings, Download } from "lucide-react";
import Dock from "../components/Dock/Dock";

const dockItems = [
  { icon: <Home size={18} />, label: 'Home', onClick: () => {} },
  { icon: <Cpu size={18} />, label: 'System', onClick: () => {} },
  { icon: <Settings size={18} />, label: 'Settings', onClick: () => {} },
  { icon: <Download size={18} />, label: 'Downloads', onClick: () => {} },
];

export default function RootLayout() {
  const navigate = useNavigate();

  const items = dockItems.map(item => ({
    ...item,
    onClick: item.label === 'Home' ? () => navigate('/home')
      : item.label === 'Downloads' ? () => navigate('/download')
      : item.onClick
  }));

  return (
    <div>
      <Dock items={items} />
      <Outlet />
    </div>
  );
}
