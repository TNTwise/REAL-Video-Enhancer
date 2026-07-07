import React, { useState } from "react";
import { Home, Cpu, Settings, Download } from "lucide-react";

export default function GlassTabBarUI() {
  const [activeTab, setActiveTab] = useState("home");

  const navItems = [
    { id: "home", icon: Home, label: "Home" },
    { id: "cpu", icon: Cpu, label: "System" },
    { id: "settings", icon: Settings, label: "Settings" },
    { id: "download", icon: Download, label: "Downloads" },
  ];

  return (
    <div className="flex h-screen w-full items-center justify-center bg-[#16191d] p-6">
      {/* Main Container Card */}
      <div className="relative flex h-[500px] w-[700px] flex-col items-center justify-end rounded-[32px] bg-gradient-to-b from-[#2c313c] via-[#1f232a] to-[#16191d] p-6 shadow-2xl border border-[#343b47]/30">
        {/* Floating Glassmorphic Tab Bar */}
        <div className="mb-2 flex w-full max-w-md items-center justify-around rounded-full border border-[#676e7b]/20 bg-[#1f232a]/40 px-6 py-3 backdrop-blur-md shadow-lg">
          {navItems.map((item) => {
            const IconComponent = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className="group relative p-2 transition-all duration-200"
                aria-label={item.label}
              >
                <IconComponent
                  size={24}
                  strokeWidth={1.5}
                  className={`transition-colors duration-200 ${
                    isActive
                      ? "text-[#fff]"
                      : "text-[#838ea2] hover:text-[#fff]"
                  }`}
                />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
