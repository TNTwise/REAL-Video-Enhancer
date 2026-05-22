import { useState } from "react";
import rveLogo from "./assets/logo-v2.svg";
import { invoke } from "@tauri-apps/api/core";
import "./styles/App.css";
import {PageSelector} from "./pages/PageSelector/PageSelector";

function App() {
  return (
    <div className="layout">
        <PageSelector />
        <main className="content">
      </main>
    </div>
  );
}

export default App;
