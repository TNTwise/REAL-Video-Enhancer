import { useState } from "react";
import rveLogo from "./assets/logo-v2.svg";
import { invoke } from "@tauri-apps/api/core";
import "./styles/App.css";
import PageSelectorNavBar from "./components/PageSelectorNavBar/PageSelectorNavBar";

function App() {
  return (
    <div className="App">
      <PageSelectorNavBar />
      <h1>
        REAL Video Enhancer
      </h1>
      <img src={rveLogo} className="logo" alt="REAL Video Enhancer logo" />
    </div>
  );
}

export default App;
