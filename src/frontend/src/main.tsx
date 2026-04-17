import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { AgenticBackground } from "./components/AgenticBackground";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <>
      <AgenticBackground />
      <div className="relative z-10">
        <App />
      </div>
    </>
  </React.StrictMode>
);
