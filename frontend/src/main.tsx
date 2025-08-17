import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ApiProvider } from "@/context/ApiContext";
import App from "./App";
import "./styles.css";

const root = createRoot(document.getElementById("root")!);

root.render(
  <React.StrictMode>
    <ApiProvider>
      <BrowserRouter basename="/ui">
        <App />
      </BrowserRouter>
    </ApiProvider>
  </React.StrictMode>
);
