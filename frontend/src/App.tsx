import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import Dashboard from "@/pages/Dashboard";
import ConfigEditorPage from "@/pages/ConfigEditorPage";
import AdvancedViewerPage from "@/pages/AdvancedViewerPage"; // NEW
import { DashboardProvider } from "@/context/DashboardContext";

export default function App() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded-md border border-transparent hover:border-slate-600 ${
      isActive ? "bg-slate-800 text-slate-100" : "text-slate-300"
    }`;

  return (
    <div className="app">
      <header className="topbar flex items-center justify-between px-4 py-2 border-b border-slate-800 bg-slate-900/70">
        <div className="brand text-slate-100 font-semibold">WaveCalling</div>
        <nav className="nav flex items-center gap-2">
          <NavLink to="/" className={linkClass} end>
            Dashboard
          </NavLink>
          {/* Optional quick link(s): */}
          {/* <NavLink to="/config" className={linkClass}>Config</NavLink> */}
          {/* <NavLink to="/viewer/advanced" className={linkClass}>Advanced</NavLink> */}
        </nav>
      </header>

      <main className="content">
        <DashboardProvider>
          <Routes>
            {/* Dashboard as default and alias at /dashboard */}
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />

            {/* Config editor page */}
            <Route path="/config" element={<ConfigEditorPage />} />

            {/* NEW: Advanced viewer */}
            <Route path="/viewer/advanced" element={<AdvancedViewerPage />} />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </DashboardProvider>
      </main>
    </div>
  );
}
