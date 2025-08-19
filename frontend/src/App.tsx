import { Link, Route, Routes, Navigate, useLocation } from "react-router-dom";
import UploadRun from "@/pages/UploadRun";
import RunsList from "@/pages/RunLists";
import Viewer from "@/pages/Viewer";

export default function App() {
  const loc = useLocation();
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">WaveCalling</div>
        <nav className="nav">
          <Link className={loc.pathname === "/runs" ? "active" : ""} to="/runs">Runs</Link>
          <Link className={loc.pathname === "/upload" ? "active" : ""} to="/upload">New run</Link>
          <Link className={loc.pathname.startsWith("/viewer") ? "active" : ""} to="/viewer">Viewer</Link>
        </nav>
      </header>

      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/upload" replace />} />
          <Route path="/upload" element={<UploadRun />} />
          <Route path="/runs" element={<RunsList />} />
          <Route path="/viewer" element={<Viewer />} />
          <Route path="*" element={<div>Not found</div>} />
        </Routes>
      </main>
    </div>
  );
}
