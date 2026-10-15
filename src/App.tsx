import { Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Chat from "./pages/Chat";
import Retrieval from "./pages/Retrieval";
import SystemHealth from "./pages/SystemHealth";
import Evaluation from "./pages/Evaluation";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/documents" element={<Documents />} />
      <Route path="/chat" element={<Chat />} />
      <Route path="/retrieval" element={<Retrieval />} />
      <Route path="/health" element={<SystemHealth />} />
      <Route path="/evaluation" element={<Evaluation />} />
      <Route path="/settings" element={<Settings />} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}
