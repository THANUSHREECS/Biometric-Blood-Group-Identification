import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { ResearchDisclaimer } from './components/ResearchDisclaimer';
import { DashboardView } from './views/DashboardView';
import { ParticipantRegistrationView } from './views/ParticipantRegistrationView';
import { ScannerConnectionView } from './views/ScannerConnectionView';
import { FingerprintCaptureView } from './views/FingerprintCaptureView';
import { QualityReviewView } from './views/QualityReviewView';
import { ExperimentalAnalysisView } from './views/ExperimentalAnalysisView';
import { ResearchResultsView } from './views/ResearchResultsView';
import { SessionHistoryView } from './views/SessionHistoryView';
import { ModelInfoView } from './views/ModelInfoView';
import { apiClient } from './api/client';
import type {
  ScreenId,
  ResearchSession,
  InferenceResult,
  SystemStatusResponse,
  ScannerDeviceInfo
} from './types';
import './App.css';

export function App() {
  const [currentScreen, setCurrentScreen] = useState<ScreenId>('dashboard');
  const [activeSession, setActiveSession] = useState<ResearchSession | null>(null);
  const [latestInference, setLatestInference] = useState<InferenceResult | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null);
  const [scannerInfo, setScannerInfo] = useState<ScannerDeviceInfo | null>(null);

  const fetchStatus = async () => {
    try {
      const data = await apiClient.getStatus();
      setSystemStatus(data);
      setScannerInfo(data.scanner);
    } catch (err) {
      console.warn('Backend status check failed:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 8000);
    return () => clearInterval(interval);
  }, []);

  const handleStartNewSession = () => {
    setActiveSession(null);
    setLatestInference(null);
    setCurrentScreen('register');
  };

  const handleSessionCreated = (session: ResearchSession) => {
    setActiveSession(session);
    setLatestInference(null);
    setCurrentScreen('capture');
  };

  const handleSessionUpdated = (session: ResearchSession) => {
    setActiveSession(session);
  };

  const handleInferenceCompleted = (result: InferenceResult) => {
    setLatestInference(result);
    // Also update active session if it matches
    if (activeSession && activeSession.session_id === result.session_id) {
      setActiveSession((prev) =>
        prev
          ? {
              ...prev,
              status: 'COMPLETED',
              predicted_group: result.predicted_group,
              confidence: result.confidence,
              canonical_probabilities: result.probabilities,
              model_name: result.model_name
            }
          : prev
      );
    }
    setCurrentScreen('results');
  };

  const handleSelectSession = (session: ResearchSession) => {
    setActiveSession(session);
  };

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col text-slate-800 font-sans selection:bg-teal-500 selection:text-white">
      {/* 1. Persistent Research Warning Banner */}
      <ResearchDisclaimer />

      {/* 2. Top Navigation Bar */}
      <Navbar
        currentScreen={currentScreen}
        onNavigate={setCurrentScreen}
        scanner={scannerInfo}
        onStartNewSession={handleStartNewSession}
      />

      {/* 3. Main Workspace Area */}
      <main className="flex-1 w-full max-w-[1600px] mx-auto px-6 sm:px-8 lg:px-10 py-8">
        {currentScreen === 'dashboard' && (
          <DashboardView
            onNavigate={setCurrentScreen}
            onSelectSession={handleSelectSession}
            onStartNewSession={handleStartNewSession}
          />
        )}

        {currentScreen === 'register' && (
          <ParticipantRegistrationView
            onSessionCreated={handleSessionCreated}
            onNavigate={setCurrentScreen}
          />
        )}

        {currentScreen === 'scanner' && (
          <ScannerConnectionView
            scanner={scannerInfo}
            onRefreshScanner={fetchStatus}
            onNavigate={setCurrentScreen}
          />
        )}

        {currentScreen === 'capture' && (
          <FingerprintCaptureView
            session={activeSession}
            onSessionUpdated={handleSessionUpdated}
            onNavigate={setCurrentScreen}
          />
        )}

        {currentScreen === 'quality' && (
          <QualityReviewView
            session={activeSession}
            onNavigate={setCurrentScreen}
          />
        )}

        {currentScreen === 'analysis' && (
          <ExperimentalAnalysisView
            session={activeSession}
            onInferenceCompleted={handleInferenceCompleted}
            onNavigate={setCurrentScreen}
          />
        )}

        {currentScreen === 'results' && (
          <ResearchResultsView
            result={latestInference}
            session={activeSession}
            onStartNewSession={handleStartNewSession}
            onNavigate={setCurrentScreen}
          />
        )}

        {currentScreen === 'history' && (
          <SessionHistoryView
            onSelectSession={handleSelectSession}
            onNavigate={setCurrentScreen}
          />
        )}

        {currentScreen === 'system' && (
          <ModelInfoView status={systemStatus} />
        )}
      </main>

      {/* 4. Academic Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12 text-center text-xs text-slate-500 space-y-1">
        <p className="font-semibold text-slate-700">
          HemoPrint AI &mdash; Biometric Blood Group Identification Academic Research System
        </p>
        <p className="text-[11px] text-slate-400">
          Mantra MFS100 Optical Integration | EXP-06C Linear Projection Reference | Department of Computer Science &amp; Engineering
        </p>
        <p className="text-[11px] text-amber-700 font-medium">
          Investigational Machine Learning Prototype &bull; Not for Clinical Transfusion or Diagnostic Decision-Making
        </p>
      </footer>
    </div>
  );
}

export default App;
