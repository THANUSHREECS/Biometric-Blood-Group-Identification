import React, { useEffect, useState } from 'react';
import {
  Users,
  Camera,
  Layers,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  Clock,
  Radio,
  FileText
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { SystemStatusResponse, ResearchSession, ScreenId } from '../types';

interface DashboardViewProps {
  onNavigate: (screen: ScreenId) => void;
  onSelectSession: (session: ResearchSession) => void;
  onStartNewSession: () => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onNavigate,
  onSelectSession,
  onStartNewSession
}) => {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [recentSessions, setRecentSessions] = useState<ResearchSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDashboardData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [sysStatus, sessions] = await Promise.all([
        apiClient.getStatus(),
        apiClient.listSessions(8)
      ]);
      setStatus(sysStatus);
      setRecentSessions(sessions);
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard metrics from backend.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const stats = status?.research_stats;
  const scanner = status?.scanner;
  const model = status?.model;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Top Banner / Heading */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Research Operations Dashboard
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Empirical investigation of 500 DPI optical fingerprint friction-ridge patterns and ABO blood groups.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadDashboardData}
            disabled={isLoading}
            className="px-3.5 py-2.5 text-xs font-semibold text-slate-700 bg-white border border-slate-300 rounded-xl shadow-2xs hover:bg-slate-50 active:bg-slate-100 flex items-center gap-1.5 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-teal-600' : ''}`} />
            <span>Refresh Data</span>
          </button>
          <button
            onClick={onStartNewSession}
            className="px-4 py-2.5 text-xs font-bold text-white bg-teal-700 hover:bg-teal-800 rounded-xl shadow-xs flex items-center gap-2 transition-colors"
          >
            <span>Start New Session</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl text-rose-800 text-xs flex items-center justify-between shadow-2xs">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={loadDashboardData}
            className="text-xs font-bold underline hover:text-rose-950 cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Real Statistics Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Stored Participants */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex flex-col justify-between min-h-[140px] hover:border-slate-300 transition-colors">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Registered Participants
            </span>
            <div className="p-2 bg-teal-50 rounded-xl text-teal-700">
              <Users className="w-4.5 h-4.5" />
            </div>
          </div>
          <div className="mt-2">
            <div className="text-3xl font-extrabold text-slate-900 font-mono">
              {isLoading ? '...' : stats?.unique_participants ?? 0}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Unique participants in SQLite database
            </p>
          </div>
        </div>

        {/* Metric 2: Valid Fingerprint Impressions */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex flex-col justify-between min-h-[140px] hover:border-slate-300 transition-colors">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Accepted Impressions
            </span>
            <div className="p-2 bg-blue-50 rounded-xl text-blue-700">
              <Camera className="w-4.5 h-4.5" />
            </div>
          </div>
          <div className="mt-2">
            <div className="text-3xl font-extrabold text-slate-900 font-mono">
              {isLoading ? '...' : stats?.valid_captures ?? 0}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Valid Left Thumb impressions stored
            </p>
          </div>
        </div>

        {/* Metric 3: Active Model */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex flex-col justify-between min-h-[140px] hover:border-slate-300 transition-colors">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Active Research Model
            </span>
            <div className="p-2 bg-indigo-50 rounded-xl text-indigo-700">
              <Layers className="w-4.5 h-4.5" />
            </div>
          </div>
          <div className="mt-2">
            <div className="text-base font-bold text-slate-900 truncate" title={model?.name}>
              {isLoading ? '...' : model?.tag || 'EXP-06C Linear'}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              5-Fold Feature Concatenation (3072-d)
            </p>
          </div>
        </div>

        {/* Metric 4: Scanner Hardware Status */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex flex-col justify-between min-h-[140px] hover:border-slate-300 transition-colors">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Scanner Hardware
            </span>
            <div className={`p-2 rounded-xl ${
              scanner?.mock_mode
                ? 'bg-amber-50 text-amber-700'
                : scanner?.is_connected
                ? 'bg-emerald-50 text-emerald-700'
                : 'bg-rose-50 text-rose-700'
            }`}>
              <Radio className="w-4.5 h-4.5" />
            </div>
          </div>
          <div className="mt-2">
            <div className="text-base font-bold text-slate-900">
              {isLoading ? '...' : scanner?.mock_mode ? 'DEMO MODE' : scanner?.is_connected ? 'CONNECTED' : 'OFFLINE'}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Mantra MFS100 (500 DPI Optical)
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid: Protocol Action Cards & Recent Sessions Table */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols): Recent Research Sessions */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col">
          <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-slate-400" />
              <h2 className="text-sm font-bold text-slate-900">
                Recent Research Sessions
              </h2>
            </div>
            <button
              onClick={() => onNavigate('history')}
              className="text-xs font-semibold text-teal-700 hover:text-teal-900 flex items-center gap-1 transition-colors"
            >
              <span>View All History</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="overflow-x-auto flex-1">
            {isLoading ? (
              <div className="p-12 text-center text-xs text-slate-400">
                <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-teal-600" />
                <span>Loading session history...</span>
              </div>
            ) : recentSessions.length === 0 ? (
              <div className="p-12 text-center">
                <FileText className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                <p className="text-sm font-semibold text-slate-700">No research sessions recorded yet.</p>
                <p className="text-xs text-slate-400 mt-1">Click "Start New Session" above to register a participant.</p>
              </div>
            ) : (
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-500 font-bold border-b border-slate-200 uppercase tracking-wider text-[10px]">
                  <tr>
                    <th className="px-5 py-3.5">Participant ID</th>
                    <th className="px-5 py-3.5">Created</th>
                    <th className="px-5 py-3.5">Scans</th>
                    <th className="px-5 py-3.5">Status</th>
                    <th className="px-5 py-3.5">Experimental Output</th>
                    <th className="px-5 py-3.5 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {recentSessions.map((s) => (
                    <tr key={s.session_id} className="hover:bg-slate-50/80 transition-colors">
                      <td className="px-5 py-3.5 font-mono font-bold text-slate-900 whitespace-nowrap">
                        {s.participant_id}
                      </td>
                      <td className="px-5 py-3.5 text-slate-500 whitespace-nowrap">
                        {new Date(s.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                      <td className="px-5 py-3.5 text-slate-700 whitespace-nowrap">
                        <span className="font-semibold">{s.capture_count ?? 0}</span>
                        <span className="text-slate-400 text-[11px]"> / 3</span>
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                          s.status === 'COMPLETED'
                            ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                            : s.status === 'CAPTURING'
                            ? 'bg-blue-50 text-blue-800 border border-blue-200'
                            : 'bg-slate-100 text-slate-700 border border-slate-200'
                        }`}>
                          {s.status}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        {s.predicted_group ? (
                          <div className="flex items-center gap-1.5 font-mono">
                            <span className="font-bold text-slate-900 px-1.5 py-0.5 bg-slate-100 rounded border border-slate-200">
                              {s.predicted_group}
                            </span>
                            <span className="text-[11px] text-slate-400">
                              ({(Number(s.confidence) * 100).toFixed(0)}%)
                            </span>
                          </div>
                        ) : (
                          <span className="text-slate-400 italic">Pending</span>
                        )}
                      </td>
                      <td className="px-5 py-3.5 text-right whitespace-nowrap">
                        <button
                          onClick={() => {
                            onSelectSession(s);
                            if (s.status === 'COMPLETED') {
                              onNavigate('results');
                            } else if (s.capture_count === 3) {
                              onNavigate('analysis');
                            } else {
                              onNavigate('capture');
                            }
                          }}
                          className="px-2.5 py-1 text-xs font-bold text-teal-700 hover:text-teal-900 hover:bg-teal-50 rounded-lg transition-colors"
                        >
                          Open &rarr;
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Right Column: Research Protocol Guidelines Card */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
            <ShieldCheck className="w-5 h-5 text-teal-700" />
            <h2 className="text-sm font-bold text-slate-900">
              Academic Protocol Guidelines
            </h2>
          </div>

          <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
              <span className="font-bold text-slate-900 block mb-0.5">1. Left Thumb Standard:</span>
              Always capture exactly three impressions of the participant's Left Thumb on the Mantra MFS100 platen.
            </div>

            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
              <span className="font-bold text-slate-900 block mb-0.5">2. Lift & Reposition:</span>
              Instruct the participant to completely lift and reposition their finger between scans to prevent identical surface contact.
            </div>

            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
              <span className="font-bold text-slate-900 block mb-0.5">3. Multi-Scan Feature Fusion:</span>
              Inference requires all 3 distinct impressions concatenated into a 3072-dimensional embedding before linear projection.
            </div>

            <div className="p-3 bg-amber-50 rounded-xl border border-amber-200 text-amber-950">
              <span className="font-bold block mb-0.5">4. No Clinical Transfusion Claims:</span>
              All outputs are research statistics. Never present model predictions as diagnostic blood types.
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={() => onNavigate('system')}
              className="w-full py-2.5 text-xs font-bold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-xl transition-colors text-center"
            >
              View Model Architecture & Epistemic Audit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

