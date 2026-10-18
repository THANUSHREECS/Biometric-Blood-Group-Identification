import React, { useState, useEffect } from 'react';
import {
  Search,
  RefreshCw
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { ResearchSession, ScreenId } from '../types';

interface SessionHistoryViewProps {
  onSelectSession: (session: ResearchSession) => void;
  onNavigate: (screen: ScreenId) => void;
}

export const SessionHistoryView: React.FC<SessionHistoryViewProps> = ({
  onSelectSession,
  onNavigate
}) => {
  const [sessions, setSessions] = useState<ResearchSession[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [selectedSessionDetail, setSelectedSessionDetail] = useState<ResearchSession | null>(null);

  const fetchSessions = async () => {
    setIsLoading(true);
    try {
      const data = await apiClient.listSessions(100);
      setSessions(data);
    } catch (err) {
      console.error('Failed to fetch session history:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const handleInspect = async (sessionId: string) => {
    try {
      const detail = await apiClient.getSession(sessionId);
      setSelectedSessionDetail(detail);
    } catch (err) {
      console.error('Failed to load session details:', err);
    }
  };

  const filteredSessions = sessions.filter((s) => {
    const matchesSearch =
      s.participant_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.session_id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'ALL' || s.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Research Session History & Database Audit
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Complete audit trail of participant registrations, Left Thumb impressions, and model predictions.
          </p>
        </div>

        <button
          onClick={fetchSessions}
          disabled={isLoading}
          className="self-start sm:self-auto px-4 py-2.5 text-xs font-semibold text-slate-700 bg-white border border-slate-300 rounded-xl shadow-2xs hover:bg-slate-50 active:bg-slate-100 flex items-center gap-2 transition-colors shrink-0"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-teal-600' : ''}`} />
          <span>Refresh Records</span>
        </button>
      </div>

      {/* Filters & Search Toolbar */}
      <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200 shadow-xs flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="relative w-full sm:max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Participant ID (e.g. P-4A9F)..."
            className="w-full pl-10 pr-4 py-2 text-xs border border-slate-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-teal-500 outline-none shadow-2xs"
          />
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          <span className="text-xs text-slate-500 font-medium whitespace-nowrap">Status Filter:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-full sm:w-44 px-3 py-2 text-xs border border-slate-300 rounded-xl bg-white outline-none focus:ring-2 focus:ring-teal-500 shadow-2xs"
          >
            <option value="ALL">All Statuses</option>
            <option value="COMPLETED">Completed</option>
            <option value="CAPTURING">Capturing</option>
            <option value="CREATED">Created</option>
          </select>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 font-bold border-b border-slate-200 uppercase tracking-wider text-[11px]">
              <tr>
                <th className="px-5 py-3.5">Participant ID</th>
                <th className="px-5 py-3.5">Timestamp</th>
                <th className="px-5 py-3.5">Impressions</th>
                <th className="px-5 py-3.5">Reference Group</th>
                <th className="px-5 py-3.5">Experimental Output</th>
                <th className="px-5 py-3.5">Protocol Status</th>
                <th className="px-5 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {isLoading ? (
                <tr>
                  <td colSpan={7} className="px-5 py-12 text-center text-slate-400">
                    <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-teal-600" />
                    <span>Loading database records from SQLite...</span>
                  </td>
                </tr>
              ) : filteredSessions.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-5 py-12 text-center text-slate-400">
                    No sessions match query "{searchQuery}" or status "{statusFilter}".
                  </td>
                </tr>
              ) : (
                filteredSessions.map((s) => (
                  <tr key={s.session_id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="px-5 py-4 font-mono font-bold text-slate-900 whitespace-nowrap">
                      {s.participant_id}
                    </td>
                    <td className="px-5 py-4 text-slate-500 whitespace-nowrap">
                      {new Date(s.created_at).toLocaleDateString()}{' '}
                      <span className="text-[11px] text-slate-400">
                        {new Date(s.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-slate-700 whitespace-nowrap">
                      <span className="font-semibold">{s.capture_count ?? 0}</span>
                      <span className="text-slate-400 text-[11px]"> / 3 scans</span>
                    </td>
                    <td className="px-5 py-4 text-slate-600 whitespace-nowrap">
                      {s.self_reported_group ? (
                        <div className="flex items-center gap-1.5">
                          <span className="font-bold text-slate-900">{s.self_reported_group}</span>
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 border border-slate-200">
                            {s.verification_status || 'unverified'}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400 italic">None</span>
                      )}
                    </td>
                    <td className="px-5 py-4 font-bold text-slate-900 whitespace-nowrap">
                      {s.predicted_group ? (
                        <div className="flex items-center gap-1.5">
                          <span className="px-2 py-0.5 bg-teal-50 text-teal-900 rounded border border-teal-200 font-mono">
                            {s.predicted_group}
                          </span>
                          <span className="text-[11px] font-normal text-slate-500 font-mono">
                            ({(Number(s.confidence) * 100).toFixed(1)}%)
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400 font-normal italic">Pending</span>
                      )}
                    </td>
                    <td className="px-5 py-4 whitespace-nowrap">
                      <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold ${
                        s.status === 'COMPLETED'
                          ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                          : s.status === 'CAPTURING'
                          ? 'bg-blue-50 text-blue-800 border border-blue-200'
                          : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        {s.status}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-right whitespace-nowrap space-x-2">
                      <button
                        onClick={() => handleInspect(s.session_id)}
                        className="px-2.5 py-1 text-xs font-semibold text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
                      >
                        Inspect
                      </button>
                      <button
                        onClick={() => {
                          onSelectSession(s);
                          if (s.status === 'COMPLETED') onNavigate('results');
                          else if (s.capture_count === 3) onNavigate('analysis');
                          else onNavigate('capture');
                        }}
                        className="px-2.5 py-1 text-xs font-bold text-white bg-teal-700 hover:bg-teal-800 rounded-lg shadow-2xs transition-colors"
                      >
                        Load &rarr;
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail Inspection Modal */}
      {selectedSessionDetail && (
        <div className="fixed inset-0 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-3xl w-full p-6 sm:p-7 space-y-5 shadow-2xl border border-slate-200 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Session Inspection: {selectedSessionDetail.participant_id}
                </h3>
                <p className="text-xs text-slate-400 font-mono mt-0.5">
                  ID: {selectedSessionDetail.session_id}
                </p>
              </div>
              <button
                onClick={() => setSelectedSessionDetail(null)}
                className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-800 flex items-center justify-center text-lg font-bold transition-colors"
              >
                &times;
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-slate-400 font-medium block">Timestamp:</span>
                <span className="font-bold text-slate-800">
                  {new Date(selectedSessionDetail.created_at).toLocaleString()}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-slate-400 font-medium block">Consent Verified:</span>
                <span className="font-bold text-emerald-700">YES (Mandatory Protocol)</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-slate-400 font-medium block">Reference Blood Group:</span>
                <span className="font-bold text-slate-800">
                  {selectedSessionDetail.self_reported_group || 'None provided'}
                </span>
              </div>
            </div>

            {/* Captures in this session */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Left Thumb Impressions ({selectedSessionDetail.captures?.length || 0} / 3)
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {selectedSessionDetail.captures?.map((cap) => {
                  const capImg = apiClient.getImageUrl(cap.image_filename || cap.image_url);
                  return (
                    <div key={cap.capture_id} className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-center space-y-2">
                      <span className="text-xs font-bold text-slate-700 block">
                        Impression #{cap.impression_number}
                      </span>
                      <div className="h-40 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center p-2">
                        <img
                          src={capImg}
                          alt={`Imp ${cap.impression_number}`}
                          className="max-h-full w-auto object-contain"
                          onError={(e) => {
                            (e.currentTarget as HTMLImageElement).style.display = 'none';
                          }}
                        />
                      </div>
                      <div className="text-[11px] text-slate-600 font-mono">
                        Q: {cap.quality_score}/100 | NFIQ: {cap.nfiq_score}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="pt-4 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-end gap-2.5">
              <button
                onClick={() => setSelectedSessionDetail(null)}
                className="w-full sm:w-auto px-4 py-2.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-xl transition-colors"
              >
                Close Inspection
              </button>
              <button
                onClick={() => {
                  onSelectSession(selectedSessionDetail);
                  setSelectedSessionDetail(null);
                  if (selectedSessionDetail.status === 'COMPLETED') onNavigate('results');
                  else if ((selectedSessionDetail.captures?.length || 0) === 3) onNavigate('analysis');
                  else onNavigate('capture');
                }}
                className="w-full sm:w-auto px-5 py-2.5 text-xs font-bold text-white bg-teal-700 hover:bg-teal-800 rounded-xl shadow-xs transition-colors"
              >
                Load Session in Workspace
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

