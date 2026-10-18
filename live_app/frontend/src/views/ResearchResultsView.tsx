import React from 'react';
import {
  AlertTriangle,
  ShieldAlert,
  UserPlus,
  Printer
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { InferenceResult, ResearchSession, ScreenId, BloodGroup } from '../types';

interface ResearchResultsViewProps {
  result: InferenceResult | null;
  session: ResearchSession | null;
  onStartNewSession: () => void;
  onNavigate: (screen: ScreenId) => void;
}

export const ResearchResultsView: React.FC<ResearchResultsViewProps> = ({
  result,
  session,
  onStartNewSession,
  onNavigate
}) => {
  // If result is null, attempt to display session prediction if available
  const displayResult = result || (session && session.predicted_group ? {
    success: true,
    session_id: session.session_id,
    participant_id: session.participant_id,
    created_at: session.created_at,
    self_reported_group: session.self_reported_group,
    label_source: session.label_source,
    verification_status: session.verification_status,
    predicted_group: session.predicted_group,
    confidence: Number(session.confidence || 0),
    probabilities: typeof session.canonical_probabilities === 'string'
      ? JSON.parse(session.canonical_probabilities)
      : (session.canonical_probabilities || { 'A+': 0.25, 'AB+': 0.25, 'B+': 0.25, 'O+': 0.25 }),
    classes: ['A+', 'AB+', 'B+', 'O+'] as BloodGroup[],
    model_name: session.model_name || 'EXP-06C Config A (Linear Reference)',
    feature_dims: '3072-d Concatenation',
    status: 'COMPLETED',
    message: 'Inference loaded from database history.',
    is_low_confidence: Number(session.confidence || 0) < 0.70,
    captures: (session.captures || []).map((c) => ({
      impression_number: c.impression_number,
      image_url: apiClient.getImageUrl(c.image_filename),
      quality_score: c.quality_score,
      nfiq_score: c.nfiq_score
    })),
    research_disclaimer: 'RESEARCH PROTOTYPE — NOT FOR MEDICAL USE'
  } as InferenceResult : null);

  if (!displayResult) {
    return (
      <div className="max-w-xl mx-auto p-8 bg-white rounded-xl border border-slate-200 text-center space-y-4">
        <AlertTriangle className="w-10 h-10 text-amber-500 mx-auto" />
        <h2 className="text-lg font-bold text-slate-900">No Experimental Results Available</h2>
        <p className="text-xs text-slate-500">
          Run 3-scan feature fusion inference on an active session to generate experimental research results.
        </p>
        <button
          onClick={() => onNavigate('dashboard')}
          className="px-4 py-2 text-xs font-semibold text-white bg-teal-600 rounded-lg hover:bg-teal-700"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  const {
    participant_id,
    created_at,
    predicted_group,
    confidence,
    probabilities,
    model_name,
    is_low_confidence,
    advisory,
    captures,
    self_reported_group,
    label_source,
    verification_status
  } = displayResult;

  const CANONICAL_CLASSES: BloodGroup[] = ['A+', 'AB+', 'B+', 'O+'];

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6 animate-fade-in">
      {/* Mandatory Prominent Research Warning Box (Exact phrasing required) */}
      <div className="p-5 bg-amber-50 border-2 border-amber-300 rounded-2xl shadow-xs text-amber-950 space-y-2">
        <div className="flex items-center gap-2.5 text-amber-900">
          <ShieldAlert className="w-6 h-6 text-amber-700 shrink-0" />
          <h2 className="text-sm font-extrabold uppercase tracking-wider">
            RESEARCH PROTOTYPE — NOT FOR MEDICAL USE
          </h2>
        </div>
        <p className="text-xs text-amber-900 leading-relaxed">
          This experimental system has not demonstrated reliable, generalizable blood-group classification performance. 
          Its output must not be used for medical diagnosis, emergency treatment, or transfusion decisions. 
          Blood group must be established through validated clinical laboratory testing.
        </p>
      </div>

      {/* Primary Results Card */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-2xs overflow-hidden">
        <div className="px-6 py-4 bg-slate-900 text-white flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-teal-400">
                Participant Research Session
              </span>
              <span className="px-2 py-0.5 text-xs font-mono font-bold bg-slate-800 text-white rounded border border-slate-700">
                {participant_id}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Evaluated: {new Date(created_at).toLocaleString()}
            </p>
          </div>

          <div className="text-xs text-slate-400">
            Model: <span className="font-semibold text-slate-200">{model_name}</span>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {/* Predicted Class Display (No deceptive medical verification badge) */}
          <div className="p-6 bg-slate-50 rounded-xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-6">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block mb-1">
                Experimental Model Output (Non-Clinical)
              </span>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-extrabold text-slate-900 font-mono tracking-tight">
                  {predicted_group}
                </span>
                <span className="text-xs font-semibold text-slate-500">
                  Model Output Score: {(confidence * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1.5 italic">
                Notice: Softmax score represents relative model activation, not calibrated clinical probability.
              </p>
            </div>

            {self_reported_group && (
              <div className="p-3 bg-white rounded-lg border border-slate-200 text-xs space-y-1">
                <span className="text-slate-400 font-medium block">Reference Blood Group:</span>
                <div className="font-bold text-slate-800 flex items-center gap-1.5">
                  <span>{self_reported_group}</span>
                  <span className={`px-1.5 py-0.2 rounded text-[10px] ${
                    verification_status === 'verified' ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-600'
                  }`}>
                    {verification_status === 'verified' ? 'Lab Verified' : 'Self-Reported (Unverified)'}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400">
                  Source: {label_source || 'self-reported'}
                </div>
              </div>
            )}
          </div>

          {advisory && (
            <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-900 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>
                {advisory.message} {is_low_confidence ? '(Low confidence margin)' : ''}
              </span>
            </div>
          )}

          {/* Canonical 4-Class Probabilities Bar Chart */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Four-Class Softmax Output Distribution [Canonical Order: A+, AB+, B+, O+]
            </h3>

            <div className="space-y-2.5">
              {CANONICAL_CLASSES.map((bg) => {
                const score = probabilities ? probabilities[bg] ?? 0 : 0;
                const pct = (score * 100).toFixed(1);
                const isWinner = bg === predicted_group;

                return (
                  <div key={bg} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className={`font-bold ${isWinner ? 'text-teal-900' : 'text-slate-700'}`}>
                        {bg} {isWinner && <span className="text-[10px] font-normal text-teal-700 ml-1">(Predicted Class)</span>}
                      </span>
                      <span className="font-mono font-semibold text-slate-600">
                        {pct}% ({score.toFixed(4)})
                      </span>
                    </div>

                    <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isWinner ? 'bg-teal-600' : 'bg-slate-400'
                        }`}
                        style={{ width: `${Math.max(2, score * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 3 Left Thumb Impressions Evaluated */}
          <div className="space-y-3 pt-4 border-t border-slate-100">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Evaluated Input Scans (3 Left Thumb Impressions)
            </h3>

            <div className="grid grid-cols-3 gap-3">
              {captures.map((cap) => (
                <div key={cap.impression_number} className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-center space-y-1.5">
                  <div className="text-[11px] font-bold text-slate-600 uppercase">
                    Impression {cap.impression_number}
                  </div>
                  <div className="h-32 bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center p-1.5 border border-slate-200">
                    <img
                      src={apiClient.getImageUrl(cap.image_filename || cap.image_url)}
                      alt={`Impression ${cap.impression_number}`}
                      className="h-full max-h-28 object-contain rounded"
                      loading="lazy"
                    />
                  </div>
                  <div className="text-[10px] text-slate-500">
                    Quality: {cap.quality_score}/100 | NFIQ: {cap.nfiq_score}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Card Footer Actions */}
        <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="text-xs text-slate-500">
            Computation completed successfully. Results archived to database.
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => window.print()}
              className="px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 flex items-center gap-1.5 shadow-2xs"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print Audit Record</span>
            </button>

            <button
              onClick={onStartNewSession}
              className="px-4 py-2 text-xs font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-lg shadow-sm flex items-center gap-1.5"
            >
              <UserPlus className="w-3.5 h-3.5" />
              <span>Start Next Session</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
