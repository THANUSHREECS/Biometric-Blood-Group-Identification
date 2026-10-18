import React, { useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  Activity
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { ResearchSession, ScreenId } from '../types';

interface QualityReviewViewProps {
  session: ResearchSession | null;
  onNavigate: (screen: ScreenId) => void;
}

export const QualityReviewView: React.FC<QualityReviewViewProps> = ({
  session,
  onNavigate
}) => {
  const [showNotes, setShowNotes] = useState<boolean>(false);

  if (!session || !session.captures || session.captures.length === 0) {
    return (
      <div className="max-w-xl mx-auto p-8 bg-white rounded-2xl border border-slate-200 text-center space-y-4 shadow-xs">
        <div className="w-12 h-12 rounded-full bg-amber-50 text-amber-600 flex items-center justify-center mx-auto">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-bold text-slate-900">No Captures in Active Session</h2>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          Complete Left Thumb captures under the 3-impression protocol before reviewing technical signal quality metrics.
        </p>
        <button
          onClick={() => onNavigate('capture')}
          className="px-5 py-2.5 text-xs font-bold text-white bg-teal-700 rounded-xl hover:bg-teal-800 shadow-2xs"
        >
          Go to Capture Screen
        </button>
      </div>
    );
  }

  const captures = session.captures;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-3">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Biometric Signal Quality Review
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-mono font-bold bg-teal-50 text-teal-800 rounded-md border border-teal-200">
              {session.participant_id}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Deterministic signal audit: Gradient Structure Tensor coherence, local contrast, and platen coverage (EXP-10 & MFS100 SDK).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('capture')}
            className="px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-300 rounded-xl hover:bg-slate-50 shadow-2xs"
          >
            Back to Capture
          </button>
          <button
            onClick={() => onNavigate('analysis')}
            className="px-4 py-2 text-xs font-bold text-white bg-teal-700 hover:bg-teal-800 rounded-xl shadow-xs flex items-center gap-1.5"
          >
            <span>Proceed to Analysis</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Impression Quality Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {captures.map((cap) => {
          const isAcc = cap.is_acceptable;
          const imgUrl = apiClient.getImageUrl(cap.image_filename || cap.image_url);

          return (
            <div
              key={cap.capture_id}
              className={`bg-white rounded-2xl border transition-all flex flex-col justify-between overflow-hidden shadow-xs ${
                isAcc ? 'border-slate-200' : 'border-rose-300 ring-2 ring-rose-500/10'
              }`}
            >
              <div>
                {/* Header */}
                <div className="px-5 py-3.5 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-slate-200 text-slate-700 text-xs font-bold flex items-center justify-center">
                      {cap.impression_number}
                    </span>
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-800">
                      Impression #{cap.impression_number}
                    </span>
                  </div>
                  <span
                    className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                      isAcc
                        ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                        : 'bg-rose-100 text-rose-800 border border-rose-300'
                    }`}
                  >
                    {isAcc ? (
                      <>
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                        <span>ACCEPTED</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="w-3 h-3 text-rose-600" />
                        <span>REJECTED</span>
                      </>
                    )}
                  </span>
                </div>

                {/* Fingerprint Image Preview */}
                <div className="p-4 bg-slate-950 flex items-center justify-center h-60 border-b border-slate-100 relative group">
                  <img
                    src={imgUrl}
                    alt={`Impression ${cap.impression_number}`}
                    className="max-h-52 w-auto object-contain rounded filter drop-shadow-md transition-transform duration-200 group-hover:scale-105"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).style.display = 'none';
                    }}
                  />
                  <div className="absolute bottom-2 right-2 px-2 py-0.5 bg-black/60 backdrop-blur-xs rounded text-[10px] text-slate-300 font-mono">
                    {cap.width || 316} × {cap.height || 354} px
                  </div>
                </div>

                {/* Quality Metrics */}
                <div className="p-5 space-y-4">
                  {/* Metric Group 1: Hardware Metrics */}
                  <div className="space-y-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                      Hardware Sensor Metrics
                    </span>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-200">
                        <span className="text-slate-500 text-[11px] block">MFS100 Quality</span>
                        <div className="flex items-baseline gap-1 mt-0.5">
                          <span className={`text-base font-extrabold font-mono ${
                            (cap.quality_score || 0) >= 60 ? 'text-emerald-700' : 'text-amber-600'
                          }`}>
                            {cap.quality_score}
                          </span>
                          <span className="text-slate-400 text-[11px]">/ 100</span>
                        </div>
                      </div>

                      <div className="p-2.5 bg-slate-50 rounded-xl border border-slate-200">
                        <span className="text-slate-500 text-[11px] block">NIST NFIQ Level</span>
                        <div className="flex items-baseline gap-1 mt-0.5">
                          <span className={`text-base font-extrabold font-mono ${
                            (cap.nfiq_score || 5) <= 2 ? 'text-emerald-700' : 'text-amber-600'
                          }`}>
                            {cap.nfiq_score}
                          </span>
                          <span className="text-slate-400 text-[11px]">(1 best)</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Metric Group 2: Ridge & Platen Analysis */}
                  <div className="space-y-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                      Ridge Structure & Platen Analysis
                    </span>
                    <div className="space-y-1.5 text-xs">
                      {cap.fg_coverage !== undefined && (
                        <div className="flex items-center justify-between p-2 bg-slate-50/70 rounded-lg">
                          <span className="text-slate-600">Platen Area Coverage:</span>
                          <span className="font-bold font-mono text-slate-800">
                            {(cap.fg_coverage * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}

                      {cap.ridge_coherence !== undefined && (
                        <div className="flex items-center justify-between p-2 bg-slate-50/70 rounded-lg">
                          <span className="text-slate-600">Ridge Flow Coherence:</span>
                          <span className="font-bold font-mono text-slate-800">
                            {cap.ridge_coherence.toFixed(3)}
                          </span>
                        </div>
                      )}

                      {cap.unusable_proportion !== undefined && (
                        <div className="flex items-center justify-between p-2 bg-slate-50/70 rounded-lg">
                          <span className="text-slate-600">Unusable Platen Ratio:</span>
                          <span className="font-bold font-mono text-slate-800">
                            {(cap.unusable_proportion * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {cap.rejection_reason && (
                    <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-rose-900 text-xs">
                      <span className="font-bold block mb-0.5">Rejection Reason:</span>
                      <span>{cap.rejection_reason}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Bottom Card Meta */}
              <div className="px-5 py-3 bg-slate-50 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
                <span>Native 500 DPI Optical</span>
                <span className="font-mono text-[10px] text-slate-400">
                  {cap.image_filename ? cap.image_filename.split('/').pop() : ''}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Expandable Scientific Research Notes */}
      <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs">
        <button
          type="button"
          onClick={() => setShowNotes(!showNotes)}
          className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-50 transition-colors"
        >
          <div className="flex items-center gap-2.5">
            <Activity className="w-4 h-4 text-teal-700" />
            <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Scientific Explanation & Confounder Limitations (EXP-10 Evidence)
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <span>{showNotes ? 'Collapse Notes' : 'Expand Notes'}</span>
            {showNotes ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        </button>

        {showNotes && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-100 text-xs text-slate-600 space-y-3 leading-relaxed">
            <p>
              <strong>Image Quality vs. Blood-Group Predictability:</strong> High optical sensor clarity, low NFIQ score (Level 1), and high ridge flow coherence verify that an impression was cleanly captured on the Mantra MFS100 platen. However, extensive experimental audits (EXP-01 through EXP-10) demonstrated that <em>clean optical image quality does NOT equate to biometric blood-group predictability</em>.
            </p>
            <p>
              In EXP-10, systematic consistency analysis across 3 impressions per participant revealed high intra-participant uniformity (mean ridge orientation cosine similarity = 0.9355; 95% of participants exhibited quality ratio &lt; 1.5). Weighting scans by technical quality scores degraded downstream classifier performance compared to standard arithmetic feature fusion because quality weighting removes the variance-reducing benefit of unweighted multi-impression aggregation.
            </p>
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-slate-700 font-mono text-[11px]">
              Key Finding: Ridge clarity is necessary for reproducible biometric extraction, but insufficient to establish non-chance association with ABO/Rh serology.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

