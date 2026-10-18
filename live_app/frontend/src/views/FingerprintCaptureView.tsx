import React, { useState } from 'react';
import {
  Camera,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Hand,
  AlertCircle
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { ResearchSession, ResearchCapture, ScreenId } from '../types';

interface FingerprintCaptureViewProps {
  session: ResearchSession | null;
  onSessionUpdated: (session: ResearchSession) => void;
  onNavigate: (screen: ScreenId) => void;
}

export const FingerprintCaptureView: React.FC<FingerprintCaptureViewProps> = ({
  session,
  onSessionUpdated,
  onNavigate
}) => {
  const [activeStep, setActiveStep] = useState<number>(1);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [captures, setCaptures] = useState<{ [key: number]: ResearchCapture }>({});

  // Sync existing session captures if any
  React.useEffect(() => {
    if (session?.captures) {
      const map: { [key: number]: ResearchCapture } = {};
      session.captures.forEach((c) => {
        map[c.impression_number] = c;
      });
      setCaptures(map);
      // Auto set step to next uncaptured impression
      if (!map[1]) setActiveStep(1);
      else if (!map[2]) setActiveStep(2);
      else if (!map[3]) setActiveStep(3);
      else setActiveStep(3);
    }
  }, [session]);

  if (!session) {
    return (
      <div className="max-w-xl mx-auto p-8 bg-white rounded-xl border border-slate-200 text-center space-y-4">
        <AlertTriangle className="w-10 h-10 text-amber-500 mx-auto" />
        <h2 className="text-lg font-bold text-slate-900">No Active Research Session</h2>
        <p className="text-xs text-slate-500">
          Please register a participant or select an existing session before capturing fingerprints.
        </p>
        <button
          onClick={() => onNavigate('register')}
          className="px-4 py-2 text-xs font-semibold text-white bg-teal-600 rounded-lg hover:bg-teal-700"
        >
          Register Participant
        </button>
      </div>
    );
  }

  const handleCapture = async (impressionNumber: number) => {
    setIsCapturing(true);
    setError(null);
    try {
      const res = await apiClient.captureImpression(session.session_id, impressionNumber);
      if (!res.success) {
        throw new Error(res.rejection_reason || 'Capture failed');
      }

      // Reload updated session from backend
      const updatedSess = await apiClient.getSession(session.session_id);
      onSessionUpdated(updatedSess);

      if (!res.is_acceptable) {
        setError(`Impression ${impressionNumber} rejected: ${res.rejection_reason}. Please ask participant to clean finger, apply even pressure, and retry.`);
      } else {
        if (impressionNumber < 3) {
          setActiveStep(impressionNumber + 1);
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Capture failed on MFS100 scanner.');
    } finally {
      setIsCapturing(false);
    }
  };

  const c1 = captures[1];
  const c2 = captures[2];
  const c3 = captures[3];

  const allThreeAccepted =
    c1?.is_acceptable && c2?.is_acceptable && c3?.is_acceptable;

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6 animate-fade-in">
      {/* Header & Protocol Information */}
      <div className="pb-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Left Thumb Capture Protocol
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-bold bg-teal-100 text-teal-800 rounded-md">
              Participant: {session.participant_id}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            MFS100 500 DPI Acquisition: 3 distinct Left Thumb impressions with mandatory lift-and-reposition between scans.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('quality')}
            className="px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-300 rounded-lg shadow-2xs hover:bg-slate-50 transition-colors"
          >
            Review Quality Audit
          </button>
          {allThreeAccepted && (
            <button
              onClick={() => onNavigate('analysis')}
              className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white font-bold text-xs rounded-lg shadow-xs flex items-center gap-1.5 transition-colors"
            >
              <span>Proceed to Analysis</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-xs flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-bold block">Capture Feedback</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Protocol Reposition Guidance Banner */}
      <div className="p-4 bg-blue-50 border border-blue-200 rounded-xl flex items-center justify-between text-xs text-blue-900">
        <div className="flex items-center gap-3">
          <Hand className="w-5 h-5 text-blue-600 shrink-0" />
          <div>
            <span className="font-bold mr-1">Mandatory Physical Protocol:</span>
            Participant places Left Thumb on optical platen. Between captures, thumb must be <strong>completely lifted and repositioned</strong> to ensure distinct physical contact.
          </div>
        </div>
        <span className="hidden md:inline-block px-2.5 py-1 bg-white font-bold rounded text-blue-800 border border-blue-200 shadow-2xs text-[11px]">
          LEFT THUMB ONLY
        </span>
      </div>

      {/* Three Aligned, Equally Sized Capture Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[1, 2, 3].map((num) => {
          const cap = captures[num];
          const isDone = cap && cap.is_acceptable;
          const isRejected = cap && !cap.is_acceptable;
          const isCurrentActive = activeStep === num;

          return (
            <div
              key={num}
              className={`bg-white rounded-2xl border-2 flex flex-col justify-between shadow-2xs overflow-hidden transition-all ${
                isDone
                  ? 'border-emerald-300 ring-1 ring-emerald-200'
                  : isRejected
                  ? 'border-rose-300 ring-1 ring-rose-200'
                  : isCurrentActive
                  ? 'border-teal-500 ring-2 ring-teal-500/20'
                  : 'border-slate-200'
              }`}
            >
              {/* Card Header */}
              <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-extrabold text-slate-700 uppercase tracking-wider block">
                    Impression {num} of 3
                  </span>
                  <span className="text-[10px] text-slate-400 font-medium">
                    {num === 1 ? 'Initial Placement' : num === 2 ? 'After 1st Lift' : 'After 2nd Lift'}
                  </span>
                </div>

                <div>
                  {isDone ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                      <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                      <span>Accepted</span>
                    </span>
                  ) : isRejected ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 text-rose-800 border border-rose-200">
                      <AlertTriangle className="w-3 h-3 text-rose-600" />
                      <span>Rejected</span>
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-200 text-slate-600">
                      Pending
                    </span>
                  )}
                </div>
              </div>

              {/* Card Body: Thumbnail Display */}
              <div className="p-5 flex-1 flex flex-col items-center justify-center space-y-4">
                <div className="w-full h-52 bg-slate-100 rounded-xl border border-slate-200 flex items-center justify-center overflow-hidden relative p-2">
                  {cap ? (
                    <img
                      src={apiClient.getImageUrl(cap.image_filename || cap.image_url)}
                      alt={`Left Thumb Impression ${num}`}
                      className="h-full max-h-48 object-contain rounded"
                      loading="lazy"
                    />
                  ) : (
                    <div className="text-center p-4 text-slate-400 space-y-2">
                      <Camera className="w-8 h-8 mx-auto text-slate-300" />
                      <p className="text-xs">No scan recorded yet</p>
                    </div>
                  )}
                </div>

                {/* Metrics Breakdown */}
                {cap ? (
                  <div className="w-full bg-slate-50 p-3 rounded-xl border border-slate-200 space-y-1.5 text-xs">
                    <div className="flex justify-between text-slate-500">
                      <span>Dimensions:</span>
                      <span className="font-mono font-semibold text-slate-700">
                        {cap.width} &times; {cap.height} px
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-500">
                      <span>MFS100 Quality:</span>
                      <span className={`font-bold ${cap.quality_score >= 60 ? 'text-emerald-700' : 'text-amber-700'}`}>
                        {cap.quality_score} / 100
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-500">
                      <span>NIST NFIQ:</span>
                      <span className="font-bold text-slate-800">
                        Level {cap.nfiq_score} {cap.nfiq_score <= 2 ? '(Excellent)' : '(Moderate)'}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="w-full bg-slate-50/50 p-3 rounded-xl border border-dashed border-slate-200 text-center text-xs text-slate-400">
                    Awaiting scan from MFS100 optical platen
                  </div>
                )}
              </div>

              {/* Card Action Button */}
              <div className="p-4 bg-slate-50 border-t border-slate-200">
                <button
                  onClick={() => handleCapture(num)}
                  disabled={isCapturing}
                  className={`w-full py-2.5 rounded-xl font-bold text-xs flex items-center justify-center gap-2 transition-all ${
                    isCapturing && activeStep === num
                      ? 'bg-slate-400 text-white cursor-not-allowed'
                      : isDone
                      ? 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-100 hover:border-slate-400 cursor-pointer shadow-2xs'
                      : 'bg-teal-600 hover:bg-teal-700 text-white shadow-sm cursor-pointer'
                  }`}
                >
                  <Camera className={`w-3.5 h-3.5 ${isCapturing && activeStep === num ? 'animate-pulse' : ''}`} />
                  <span>
                    {isCapturing && activeStep === num
                      ? 'Scanning MFS100...'
                      : isDone
                      ? `Recapture Impression ${num}`
                      : `Capture Impression ${num}`}
                  </span>
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Completion Notification */}
      {allThreeAccepted && (
        <div className="p-5 bg-emerald-50 border-2 border-emerald-300 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-4 shadow-xs">
          <div className="flex items-center gap-3 text-emerald-900">
            <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0" />
            <div>
              <h3 className="text-sm font-bold">Three Valid Left Thumb Impressions Verified</h3>
              <p className="text-xs text-emerald-800 mt-0.5">
                The session satisfies the 3-scan research requirement. You can now execute feature fusion or review quality signals.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 w-full sm:w-auto">
            <button
              onClick={() => onNavigate('quality')}
              className="w-full sm:w-auto px-4 py-2.5 bg-white border border-emerald-300 text-emerald-800 text-xs font-semibold rounded-xl hover:bg-emerald-100/50 shadow-2xs"
            >
              Inspect Quality
            </button>
            <button
              onClick={() => onNavigate('analysis')}
              className="w-full sm:w-auto px-5 py-2.5 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold rounded-xl shadow-xs flex items-center justify-center gap-2"
            >
              <span>Run Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
