import React, { useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  RefreshCw
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { BloodGroup, ResearchSession, ScreenId } from '../types';

interface ParticipantRegistrationViewProps {
  onSessionCreated: (session: ResearchSession) => void;
  onNavigate: (screen: ScreenId) => void;
}

export const ParticipantRegistrationView: React.FC<ParticipantRegistrationViewProps> = ({
  onSessionCreated,
  onNavigate
}) => {
  const [participantId, setParticipantId] = useState<string>('');
  const [consentGiven, setConsentGiven] = useState<boolean>(false);
  const [selfReportedGroup, setSelfReportedGroup] = useState<BloodGroup | ''>('');
  const [labelSource, setLabelSource] = useState<'self_reported' | 'laboratory_verified'>('self_reported');
  const [notes, setNotes] = useState<string>('');

  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const generatePseudonym = () => {
    const hex = Math.random().toString(36).substring(2, 6).toUpperCase();
    setParticipantId(`P-${hex}`);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!consentGiven) {
      setError('Participant research consent is mandatory. You must verify consent before proceeding.');
      return;
    }

    const finalPid = participantId.trim() || `P-${Math.random().toString(36).substring(2, 6).toUpperCase()}`;

    setIsSubmitting(true);
    try {
      const res = await apiClient.createSession({
        participant_id: finalPid,
        consent_given: consentGiven,
        self_reported_group: selfReportedGroup || undefined,
        label_source: labelSource,
        notes: notes.trim()
      });

      const fullSession = await apiClient.getSession(res.session_id);
      onSessionCreated(fullSession);
      onNavigate('capture');
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to register session.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-200 gap-2">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Participant Registration & Consent
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Standard Operating Procedure: Pseudonymous enrollment for 3-scan Left Thumb research protocol.
          </p>
        </div>
        <span className="self-start sm:self-auto text-[11px] font-semibold px-2.5 py-1 bg-slate-100 text-slate-600 rounded-md border border-slate-200">
          Step 1 of 4
        </span>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-xs flex items-start gap-3 shadow-2xs">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-bold text-rose-900 block mb-0.5">Registration Validation Error</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200 shadow-xs divide-y divide-slate-100">
        {/* Section 1: Participant Identity */}
        <div className="p-6 sm:p-7 space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-slate-100 text-slate-700 font-bold text-xs flex items-center justify-center">1</div>
            <h2 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              Participant Identification
            </h2>
          </div>

          <div className="space-y-1.5 max-w-xl">
            <label className="block text-xs font-bold text-slate-700">
              Pseudonymous Participant ID <span className="text-rose-500">*</span>
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={participantId}
                onChange={(e) => setParticipantId(e.target.value.toUpperCase())}
                placeholder="e.g. P-4A9F"
                className="flex-1 h-11 px-3.5 text-sm border border-slate-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-teal-500 outline-none uppercase font-mono shadow-2xs"
              />
              <button
                type="button"
                onClick={generatePseudonym}
                className="h-11 px-4 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 active:bg-slate-300 border border-slate-300 rounded-xl flex items-center gap-1.5 transition-colors shadow-2xs shrink-0"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Generate ID</span>
              </button>
            </div>
            <p className="text-[11px] text-slate-400">
              Mandatory pseudonymity. Do not enter names, institutional USNs, or contact details.
            </p>
          </div>
        </div>

        {/* Section 2: Informed Research Consent */}
        <div className="p-6 sm:p-7 space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-slate-100 text-slate-700 font-bold text-xs flex items-center justify-center">2</div>
            <h2 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              Informed Research Consent
            </h2>
          </div>

          <div className={`p-4 rounded-xl border transition-all ${
            consentGiven
              ? 'bg-emerald-50/70 border-emerald-300 ring-2 ring-emerald-500/10'
              : 'bg-amber-50/60 border-amber-200'
          }`}>
            <label className="flex items-start gap-3 cursor-pointer select-none">
              <input
                type="checkbox"
                id="consent"
                checked={consentGiven}
                onChange={(e) => setConsentGiven(e.target.checked)}
                className="mt-1 w-4.5 h-4.5 text-teal-600 border-slate-300 rounded focus:ring-teal-500 cursor-pointer shrink-0"
              />
              <div className="text-xs text-slate-800 leading-relaxed">
                <span className={`font-bold block mb-1 ${consentGiven ? 'text-emerald-950' : 'text-slate-900'}`}>
                  Research Consent Confirmed <span className="text-rose-500">*</span>
                </span>
                The participant has been explicitly informed that this software is an experimental academic prototype under evaluation, is NOT a clinical blood typing instrument, and will not provide diagnostic results. Fingerprint images are stored pseudonymously for scientific evaluation.
              </div>
            </label>
          </div>
        </div>

        {/* Section 3: Optional Research Metadata */}
        <div className="p-6 sm:p-7 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-slate-100 text-slate-700 font-bold text-xs flex items-center justify-center">3</div>
              <h2 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                Reference Metadata (Optional)
              </h2>
            </div>
            <span className="text-[11px] text-slate-400 italic">For algorithm evaluation only</span>
          </div>

          <div className="space-y-3">
            <label className="block text-xs font-semibold text-slate-700">
              Reference Blood Group Label:
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {(['A+', 'AB+', 'B+', 'O+'] as BloodGroup[]).map((bg) => {
                const isSelected = selfReportedGroup === bg;
                return (
                  <button
                    type="button"
                    key={bg}
                    onClick={() => setSelfReportedGroup(isSelected ? '' : bg)}
                    className={`h-11 flex items-center justify-center font-bold text-sm rounded-xl border transition-all ${
                      isSelected
                        ? 'bg-teal-700 text-white border-teal-800 shadow-xs ring-2 ring-teal-500/30'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100 hover:border-slate-300'
                    }`}
                  >
                    <span>{bg}</span>
                    {isSelected && <span className="ml-1.5 text-teal-200 text-xs">✓</span>}
                  </button>
                );
              })}
            </div>
          </div>

          {selfReportedGroup && (
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
              <label className="block text-xs font-semibold text-slate-700">
                Ground Truth Verification Level:
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <label className={`flex items-start gap-2.5 p-3 rounded-xl border cursor-pointer transition-all ${
                  labelSource === 'self_reported'
                    ? 'bg-white border-teal-500 shadow-2xs ring-1 ring-teal-500/20'
                    : 'bg-white/60 border-slate-200'
                }`}>
                  <input
                    type="radio"
                    name="labelSource"
                    value="self_reported"
                    checked={labelSource === 'self_reported'}
                    onChange={() => setLabelSource('self_reported')}
                    className="mt-0.5 text-teal-600 shrink-0"
                  />
                  <div>
                    <span className="font-bold text-slate-900 block">Self-Reported by Participant</span>
                    <span className="text-[11px] text-slate-500 leading-tight">Unverified participant recall (typical 5-10% error margin).</span>
                  </div>
                </label>

                <label className={`flex items-start gap-2.5 p-3 rounded-xl border cursor-pointer transition-all ${
                  labelSource === 'laboratory_verified'
                    ? 'bg-white border-teal-500 shadow-2xs ring-1 ring-teal-500/20'
                    : 'bg-white/60 border-slate-200'
                }`}>
                  <input
                    type="radio"
                    name="labelSource"
                    value="laboratory_verified"
                    checked={labelSource === 'laboratory_verified'}
                    onChange={() => setLabelSource('laboratory_verified')}
                    className="mt-0.5 text-teal-600 shrink-0"
                  />
                  <div>
                    <span className="font-bold text-slate-900 block">Lab Serology Record</span>
                    <span className="text-[11px] text-slate-500 leading-tight">Clinical agglutination report checked.</span>
                  </div>
                </label>
              </div>

              {labelSource === 'self_reported' && (
                <p className="text-[11px] text-amber-800 bg-amber-50 p-2.5 rounded-lg border border-amber-200">
                  Audit Flag: Self-reported labels will be tagged as <strong>UNVERIFIED</strong> to prevent confounding classification benchmarks.
                </p>
              )}
            </div>
          )}

          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-slate-700">
              Protocol & Acquisition Notes (Optional)
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              placeholder="e.g. Ambient lighting variance, dry skin on platen, participant left thumb injury history..."
              className="w-full px-3.5 py-2.5 text-xs border border-slate-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-teal-500 outline-none shadow-2xs"
            />
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-6 bg-slate-50/50 rounded-b-2xl flex flex-col-reverse sm:flex-row items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => onNavigate('dashboard')}
            className="w-full sm:w-auto px-4 py-2.5 text-xs font-semibold text-slate-600 hover:text-slate-800 transition-colors"
          >
            Cancel Session
          </button>
          <button
            type="submit"
            disabled={isSubmitting || !consentGiven}
            className={`w-full sm:w-auto h-11 px-6 text-xs font-bold text-white rounded-xl shadow-xs flex items-center justify-center gap-2 transition-all ${
              consentGiven && !isSubmitting
                ? 'bg-teal-700 hover:bg-teal-800 cursor-pointer'
                : 'bg-slate-300 text-slate-500 cursor-not-allowed'
            }`}
          >
            <span>{isSubmitting ? 'Registering...' : 'Proceed to Left Thumb Capture'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );
};

