import React, { useState } from 'react';
import {
  Layers,
  AlertTriangle,
  Play,
  ArrowRight
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { ResearchSession, InferenceResult, ScreenId } from '../types';

interface ExperimentalAnalysisViewProps {
  session: ResearchSession | null;
  onInferenceCompleted: (result: InferenceResult) => void;
  onNavigate: (screen: ScreenId) => void;
}

export const ExperimentalAnalysisView: React.FC<ExperimentalAnalysisViewProps> = ({
  session,
  onInferenceCompleted,
  onNavigate
}) => {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!session) {
    return (
      <div className="max-w-xl mx-auto p-8 bg-white rounded-xl border border-slate-200 text-center space-y-4">
        <AlertTriangle className="w-10 h-10 text-amber-500 mx-auto" />
        <h2 className="text-lg font-bold text-slate-900">No Active Session Selected</h2>
        <p className="text-xs text-slate-500">
          Please select or register a research participant to evaluate.
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

  const captures = session.captures || [];
  const validCaptures = captures.filter((c) => c.is_acceptable);
  const hasThreeValid = validCaptures.length >= 3;

  const handleExecuteInference = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const result = await apiClient.runInference(session.session_id);
      onInferenceCompleted(result);
      onNavigate('results');
    } catch (err: any) {
      setError(
        err.response?.data?.error ||
        err.message ||
        'Inference failed across the 5-fold ensemble.'
      );
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6 animate-fade-in">
      <div className="pb-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Multi-Scan Feature Fusion & Inference
            </h1>
            <span className="px-2 py-0.5 text-xs font-bold bg-teal-100 text-teal-800 rounded">
              Participant: {session.participant_id}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Execution of the verified EXP-06C Config A 5-fold ensemble with concatenated 3072-dimensional embeddings.
          </p>
        </div>

        {session.status === 'COMPLETED' && (
          <button
            onClick={() => onNavigate('results')}
            className="px-3 py-1.5 text-xs font-semibold text-teal-700 bg-teal-50 border border-teal-200 rounded-lg hover:bg-teal-100 flex items-center gap-1.5"
          >
            <span>View Previous Results</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-xs flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-bold block">Inference Error</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Pipeline Stage Architecture Diagram */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-2xs p-6 space-y-6">
        <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Layers className="w-4 h-4 text-teal-600" />
          <span>Verified Inference Pipeline Flow</span>
        </h2>

        {/* 3 Input Thumbnails */}
        <div className="grid grid-cols-3 gap-4">
          {[1, 2, 3].map((num) => {
            const cap = captures.find((c) => c.impression_number === num && c.is_acceptable);
            return (
              <div
                key={num}
                className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-center space-y-2"
              >
                <div className="text-[11px] font-bold text-slate-700 uppercase">
                  Impression {num} (Left Thumb)
                </div>
                <div className="h-32 bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center p-1.5 border border-slate-200">
                  {cap ? (
                    <img
                      src={apiClient.getImageUrl(cap.image_filename || cap.image_url)}
                      alt={`Imp ${num}`}
                      className="h-full max-h-28 object-contain rounded"
                    />
                  ) : (
                    <span className="text-[11px] text-slate-400 italic">Missing scan</span>
                  )}
                </div>
                <div className="text-[10px] text-slate-500 font-medium">
                  {cap ? `Q: ${cap.quality_score}/100 | NFIQ: ${cap.nfiq_score}` : 'Pending capture'}
                </div>
              </div>
            );
          })}
        </div>

        {/* Feature Fusion Specification Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 text-xs">
          <div className="p-3.5 bg-slate-50 rounded-lg border border-slate-200">
            <span className="text-slate-400 font-medium block">1. C5 Preprocessing</span>
            <span className="font-bold text-slate-800">
              Alpha composite &rarr; Platen Crop &rarr; CLAHE &rarr; 112x112
            </span>
          </div>

          <div className="p-3.5 bg-slate-50 rounded-lg border border-slate-200">
            <span className="text-slate-400 font-medium block">2. Feature Extraction</span>
            <span className="font-bold text-slate-800">
              DenseNet-121 Frozen (GAP: 3 x 1024 = 3072-d)
            </span>
          </div>

          <div className="p-3.5 bg-slate-50 rounded-lg border border-slate-200">
            <span className="text-slate-400 font-medium block">3. Linear Head & 5-Fold Ensemble</span>
            <span className="font-bold text-slate-800">
              Dense(512) &rarr; Dense(4) [A+, AB+, B+, O+]
            </span>
          </div>
        </div>

        {/* Action Trigger Box */}
        <div className="p-5 bg-slate-50 rounded-xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="text-xs font-bold text-slate-900">
              {isRunning
                ? 'Active computation: Extracting 3072-d feature fusion & evaluating 5 folds...'
                : hasThreeValid
                ? 'Three impressions verified and ready for feature fusion.'
                : 'Protocol incomplete: exactly 3 valid Left Thumb impressions required.'}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              {isRunning
                ? 'Please wait while DenseNet-121 extracts embeddings and linear heads compute predictions.'
                : 'Inference evaluates the participant embedding across all 5 cross-validation folds.'}
            </p>
          </div>

          <button
            onClick={handleExecuteInference}
            disabled={!hasThreeValid || isRunning}
            className={`px-6 py-3 rounded-xl text-xs font-bold text-white shadow-sm flex items-center justify-center gap-2 transition-all ${
              hasThreeValid && !isRunning
                ? 'bg-teal-600 hover:bg-teal-700 cursor-pointer shadow-teal-700/20 shadow-md'
                : 'bg-slate-300 cursor-not-allowed opacity-80'
            }`}
          >
            <Play className={`w-4 h-4 ${isRunning ? 'animate-spin' : ''}`} />
            <span>{isRunning ? 'Running Ensemble (1-2s)...' : 'Execute Research Inference'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
