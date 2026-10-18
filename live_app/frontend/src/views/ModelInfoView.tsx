import React from 'react';
import { ShieldAlert, Cpu, CheckCircle2, AlertTriangle, Layers, Database } from 'lucide-react';
import type { SystemStatusResponse } from '../types';

interface ModelInfoViewProps {
  status: SystemStatusResponse | null;
}

export const ModelInfoView: React.FC<ModelInfoViewProps> = ({ status }) => {
  const model = status?.model;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Model Architecture & Scientific Evidence Audit
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Detailed transparency on neural architecture, feature dimensionality, canonical mappings, and failure-mode audit evidence.
          </p>
        </div>
        <span className="self-start sm:self-auto text-xs px-3 py-1 rounded-full font-bold bg-teal-50 text-teal-800 border border-teal-200">
          Ensemble: {model?.folds_loaded ?? 5}-Fold Cross-Validation
        </span>
      </div>

      {/* Mandatory Standardized Scientific Statement */}
      <div className="p-6 bg-slate-900 text-white rounded-2xl shadow-xs space-y-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
          <span className="text-xs font-bold uppercase tracking-widest text-teal-400">
            Mandatory Standardized Scientific Conclusion
          </span>
        </div>
        <blockquote className="text-base sm:text-lg font-serif italic text-slate-100 leading-relaxed pl-2 border-l-2 border-teal-400">
          &ldquo;Under the evaluated datasets, fingerprint representations, and participant-level validation procedures, the experiments did not establish reliable, generalizable blood-group classification performance.&rdquo;
        </blockquote>
      </div>

      {/* 6 Mandatory Research Disclaimers & Boundaries */}
      <div className="p-6 bg-amber-50/80 border border-amber-300 rounded-2xl space-y-3.5 text-xs text-amber-950 shadow-xs">
        <div className="flex items-center gap-2 font-bold text-amber-900 text-sm">
          <ShieldAlert className="w-5 h-5 text-amber-700 shrink-0" />
          <span>Six Mandatory Epistemic Disclaimers & Ethical Boundaries:</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
          <div className="p-3 bg-white/70 rounded-xl border border-amber-200/60 leading-relaxed">
            <span className="font-bold text-amber-900 block mb-0.5">1. Research Prototype Only</span>
            The system is strictly an experimental computational research prototype created for academic investigation.
          </div>
          <div className="p-3 bg-white/70 rounded-xl border border-amber-200/60 leading-relaxed">
            <span className="font-bold text-amber-900 block mb-0.5">2. Zero Clinical Clearance</span>
            The models are not clinically validated and have zero diagnostic clearance from any medical regulatory authority.
          </div>
          <div className="p-3 bg-white/70 rounded-xl border border-amber-200/60 leading-relaxed">
            <span className="font-bold text-amber-900 block mb-0.5">3. Medical Treatment Prohibition</span>
            Fingerprints must NEVER be used to determine blood group for transfusions, surgery, medical treatment, or emergency triage.
          </div>
          <div className="p-3 bg-white/70 rounded-xl border border-amber-200/60 leading-relaxed">
            <span className="font-bold text-amber-900 block mb-0.5">4. Self-Reported Label Noise</span>
            Self-reported ground truth contains 5–10% recalled error margin, potentially introducing non-biological noise into benchmarks.
          </div>
          <div className="p-3 bg-white/70 rounded-xl border border-amber-200/60 leading-relaxed">
            <span className="font-bold text-amber-900 block mb-0.5">5. Sample Size & Selection Bias</span>
            Small cohort size (N=80) and model-selection bias constrain the statistical power of the empirical observations.
          </div>
          <div className="p-3 bg-white/70 rounded-xl border border-amber-200/60 leading-relaxed">
            <span className="font-bold text-amber-900 block mb-0.5">6. Epistemic Scope</span>
            Findings demonstrate that examined representations lack predictive utility; they do not claim that biological links are impossible across nature.
          </div>
        </div>
      </div>

      {/* Architecture Specifications */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 sm:p-7 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-teal-50 text-teal-700 rounded-xl">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">
                Active Architecture: {model?.name || 'EXP-06C Config A'}
              </h2>
              <span className="text-xs text-slate-500">
                Multi-scan fused DenseNet-121 with Supervised Contrastive Pretraining
              </span>
            </div>
          </div>
          <span className="self-start sm:self-auto text-xs px-3 py-1 rounded-full font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
            {(model?.folds_loaded ?? 0) > 0 ? `${model?.folds_loaded} Folds Operational` : 'Standby'}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center gap-1.5 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
              <Layers className="w-3.5 h-3.5 text-teal-600" />
              <span>Backbone</span>
            </div>
            <div className="font-bold text-slate-900 text-sm">DenseNet-121</div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              ImageNet pretrained, 7,037,504 frozen parameters. Single-pass shared extraction for all 3 scans.
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center gap-1.5 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
              <Database className="w-3.5 h-3.5 text-teal-600" />
              <span>Feature Fusion</span>
            </div>
            <div className="font-bold text-slate-900 text-sm">3,072 Dimensions</div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              Global Average Pooling across 3 Left Thumb impressions (3 × 1,024-d concatenation).
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center gap-1.5 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
              <Cpu className="w-3.5 h-3.5 text-teal-600" />
              <span>Projection Head</span>
            </div>
            <div className="font-bold text-slate-900 text-sm">Dense(128) Linear</div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              SupCon pretraining objective (temperature &tau;=0.05, contrastive loss weight &lambda;=0.50).
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center gap-1.5 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
              <CheckCircle2 className="w-3.5 h-3.5 text-teal-600" />
              <span>Classification Head</span>
            </div>
            <div className="font-bold text-slate-900 text-sm">Dense(512) &rarr; Dense(4)</div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              Dense(512, ReLU) &rarr; Dropout(0.5) &rarr; Dense(4, Softmax) ensemble averaged across 5 folds.
            </p>
          </div>
        </div>

        {/* Canonical vs Native Class Mapping Table */}
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
              Canonical vs. Native Checkpoint Index Reconciliation
            </h3>
            <span className="text-[11px] text-slate-400">
              Strictly verified against EXP-06C fold checkpoints
            </span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200 uppercase tracking-wider text-[10px]">
                <tr>
                  <th className="px-5 py-3">Canonical Blood Group</th>
                  <th className="px-5 py-3">Canonical UI Index</th>
                  <th className="px-5 py-3">Native Keras Output Index</th>
                  <th className="px-5 py-3">Mathematical Alignment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                <tr className="hover:bg-slate-50/60">
                  <td className="px-5 py-3 font-bold text-slate-900 font-mono">A+</td>
                  <td className="px-5 py-3 font-mono text-slate-700">0</td>
                  <td className="px-5 py-3 font-mono text-slate-700">0</td>
                  <td className="px-5 py-3 text-emerald-700 font-semibold">Direct Index Match</td>
                </tr>
                <tr className="hover:bg-slate-50/60">
                  <td className="px-5 py-3 font-bold text-slate-900 font-mono">AB+</td>
                  <td className="px-5 py-3 font-mono text-slate-700">1</td>
                  <td className="px-5 py-3 font-mono text-slate-700">2</td>
                  <td className="px-5 py-3 text-blue-700 font-semibold">Mapped in Predictor Adapter (Native 2 &rarr; Canonical 1)</td>
                </tr>
                <tr className="hover:bg-slate-50/60">
                  <td className="px-5 py-3 font-bold text-slate-900 font-mono">B+</td>
                  <td className="px-5 py-3 font-mono text-slate-700">2</td>
                  <td className="px-5 py-3 font-mono text-slate-700">1</td>
                  <td className="px-5 py-3 text-blue-700 font-semibold">Mapped in Predictor Adapter (Native 1 &rarr; Canonical 2)</td>
                </tr>
                <tr className="hover:bg-slate-50/60">
                  <td className="px-5 py-3 font-bold text-slate-900 font-mono">O+</td>
                  <td className="px-5 py-3 font-mono text-slate-700">3</td>
                  <td className="px-5 py-3 font-mono text-slate-700">3</td>
                  <td className="px-5 py-3 text-emerald-700 font-semibold">Direct Index Match</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Forensic Confounder Audit Findings */}
        <div className="p-5 bg-slate-50 rounded-2xl border border-slate-200 text-xs space-y-2.5">
          <div className="flex items-center gap-2 text-slate-900 font-bold text-sm">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            <span>Forensic Failure-Mode Discovery: The Scanner Border Optical Confounder</span>
          </div>
          <p className="text-slate-600 leading-relaxed">
            Historical machine learning experiments reported misleadingly high accuracy (78%–88%) due to <strong>scan-level leakage</strong> (splitting impressions of the same participant across train and test sets). When participant grouping was enforced via StratifiedGroupKFold, cross-validated accuracy dropped to ~53.75%.
          </p>
          <p className="text-slate-600 leading-relaxed">
            Subsequent failure-mode analysis demonstrated that <strong>non-ridge optical platen border brightness correlated with blood groups (ANOVA F=5.996, p=0.0006)</strong> due to temporal acquisition batch artifacts. When evaluated on the held-out locked test cohort where ambient lighting reversed, CNN accuracy collapsed to 33.33% (below the 36.25% majority-class baseline).
          </p>
        </div>
      </div>
    </div>
  );
};

