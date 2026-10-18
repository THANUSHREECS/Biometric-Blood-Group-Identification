import React from 'react';
import { ShieldAlert } from 'lucide-react';

export const ResearchDisclaimer: React.FC = () => {
  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5 text-xs text-amber-900 flex items-center justify-between shadow-xs">
      <div className="max-w-7xl mx-auto w-full flex items-center gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-700 shrink-0" />
        <div className="leading-snug">
          <span className="font-bold tracking-wide text-amber-950 uppercase mr-2">
            RESEARCH PROTOTYPE — STRICTLY NOT FOR MEDICAL OR CLINICAL USE:
          </span>
          Biometric fingerprint classification has not established reliable or clinically generalizable blood-group discrimination. 
          Never use predictions for medical diagnosis, transfusion, surgery, or emergency triage. Standard laboratory serological agglutination testing is mandatory.
        </div>
      </div>
    </div>
  );
};
