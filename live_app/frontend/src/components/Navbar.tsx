import React from 'react';
import {
  Fingerprint,
  LayoutDashboard,
  UserPlus,
  Radio,
  Camera,
  CheckCircle2,
  Cpu,
  Award,
  History,
  Info
} from 'lucide-react';
import type { ScreenId, ScannerDeviceInfo } from '../types';

interface NavbarProps {
  currentScreen: ScreenId;
  onNavigate: (screen: ScreenId) => void;
  scanner: ScannerDeviceInfo | null;
  onOpenScannerModal?: () => void;
  onStartNewSession: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentScreen,
  onNavigate,
  scanner,
  onStartNewSession
}) => {
  const navItems: { id: ScreenId; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-3.5 h-3.5" /> },
    { id: 'register', label: 'Register', icon: <UserPlus className="w-3.5 h-3.5" /> },
    { id: 'scanner', label: 'Scanner', icon: <Radio className="w-3.5 h-3.5" /> },
    { id: 'capture', label: 'Capture', icon: <Camera className="w-3.5 h-3.5" /> },
    { id: 'quality', label: 'Quality', icon: <CheckCircle2 className="w-3.5 h-3.5" /> },
    { id: 'analysis', label: 'Analysis', icon: <Cpu className="w-3.5 h-3.5" /> },
    { id: 'results', label: 'Results', icon: <Award className="w-3.5 h-3.5" /> },
    { id: 'history', label: 'History', icon: <History className="w-3.5 h-3.5" /> },
    { id: 'system', label: 'Model Info', icon: <Info className="w-3.5 h-3.5" /> }
  ];

  const isScannerConnected = scanner?.is_connected;
  const isMock = scanner?.mock_mode;

  return (
    <header className="bg-slate-900 text-slate-100 shadow-md border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-[1600px] w-full mx-auto px-6 sm:px-8 lg:px-10">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Project Branding */}
          <div
            className="flex items-center gap-3 cursor-pointer group"
            onClick={() => onNavigate('dashboard')}
          >
            <div className="w-10 h-10 rounded-lg bg-teal-600 flex items-center justify-center text-white shadow-sm group-hover:bg-teal-500 transition-colors">
              <Fingerprint className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-lg tracking-tight text-white">
                  HemoPrint <span className="text-teal-400">AI</span>
                </span>
                <span className="px-1.5 py-0.5 text-[10px] uppercase font-semibold bg-slate-800 text-teal-300 rounded border border-teal-800">
                  Research Prototype
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium">
                Optical Fingerprint Biometrics & Serology Research
              </p>
            </div>
          </div>

          {/* Center Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = currentScreen === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-teal-700/80 text-white font-semibold shadow-xs border border-teal-500/40'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/80'
                  }`}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Action Controls: Scanner Status & Quick Action */}
          <div className="flex items-center gap-3">
            {/* Real Scanner Connection Status Pill */}
            <button
              onClick={() => onNavigate('scanner')}
              title="Click to view scanner connection details"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                isMock
                  ? 'bg-amber-950/40 border-amber-600 text-amber-300 hover:bg-amber-950/60'
                  : isScannerConnected
                  ? 'bg-emerald-950/40 border-emerald-500 text-emerald-300 hover:bg-emerald-950/60'
                  : 'bg-rose-950/40 border-rose-600 text-rose-300 hover:bg-rose-950/60'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isMock
                    ? 'bg-amber-400 animate-pulse'
                    : isScannerConnected
                    ? 'bg-emerald-400'
                    : 'bg-rose-500'
                }`}
              />
              <span className="hidden sm:inline">
                {isMock ? 'DEMO MOCK MODE' : isScannerConnected ? 'MFS100 CONNECTED' : 'MFS100 DISCONNECTED'}
              </span>
            </button>

            {/* Start Session Quick Action */}
            <button
              onClick={onStartNewSession}
              className="bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold px-3 py-1.5 rounded-md shadow-xs flex items-center gap-1.5 transition-colors border border-teal-500"
            >
              <UserPlus className="w-3.5 h-3.5" />
              <span>New Session</span>
            </button>
          </div>
        </div>

        {/* Mobile/Tablet Screen Selector */}
        <div className="flex lg:hidden overflow-x-auto py-2 gap-1 border-t border-slate-800 scrollbar-none">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded text-[11px] whitespace-nowrap ${
                currentScreen === item.id
                  ? 'bg-teal-700 text-white font-semibold'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
