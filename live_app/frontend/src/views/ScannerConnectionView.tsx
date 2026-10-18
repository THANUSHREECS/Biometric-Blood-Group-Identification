import React, { useState } from 'react';
import {
  Radio,
  RefreshCw,
  Power,
  PowerOff,
  CheckCircle2,
  AlertTriangle,
  Info,
  Server,
  ArrowRight
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { ScannerDeviceInfo, ScreenId } from '../types';

interface ScannerConnectionViewProps {
  scanner: ScannerDeviceInfo | null;
  onRefreshScanner: () => Promise<void>;
  onNavigate: (screen: ScreenId) => void;
}

export const ScannerConnectionView: React.FC<ScannerConnectionViewProps> = ({
  scanner,
  onRefreshScanner,
  onNavigate
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);

  const handleConnect = async () => {
    setIsProcessing(true);
    setFeedback(null);
    try {
      const res = await apiClient.connectScanner();
      await onRefreshScanner();
      if (res.connected) {
        setFeedback({ type: 'success', message: 'Connected to Mantra MFS100 optical scanner successfully.' });
      } else {
        setFeedback({
          type: 'error',
          message: 'Failed to connect physical scanner. Ensure 32-bit bridge is running on port 5100 and USB device is attached.'
        });
      }
    } catch (err: any) {
      setFeedback({ type: 'error', message: err.response?.data?.error || err.message || 'Connection request failed.' });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDisconnect = async () => {
    setIsProcessing(true);
    setFeedback(null);
    try {
      await apiClient.disconnectScanner();
      await onRefreshScanner();
      setFeedback({ type: 'info', message: 'Scanner released and disconnected.' });
    } catch (err: any) {
      setFeedback({ type: 'error', message: err.message || 'Failed to disconnect scanner.' });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleToggleMock = async (targetMock: boolean) => {
    setIsProcessing(true);
    setFeedback(null);
    try {
      const res = await apiClient.setScannerMode(targetMock);
      await onRefreshScanner();
      if (res.mock_mode) {
        setFeedback({
          type: 'info',
          message: 'DEMO MODE enabled. Authentic fingerprint samples will be used for demonstration without physical USB hardware.'
        });
      } else {
        setFeedback({
          type: 'info',
          message: 'Hardware mode enabled. Communicating with physical Mantra MFS100 via 32-bit HTTP bridge.'
        });
      }
    } catch (err: any) {
      setFeedback({ type: 'error', message: err.message || 'Failed to change scanner mode.' });
    } finally {
      setIsProcessing(false);
    }
  };

  const isConnected = scanner?.is_connected;
  const isMock = scanner?.mock_mode;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="pb-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Scanner Hardware & Bridge Status
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Mantra MFS100 500 DPI optical sensor integration via 32-bit local architecture bridge.
          </p>
        </div>
        <button
          onClick={() => onRefreshScanner()}
          disabled={isProcessing}
          className="self-start sm:self-auto px-4 py-2.5 text-xs font-semibold text-slate-700 bg-white border border-slate-300 rounded-xl shadow-2xs hover:bg-slate-50 active:bg-slate-100 flex items-center gap-1.5 transition-colors shrink-0"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isProcessing ? 'animate-spin text-teal-600' : ''}`} />
          <span>Probe Bridge</span>
        </button>
      </div>

      {feedback && (
        <div
          className={`p-4 rounded-xl border text-sm flex items-start gap-3 ${
            feedback.type === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
              : feedback.type === 'error'
              ? 'bg-rose-50 border-rose-200 text-rose-800'
              : 'bg-blue-50 border-blue-200 text-blue-800'
          }`}
        >
          {feedback.type === 'success' ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
          ) : feedback.type === 'error' ? (
            <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          ) : (
            <Info className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
          )}
          <span>{feedback.message}</span>
        </div>
      )}

      {/* Main Connection Status Card */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
        <div className="p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100">
          <div className="flex items-center gap-4">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center ${
                isMock
                  ? 'bg-amber-100 text-amber-700 ring-4 ring-amber-50'
                  : isConnected
                  ? 'bg-emerald-100 text-emerald-700 ring-4 ring-emerald-50'
                  : 'bg-rose-100 text-rose-700 ring-4 ring-rose-50'
              }`}
            >
              <Radio className="w-7 h-7" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold text-slate-900">
                  {isMock
                    ? 'DEMO MOCK EMULATION ACTIVE'
                    : isConnected
                    ? 'MFS100 HARDWARE CONNECTED'
                    : 'HARDWARE DISCONNECTED / OFFLINE'}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                {isMock
                  ? 'Using authentic biometric dataset samples from data/sample/'
                  : isConnected
                  ? 'Live optical capture ready on USB port'
                  : 'Physical scanner cannot be reached or bridge service is offline'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {!isMock && (
              isConnected ? (
                <button
                  onClick={handleDisconnect}
                  disabled={isProcessing}
                  className="px-4 py-2 text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg flex items-center gap-1.5 transition-colors"
                >
                  <PowerOff className="w-3.5 h-3.5" />
                  <span>Disconnect</span>
                </button>
              ) : (
                <button
                  onClick={handleConnect}
                  disabled={isProcessing}
                  className="px-4 py-2 text-xs font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-lg shadow-sm flex items-center gap-1.5 transition-colors"
                >
                  <Power className="w-3.5 h-3.5" />
                  <span>Connect Hardware</span>
                </button>
              )
            )}
          </div>
        </div>

        {/* Device Specifications Grid */}
        <div className="p-6 bg-slate-50/50 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div>
            <span className="text-slate-400 font-medium block">Hardware Make</span>
            <span className="font-bold text-slate-800">{scanner?.make || 'Mantra Softech'}</span>
          </div>
          <div>
            <span className="text-slate-400 font-medium block">Model Identifier</span>
            <span className="font-bold text-slate-800">{scanner?.model || 'MFS100'}</span>
          </div>
          <div>
            <span className="text-slate-400 font-medium block">Native Resolution</span>
            <span className="font-bold text-slate-800">500 DPI (316 x 354 px)</span>
          </div>
          <div>
            <span className="text-slate-400 font-medium block">Local Bridge Port</span>
            <span className="font-bold font-mono text-slate-800">127.0.0.1:5100</span>
          </div>
        </div>
      </div>

      {/* Mode Switching & Troubleshooting Card */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-2xs p-6 space-y-4">
        <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Server className="w-4 h-4 text-teal-600" />
          <span>Operational Mode Control</span>
        </h2>
        <p className="text-xs text-slate-600 leading-relaxed">
          The Mantra MFS100 driver uses a 32-bit Windows SDK DLL. A local out-of-process 32-bit HTTP bridge (`src/mfs100_bridge/bridge.py`) translates REST calls from this 64-bit application to the physical scanner.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
          <div
            onClick={() => handleToggleMock(false)}
            className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
              !isMock
                ? 'border-teal-600 bg-teal-50/30'
                : 'border-slate-200 bg-white hover:border-slate-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-slate-900">Physical USB Hardware Mode</span>
              {!isMock && <CheckCircle2 className="w-4 h-4 text-teal-600" />}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              Connects to the physical MFS100 scanner plugged into this machine via the 32-bit bridge.
            </p>
          </div>

          <div
            onClick={() => handleToggleMock(true)}
            className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
              isMock
                ? 'border-amber-500 bg-amber-50/30'
                : 'border-slate-200 bg-white hover:border-slate-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs text-amber-900">Development Mock Mode (DEMO)</span>
              {isMock && <CheckCircle2 className="w-4 h-4 text-amber-600" />}
            </div>
            <p className="text-[11px] text-amber-800 mt-1">
              Simulates capture using genuine sample fingerprints. Clearly labeled DEMO MODE.
            </p>
          </div>
        </div>

        {/* Troubleshooting Instructions */}
        {!isConnected && !isMock && (
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-2">
            <span className="font-bold text-slate-800 block">Bridge Troubleshooting Instructions:</span>
            <ol className="list-decimal list-inside space-y-1 text-slate-600">
              <li>Plug the Mantra MFS100 scanner into a USB port on this computer.</li>
              <li>Open a 32-bit command prompt: <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">C:\Python310-32\python.exe -m src.mfs100_bridge.bridge</code></li>
              <li>Verify the bridge is listening at <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">http://127.0.0.1:5100/status</code>.</li>
              <li>Click "Probe Bridge" above to establish communication.</li>
            </ol>
          </div>
        )}

        {isConnected && (
          <div className="pt-2 flex justify-end">
            <button
              onClick={() => onNavigate('register')}
              className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition-colors cursor-pointer"
            >
              <span>Proceed to Participant Registration</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
