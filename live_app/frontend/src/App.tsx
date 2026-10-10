import { useState, useEffect } from 'react';
import axios from 'axios';
import { Fingerprint, AlertTriangle, CheckCircle, ChevronRight, Activity, ArrowRight, RotateCcw, UserPlus, Database } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000/api/mfs100';

type BloodGroup = 'A+' | 'B+' | 'AB+' | 'O+';

interface Probabilities {
  'A+': number;
  'B+': number;
  'AB+': number;
  'O+': number;
}

interface ScanData {
  capture_number: number;
  prediction: BloodGroup;
  confidence: number;
  probabilities: Probabilities;
}

interface FinalResult {
  final_prediction: BloodGroup;
  final_confidence: number;
  combined_probabilities: Probabilities;
  consistency: {
    level: string;
    agreement_count: number;
  };
  scans: ScanData[];
}

function App() {
  const [appMode, setAppMode] = useState<'select' | 'test' | 'research'>('select');
  const [appState, setAppState] = useState<'welcome' | 'scanning' | 'result'>('welcome');
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  const [currentScan, setCurrentScan] = useState<number>(1);
  const [scans, setScans] = useState<ScanData[]>([]);
  const [scanStatus, setScanStatus] = useState<string>('');
  const [isScanning, setIsScanning] = useState<boolean>(false);
  
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scannerStatus, setScannerStatus] = useState<any>(null);
  const [isCheckingScanner, setIsCheckingScanner] = useState<boolean>(true);

  // Research Mode State
  const [participantId, setParticipantId] = useState<string>('');
  const [consent, setConsent] = useState<boolean>(false);
  const [actualBg, setActualBg] = useState<string>('');
  const [verificationSource, setVerificationSource] = useState<string>('');
  const [metadataSaved, setMetadataSaved] = useState<boolean>(false);

  const checkScannerStatus = async () => {
    setIsCheckingScanner(true);
    try {
      const res = await axios.get(`${API_BASE}/status`);
      setScannerStatus(res.data);
    } catch (err: any) {
      setScannerStatus({ scanner_detected: false, error: 'Cannot connect to backend server.' });
    } finally {
      setIsCheckingScanner(false);
    }
  };

  useEffect(() => {
    checkScannerStatus();
  }, []);

  const startSession = async () => {
    setError(null);
    try {
      if (appMode === 'research') {
        if (!participantId.trim()) {
           setError("Participant ID is required.");
           return;
        }
        if (!consent) {
           setError("Consent is required.");
           return;
        }
        const res = await axios.post(`${API_BASE}/research/start-session`, {
           participant_id: participantId,
           consent: true
        });
        setSessionId(res.data.session_id);
      } else {
        const res = await axios.post(`${API_BASE}/start-session`);
        setSessionId(res.data.session_id);
      }
      setAppState('scanning');
      setCurrentScan(1);
      setScans([]);
      setScanStatus('Waiting for fingerprint...');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to start session.');
    }
  };

  const captureAndProcess = async () => {
    if (!sessionId) return;
    setIsScanning(true);
    setError(null);
    setScanStatus('Capturing...');
    
    try {
      const captureRes = await axios.post(`${API_BASE}/capture`, {
        session_id: sessionId,
        scan_number: currentScan
      });
      
      setScanStatus('Processing fingerprint...');
      
      const processRes = await axios.post(`${API_BASE}/process`, {
        session_id: sessionId,
        scan_number: currentScan,
        image_path: captureRes.data.image_path
      });
      
      setScanStatus('Scan completed');
      setScans([...scans, {
        capture_number: currentScan,
        prediction: processRes.data.prediction,
        confidence: processRes.data.confidence,
        probabilities: processRes.data.probabilities
      }]);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Capture failed.');
      setScanStatus('Error');
    } finally {
      setIsScanning(false);
    }
  };

  const finalizeSession = async () => {
    if (!sessionId) return;
    setIsScanning(true);
    setScanStatus('Calculating combined result...');
    try {
      const res = await axios.post(`${API_BASE}/finalize`, { session_id: sessionId });
      setFinalResult(res.data);
      setAppState('result');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Finalize failed.');
    } finally {
      setIsScanning(false);
    }
  };

  const saveMetadata = async () => {
    if (!actualBg || !verificationSource) {
       setError("Please select blood group and verification source.");
       return;
    }
    setError(null);
    setIsScanning(true);
    try {
      await axios.post(`${API_BASE}/research/save-metadata`, {
        session_id: sessionId,
        blood_group: actualBg,
        verification_source: verificationSource
      });
      setMetadataSaved(true);
    } catch (err: any) {
       setError(err.response?.data?.detail || err.message || 'Failed to save metadata.');
    } finally {
      setIsScanning(false);
    }
  };

  const handleNextScan = () => {
    if (currentScan < 3) {
      setCurrentScan(currentScan + 1);
      setScanStatus('Waiting for fingerprint...');
    } else {
      finalizeSession();
    }
  };

  const resetNewPerson = () => {
    if (appMode === 'research') {
       setAppMode('select');
    }
    setAppState('welcome');
    setSessionId(null);
    setCurrentScan(1);
    setScans([]);
    setFinalResult(null);
    setError(null);
    setParticipantId('');
    setConsent(false);
    setActualBg('');
    setVerificationSource('');
    setMetadataSaved(false);
  };

  const renderModeSelect = () => (
    <div className="card welcome-card">
      <div className="icon-wrapper">
        <Fingerprint className="icon-large" />
      </div>
      <h1 className="title">Select Module</h1>
      <div className="action-buttons" style={{ flexDirection: 'column', gap: '1rem', marginTop: '2rem' }}>
        <button onClick={() => { setAppMode('test'); setAppState('welcome'); }} className="btn btn-primary btn-large">
          <Activity className="icon-inline" /> LIVE TEST MODE
        </button>
        <button onClick={() => { setAppMode('research'); setAppState('welcome'); }} className="btn btn-secondary btn-large">
          <Database className="icon-inline" /> RESEARCH DATA COLLECTION
        </button>
      </div>
    </div>
  );

  const renderWelcome = () => (
    <div className="card welcome-card">
      <div className="icon-wrapper">
        {appMode === 'research' ? <Database className="icon-large" /> : <Fingerprint className="icon-large" />}
      </div>
      
      <h1 className="title">{appMode === 'research' ? 'FINAL MFS100 RESEARCH DATA COLLECTION' : 'Biometric Blood Group Detection'}</h1>
      <p className="subtitle">{appMode === 'research' ? 'Independent real-world validation dataset collection' : 'AI-assisted fingerprint-based research prototype'}</p>
      
      <div className="status-container">
        {isCheckingScanner ? (
          <div className="status-box status-checking">Checking scanner status...</div>
        ) : scannerStatus?.scanner_detected ? (
          <div className="status-box status-success">
            <CheckCircle className="icon-small icon-success" /> MFS100 SCANNER CONNECTED ✓
          </div>
        ) : (
          <div className="status-box status-error">
            <div className="status-error-header">
              <AlertTriangle className="icon-small icon-error" /> ✕ MFS100 SCANNER NOT DETECTED
            </div>
            <p className="status-error-msg">{scannerStatus?.error || 'Unknown error'}</p>
            <button onClick={checkScannerStatus} className="btn btn-secondary btn-full">RETRY SCANNER</button>
          </div>
        )}
      </div>

      {appMode === 'research' && (
        <div className="instruction-box" style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.5rem' }}>Participant ID</label>
            <input 
               type="text" 
               placeholder="e.g. P001" 
               value={participantId} 
               onChange={(e) => setParticipantId(e.target.value)}
               className="input-field"
               style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', border: '1px solid #ccc' }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input 
              type="checkbox" 
              id="consent" 
              checked={consent} 
              onChange={(e) => setConsent(e.target.checked)} 
              style={{ width: '20px', height: '20px' }}
            />
            <label htmlFor="consent" style={{ fontSize: '0.9rem' }}>I confirm that participant consent has been obtained.</label>
          </div>
          <div style={{ marginTop: '1rem' }}>
             <strong>Finger:</strong> LEFT THUMB ✓
          </div>
        </div>
      )}

      <button onClick={startSession} disabled={!scannerStatus?.scanner_detected} className="btn btn-primary btn-large">
        {appMode === 'research' ? '[ START 3-SCAN COLLECTION ]' : 'START SCANNING'} <ArrowRight className="icon-inline" />
      </button>

      {error && <div className="error-box">{error}</div>}

      <p className="disclaimer">
        Research data collection only. Fingerprint-based blood-group prediction is
        experimental and is not medically validated. Do not use model predictions
        for medical decisions.
      </p>
    </div>
  );

  const renderScanning = () => {
    const currentScanData = scans.find(s => s.capture_number === currentScan);
    const isCompleted = !!currentScanData;

    return (
      <div className="card scan-card">
        <div className="scan-header">
          <h2>SCAN {currentScan} OF 3</h2>
          <span className="tag">LEFT THUMB</span>
        </div>

        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${(currentScan / 3) * 100}%` }}></div>
        </div>

        {!isCompleted ? (
          <div className="scan-active">
            <div className={`fingerprint-indicator ${isScanning ? 'pulsing' : ''}`}>
               <Fingerprint className="icon-giant" />
            </div>
            <p className="scan-instruction">
              {currentScan === 1 ? "Place your LEFT THUMB firmly on the MFS100 scanner." : "Place the SAME LEFT THUMB on the scanner again."}
            </p>
            <p className="scan-status">{scanStatus}</p>

            <button 
              onClick={captureAndProcess} 
              disabled={isScanning}
              className="btn btn-primary"
            >
              {isScanning ? 'CAPTURING...' : 'CAPTURE NOW'}
            </button>
            {error && <div className="error-box">{error}</div>}
          </div>
        ) : (
          <div className="scan-completed">
            <div className="success-header">
              <CheckCircle className="icon-medium" />
              <h3>Scan {currentScan} Complete</h3>
            </div>
            
            {appMode !== 'research' && (
              <div className="prediction-box">
                <div className="pred-left">
                  <span className="label">Predicted</span>
                  <span className="value">{currentScanData.prediction}</span>
                </div>
                <div className="pred-right">
                  <span className="label">Confidence</span>
                  <span className="value">{(currentScanData.confidence * 100).toFixed(2)}%</span>
                </div>
              </div>
            )}

            <button onClick={handleNextScan} className="btn btn-primary btn-large" style={{ marginTop: '2rem' }}>
              {currentScan < 3 ? `PROCEED TO SCAN ${currentScan + 1}` : 'FINALIZE COLLECTION'} <ChevronRight className="icon-inline" />
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderResult = () => {
    if (!finalResult) return null;

    if (appMode === 'research') {
       return (
        <div className="card result-card" style={{ textAlign: 'left' }}>
          <div className="result-header">
            <h1>COLLECTION COMPLETE</h1>
            <div className="result-meta">
              Participant: {participantId} | LEFT THUMB | 3/3 Scans ✓
            </div>
          </div>

          <div style={{ marginBottom: '1.5rem', background: '#f8f9fa', padding: '1rem', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 1rem 0' }}>Model prediction</h3>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#3b82f6' }}>
               {finalResult.final_prediction} ({(finalResult.final_confidence * 100).toFixed(1)}%)
            </div>
          </div>

          {!metadataSaved ? (
            <div style={{ background: '#fff', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
              <h3 style={{ margin: '0 0 1rem 0' }}>Record Ground Truth</h3>
              
              <div style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.5rem' }}>Confirmed blood group:</label>
                <select 
                  value={actualBg} 
                  onChange={(e) => setActualBg(e.target.value)}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', border: '1px solid #ccc' }}
                >
                  <option value="">-- Select --</option>
                  <option value="A+">A+</option>
                  <option value="B+">B+</option>
                  <option value="AB+">AB+</option>
                  <option value="O+">O+</option>
                </select>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '0.5rem' }}>Verification source:</label>
                <select 
                  value={verificationSource} 
                  onChange={(e) => setVerificationSource(e.target.value)}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', border: '1px solid #ccc' }}
                >
                  <option value="">-- Select --</option>
                  <option value="lab_report">Lab report</option>
                  <option value="medical_record">Medical record</option>
                  <option value="self_reported">Self-reported</option>
                </select>
              </div>

              {error && <div className="error-box">{error}</div>}

              <button onClick={saveMetadata} disabled={isScanning} className="btn btn-primary btn-large btn-full">
                [ SAVE PARTICIPANT ]
              </button>
            </div>
          ) : (
            <div className="status-box status-success" style={{ marginBottom: '2rem' }}>
              <CheckCircle className="icon-small icon-success" /> Participant saved successfully.
            </div>
          )}

          <div className="disclaimer-box" style={{ marginTop: '2rem' }}>
            <p><strong>Research Notice</strong> — Research data collection only. Fingerprint-based blood-group prediction is experimental and is not medically validated. Do not use model predictions for medical decisions.</p>
          </div>

          <div className="action-buttons" style={{ marginTop: '1.5rem' }}>
             <button onClick={resetNewPerson} className="btn btn-secondary btn-full">
               <UserPlus className="icon-inline" /> BACK TO MENU
             </button>
          </div>
        </div>
       );
    }

    const isLowConsistency = finalResult.consistency.level === 'LOW';

    return (
      <div className="card result-card">
        <div className="result-header">
          <h1>FINAL RESULT</h1>
          <div className="result-meta">
            LEFT THUMB • 3 SCANS COMPLETED <CheckCircle className="icon-tiny" />
          </div>
        </div>

        <div className="result-body">
          <div className="final-prediction-box">
            <div className="pred-left">
              <span className="label">Predicted Blood Group</span>
              <span className="value-large">{finalResult.final_prediction}</span>
            </div>
            <div className="pred-right">
              <span className="label">Combined Confidence</span>
              <span className="value-large">{(finalResult.final_confidence * 100).toFixed(2)}%</span>
            </div>
          </div>

          <div className="action-buttons">
            <button onClick={() => { setAppState('scanning'); setCurrentScan(1); setScans([]); }} className="btn btn-secondary">
              <RotateCcw className="icon-inline" /> SCAN AGAIN
            </button>
            <button onClick={resetNewPerson} className="btn btn-primary">
              <UserPlus className="icon-inline" /> NEW PERSON
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      {appMode === 'select' && renderModeSelect()}
      {appMode !== 'select' && appState === 'welcome' && renderWelcome()}
      {appMode !== 'select' && appState === 'scanning' && renderScanning()}
      {appMode !== 'select' && appState === 'result' && renderResult()}
    </div>
  );
}

export default App;
