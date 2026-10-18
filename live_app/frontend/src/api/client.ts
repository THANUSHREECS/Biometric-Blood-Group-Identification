import axios from 'axios';
import type {
  SystemStatusResponse,
  ResearchSession,
  InferenceResult,
  ResearchStats,
  BloodGroup
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 120 seconds to prevent prematurely aborting active computation
  headers: {
    'Content-Type': 'application/json'
  }
});

export const apiClient = {
  // System & Health
  async getStatus(): Promise<SystemStatusResponse> {
    const res = await api.get<SystemStatusResponse>('/status');
    return res.data;
  },

  async getResearchStats(): Promise<ResearchStats> {
    const res = await api.get<{ success: boolean; stats: ResearchStats }>('/research/stats');
    return res.data.stats;
  },

  // Scanner Hardware & Bridge
  async getScannerStatus() {
    const res = await api.get('/scanner/status');
    return res.data;
  },

  async connectScanner() {
    const res = await api.post('/scanner/connect');
    return res.data;
  },

  async disconnectScanner() {
    const res = await api.post('/scanner/disconnect');
    return res.data;
  },

  async setScannerMode(mock: boolean) {
    const res = await api.post('/scanner/mode', { mock });
    return res.data;
  },

  // Research Session & Protocol
  async createSession(payload: {
    participant_id?: string;
    consent_given: boolean;
    self_reported_group?: BloodGroup | '';
    label_source?: 'self_reported' | 'laboratory_verified';
    notes?: string;
  }): Promise<{ session_id: string; participant_id: string }> {
    const res = await api.post('/research/session', payload);
    return res.data;
  },

  async getSession(sessionId: string): Promise<ResearchSession> {
    const res = await api.get<{ success: boolean; session: ResearchSession }>(`/research/session/${sessionId}`);
    return res.data.session;
  },

  async listSessions(limit = 50): Promise<ResearchSession[]> {
    const res = await api.get<{ success: boolean; total: number; sessions: ResearchSession[] }>(`/research/sessions?limit=${limit}`);
    return res.data.sessions;
  },

  // Capture impression (1, 2, or 3) for the session
  async captureImpression(sessionId: string, impressionNumber: number): Promise<{
    success: boolean;
    capture_id: string;
    impression_number: number;
    image_url: string;
    quality: any;
    is_acceptable: boolean;
    rejection_reason?: string;
    scanner_mode: string;
  }> {
    const res = await api.post('/research/capture', {
      session_id: sessionId,
      impression_number: impressionNumber
    });
    return res.data;
  },

  // Run 3-scan multi-scan inference
  async runInference(sessionId: string): Promise<InferenceResult> {
    const res = await api.post<InferenceResult>('/research/infer', {
      session_id: sessionId
    });
    return res.data;
  },

  // Image URL helper (handles raw filename, relative /api/image path, or absolute URL)
  getImageUrl(pathOrFilename?: string): string {
    if (!pathOrFilename) return '';
    if (pathOrFilename.startsWith('http://') || pathOrFilename.startsWith('https://')) {
      return pathOrFilename;
    }
    const clean = pathOrFilename.replace(/^\/?api\/image\//, '').replace(/^\/+/, '');
    return `${API_BASE}/image/${clean}`;
  }
};
