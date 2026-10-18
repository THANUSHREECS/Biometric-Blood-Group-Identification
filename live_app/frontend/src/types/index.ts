export type BloodGroup = 'A+' | 'AB+' | 'B+' | 'O+';

export type ScreenId =
  | 'dashboard'
  | 'register'
  | 'scanner'
  | 'capture'
  | 'quality'
  | 'analysis'
  | 'results'
  | 'history'
  | 'system';

export interface Probabilities {
  'A+': number;
  'AB+': number;
  'B+': number;
  'O+': number;
}

export interface ImageQualityEvaluation {
  is_valid: boolean;
  is_acceptable: boolean;
  rejection_reason: string | null;
  width: number;
  height: number;
  format?: string;
  std_dev?: number;
  fg_coverage?: number;
  ridge_coherence?: number;
  ridge_contrast?: number;
  unusable_proportion?: number;
  quality_score_q?: number;
  quality_score_100?: number;
  nfiq_score?: number;
}

export interface ResearchCapture {
  capture_id: string;
  session_id: string;
  impression_number: number;
  finger_designation: string;
  image_filename: string;
  image_url: string;
  captured_at?: string;
  width?: number;
  height?: number;
  quality_score: number;
  nfiq_score: number;
  fg_coverage?: number;
  ridge_coherence?: number;
  ridge_contrast?: number;
  unusable_proportion?: number;
  is_acceptable: boolean;
  rejection_reason?: string | null;
}

export interface ResearchSession {
  session_id: string;
  participant_id: string;
  created_at: string;
  consent_given: boolean;
  self_reported_group?: BloodGroup | null;
  label_source: 'self_reported' | 'laboratory_verified';
  verification_status: 'unverified' | 'verified';
  notes?: string;
  status: 'CREATED' | 'CAPTURING' | 'COMPLETED' | 'FAILED';
  predicted_group?: BloodGroup | null;
  confidence?: number | null;
  canonical_probabilities?: Probabilities | string | null;
  model_name?: string;
  model_version?: string;
  inference_completed_at?: string;
  captures?: ResearchCapture[];
  capture_count?: number;
}

export interface ResearchStats {
  unique_participants: number;
  total_sessions: number;
  valid_captures: number;
  legacy_predictions: number;
}

export interface ScannerDeviceInfo {
  make?: string;
  model?: string;
  serial?: string;
  firmware?: string;
  mock_mode: boolean;
  status: string;
  is_connected: boolean;
  bridge_status?: string;
}

export interface SystemStatusResponse {
  status: string;
  timestamp: string;
  model: {
    name: string;
    tag: string;
    accuracy?: string;
    type: string;
    classes: string[];
    canonical_classes: string[];
    input_shape: number[];
    folds_loaded: number;
    ADVISORY_CONFIRMATION_THRESHOLD: number;
    advisory_layer?: {
      enabled: boolean;
      threshold: number;
      status: string;
    };
    research_disclaimer?: string;
  };
  scanner: ScannerDeviceInfo;
  research_stats: ResearchStats;
  disclaimer: string;
}

export interface InferenceResult {
  success: boolean;
  session_id: string;
  participant_id: string;
  created_at: string;
  self_reported_group?: BloodGroup | null;
  label_source?: string;
  verification_status?: string;
  predicted_group: BloodGroup;
  confidence: number;
  probabilities: Probabilities;
  classes: BloodGroup[];
  model_name: string;
  feature_dims: string;
  status: string;
  message: string;
  is_low_confidence: boolean;
  advisory?: {
    needs_confirmation: boolean;
    level: string;
    threshold: number;
    message: string;
  };
  captures: {
    impression_number: number;
    image_url: string;
    image_filename?: string;
    quality_score: number;
    nfiq_score: number;
    ridge_coherence?: number;
    fg_coverage?: number;
  }[];
  research_disclaimer: string;
}
