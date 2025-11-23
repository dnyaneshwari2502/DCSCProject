// Type definitions matching the Python backend response structure
export interface PredictionResponse {
  clickbait_prediction: string;
  clickbait_score: number;
  emotion_prediction: string;
  emotion_score: number;
  model_status: string;
}

export interface AnalysisState {
  isLoading: boolean;
  error: string | null;
  data: PredictionResponse | null;
}

export enum ScoreLevel {
  SAFE = 'SAFE',
  CAUTION = 'CAUTION',
  DANGER = 'DANGER'
}