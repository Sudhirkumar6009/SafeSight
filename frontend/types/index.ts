// Video Types
export interface Video {
  _id: string;
  filename: string;
  originalName: string | null;
  path: string;
  size: number;
  mimetype: string;
  duration?: number;
  resolution?: {
    width: number;
    height: number;
  };
  fps?: number;
  status: "uploaded" | "processing" | "completed" | "failed";
  uploadedAt: string;
  processedAt?: string;
  metadata?: Record<string, any>;
}

// Prediction Types
export interface Prediction {
  _id: string;
  videoId: string | Video;
  modelId: string | ModelConfig;
  classification: "violence" | "non-violence";
  confidence: number;
  probabilities: {
    violence: number;
    nonViolence: number;
  };
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    inferenceTime?: number;
  };
  frameAnalysis?: {
    totalFrames: number;
    violentFrames: number;
    nonViolentFrames: number;
    frameScores?: number[];
  };
  status: "pending" | "running" | "completed" | "failed";
  error?: string;
  createdAt: string;
  completedAt?: string;
}

// Model Configuration Types
export interface ModelConfig {
  _id: string;
  name: string;
  description?: string;
  modelPath: string;
  architecture: string;
  version: string;
  inputSize: {
    frames: number;
    height: number;
    width: number;
  };
  classes: string[];
  isActive: boolean;
  isLoaded: boolean;
  performance?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    avgInferenceTime?: number;
    totalPredictions?: number;
  };
  createdAt: string;
  updatedAt: string;
  loadedAt?: string;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
  };
}

// Model Status Response
export interface ModelStatusResponse {
  hasActiveModel: boolean;
  model: {
    id: string;
    name: string;
    modelPath: string;
    architecture: string;
    isLoaded: boolean;
    loadedAt?: string;
  } | null;
  mlService: {
    isLoaded: boolean;
    currentModel?: {
      path: string;
      architecture: string;
      loadedAt: string;
    };
    gpuAvailable: boolean;
    gpuMemory?: {
      total: number;
      used: number;
      free: number;
    };
  };
}

// Model Metrics Response
export interface ModelMetricsResponse {
  model: {
    id: string;
    name: string;
  };
  performance: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    avgInferenceTime?: number;
    totalPredictions?: number;
  };
  mlMetrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    confusionMatrix?: {
      truePositive: number;
      trueNegative: number;
      falsePositive: number;
      falseNegative: number;
    };
    totalPredictions?: number;
    avgInferenceTime?: number;
  };
}

// Prediction Stats
export interface PredictionStats {
  total: number;
  violent: number;
  nonViolent: number;
  completed: number;
  failed: number;
  avgConfidence: number;
  avgInferenceTime: number;
}

// Health Check Response
export interface HealthResponse {
  status: "healthy" | "unhealthy";
  timestamp: string;
  uptime: number;
  services: {
    api: {
      status: string;
      version: string;
      environment: string;
    };
    mongodb: {
      status: string;
      host: string;
    };
    mlService: {
      status: string;
      url: string;
    };
  };
  memory: {
    rss: string;
    heapTotal: string;
    heapUsed: string;
  };
}

// Upload Progress
export interface UploadProgress {
  percentage: number;
  loaded: number;
  total: number;
}

// Inference Request
export interface InferenceRequest {
  videoId: string;
}

// Model Load Request
export interface ModelLoadRequest {
  modelPath: string;
  architecture?: string;
  name?: string;
  description?: string;
}

// ============================================
// RTSP Stream Types
// ============================================

export interface Stream {
  id: string;
  name: string;
  url: string;
  rtsp_url: string; // Alias for url
  stream_type: "rtsp" | "rtmp" | "webcam" | "file";
  type?: string; // Backend sends 'type' as well
  location?: string;
  is_active: boolean;
  is_running?: boolean;
  is_connected?: boolean;
  inference_enabled?: boolean;
  status:
    | "online"
    | "offline"
    | "error"
    | "connecting"
    | "running" // Active stream
    | "stopped" // Inactive stream
    | "starting"
    | "stopping"
    | "reconnecting"; // Attempting to reconnect
  last_frame_at?: string;
  last_frame_time?: string; // Backend alias
  error_message?: string;
  custom_threshold?: number;
  custom_window_seconds?: number;
  frame_count?: number;
  buffer_size?: number;
  reconnect_attempts?: number;
  created_at?: string;
  updated_at?: string;
  pending_events?: number;
  last_prediction?: {
    violence_score: number;
    non_violence_score: number;
    is_violent: boolean;
    timestamp: string;
    stream_id?: string;
    stream_name?: string;
  };
}

export interface StreamCreateRequest {
  name: string;
  url?: string;
  rtsp_url?: string;
  stream_type?: "rtsp" | "rtmp" | "webcam" | "file";
  location?: string;
  auto_start?: boolean;
  inference_enabled?: boolean;
  custom_threshold?: number;
}

export interface StreamStats {
  stream_id: string;
  is_running: boolean;
  is_connected: boolean;
  frame_count: number;
  last_frame_time?: string;
  ingestion: {
    status: string;
    frame_count: number;
    buffer_size: number;
  };
  detector: {
    phase: string;
    total_events: number;
    current_event_id?: string;
  };
}

// ============================================
// Violence Event Types
// ============================================

export type EventSeverity = "low" | "medium" | "high" | "critical";
export type EventStatus =
  | "NEW"
  | "PENDING"
  | "CONFIRMED"
  | "DISMISSED"
  | "REVIEWED"
  | "AUTO_DISMISSED"
  | "ACTION_EXECUTED"
  | "NO_ACTION_REQUIRED";

export interface ViolenceEvent {
  id: string;
  event_id?: string; // Alias for id (used by RTSP service)
  stream_id: string;
  stream_name?: string;
  stream_location?: string;
  start_time?: string;
  started_at: string; // Alias for start_time
  end_time?: string;
  ended_at?: string; // Alias for end_time
  duration_seconds?: number;
  max_confidence?: number;
  max_score: number; // Alias for max_confidence
  avg_confidence?: number;
  avg_score?: number; // Alias for avg_confidence
  min_confidence?: number;
  frame_count?: number;
  severity: EventSeverity;
  status: EventStatus;
  clip_path?: string;
  clip_duration?: number;
  thumbnail_path?: string;
  person_images?: string[];
  person_count?: number;
  face_paths?: string[]; // Detected participant faces from clip
  participants_count?: number;
  reviewed_at?: string;
  reviewed_by?: string;
  notes?: string;
  created_at?: string;
  updated_at?: string;
}

export interface EventUpdateRequest {
  status: "confirmed" | "dismissed";
  reviewed_by?: string;
  notes?: string;
}

export interface EventFilters {
  status?: EventStatus;
  severity?: EventSeverity;
  stream_id?: string;
  start_after?: string;
  start_before?: string;
  limit?: number;
  offset?: number;
}

export interface EventStats {
  period_days: number;
  total_events: number;
  by_status: Record<string, number>;
  by_severity: Record<string, number>;
  daily_breakdown: Array<{
    date: string;
    total: number;
    confirmed: number;
    avg_confidence: number;
  }>;
  top_streams: Array<{
    stream_name: string;
    event_count: number;
    max_confidence: number;
  }>;
}

// ============================================
// Real-time WebSocket Types
// ============================================

export type WebSocketMessageType =
  | "inference_score"
  | "event_start"
  | "event_started"
  | "event_end"
  | "event_ended"
  | "violence_alert"
  | "alert"
  | "ping"
  | "pong"
  | "stream_status"
  | "stream_started"
  | "stream_stopped";

export interface WebSocketMessage {
  type: WebSocketMessageType;
  data: any;
  timestamp?: string;
}

export interface InferenceScoreMessage {
  stream_id: string;
  stream_name: string;
  violence_score: number;
  non_violence_score: number;
  is_violent: boolean;
  timestamp: string; // Required!
  fps?: number;
}

export interface AlertMessage {
  type?: "event_start" | "event_end" | "alert" | "violence_alert";
  event_id: string;
  stream_id: string;
  stream_name: string;
  start_time?: string;
  timestamp: string;
  confidence?: number;
  max_score: number;
  max_confidence?: number;
  avg_confidence?: number;
  severity?: EventSeverity;
  message?: string;
  clip_path?: string;
  thumbnail_path?: string;
  clip_duration?: number;
  duration?: number;
}

export interface StreamStatusMessage {
  stream_id: string;
  status: Stream["status"];
  timestamp?: string;
}

// ============================================
// Dashboard Types
// ============================================

export interface DashboardStats {
  streams: {
    total: number;
    online: number;
    offline: number;
    error: number;
  };
  events: {
    pending: number;
    confirmed_today: number;
    total_today: number;
  };
  system: {
    ml_service_status: "online" | "offline";
    rtsp_service_status: "online" | "offline";
    backend_status: "online" | "offline";
  };
}
