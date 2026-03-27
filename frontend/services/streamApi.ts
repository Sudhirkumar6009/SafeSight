/**
 * SafeSight - Stream & Event API Services
 * ============================================
 * API services for RTSP streams and violence events.
 */

import axios, { AxiosInstance } from "axios";
import {
  ApiResponse,
  Stream,
  StreamCreateRequest,
  StreamStats,
  ViolenceEvent,
  EventUpdateRequest,
  EventFilters,
  EventStats,
  VideoClip,
  VideoClipsResponse,
  ExtractedFace,
  ExtractedFacesResponse,
} from "@/types";

const RTSP_SERVICE_URL =
  process.env.NEXT_PUBLIC_RTSP_SERVICE_URL || "http://localhost:8080";
const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api/v1";

// ============================================
// RTSP Stream Service
// ============================================

class StreamService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: `${RTSP_SERVICE_URL}/api/v1`,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error(
          "RTSP Service Error:",
          error.response?.data || error.message,
        );
        return Promise.reject(error);
      },
    );
  }

  // Health check
  async getHealth(): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get("/health");
      return { success: true, data: response.data };
    } catch (error) {
      return { success: false, error: "RTSP service unavailable" };
    }
  }

  // List all streams
  async getStreams(): Promise<ApiResponse<Stream[]>> {
    try {
      const response = await this.client.get("/streams");
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message, data: [] };
    }
  }

  // Get single stream
  async getStream(id: string): Promise<ApiResponse<Stream>> {
    const response = await this.client.get(`/streams/${id}`);
    return response.data;
  }

  // Create new stream
  async createStream(data: StreamCreateRequest): Promise<ApiResponse<Stream>> {
    // Transform frontend field names to backend expected format
    const payload = {
      name: data.name,
      url: data.rtsp_url || data.url, // Backend expects 'url' not 'rtsp_url'
      stream_type: data.stream_type || "rtsp",
      location: data.location,
      auto_start: data.auto_start ?? false,
      custom_threshold: data.custom_threshold,
    };
    const response = await this.client.post("/streams", payload);
    return response.data;
  }

  // Update stream
  async updateStream(
    id: string,
    data: Partial<StreamCreateRequest>,
  ): Promise<ApiResponse<Stream>> {
    // Transform frontend field names to backend expected format
    const payload: Record<string, any> = {};
    if (data.name) payload.name = data.name;
    if (data.rtsp_url || data.url) payload.url = data.rtsp_url || data.url;
    if (data.stream_type) payload.stream_type = data.stream_type;
    if (data.location !== undefined) payload.location = data.location;
    if (data.custom_threshold !== undefined)
      payload.custom_threshold = data.custom_threshold;

    try {
      const response = await this.client.patch(`/streams/${id}`, payload);
      return response.data;
    } catch (error: any) {
      // Some gateways/proxies block PATCH; retry with PUT for compatibility.
      const status = error?.response?.status;
      if (status === 403 || status === 405 || status === 501) {
        const response = await this.client.put(`/streams/${id}`, payload);
        return response.data;
      }
      throw error;
    }
  }

  // Delete stream
  async deleteStream(id: string): Promise<ApiResponse<void>> {
    const response = await this.client.delete(`/streams/${id}`);
    return response.data;
  }

  // Start stream
  async startStream(id: string): Promise<ApiResponse<any>> {
    const response = await this.client.post(`/streams/${id}/start`);
    return response.data;
  }

  // Stop stream
  async stopStream(id: string): Promise<ApiResponse<any>> {
    const response = await this.client.post(`/streams/${id}/stop`);
    return response.data;
  }

  // Get stream status
  async getStreamStatus(id: string): Promise<ApiResponse<StreamStats>> {
    const response = await this.client.get(`/streams/${id}/status`);
    return response.data;
  }

  // Get all stream statuses
  async getAllStreamStatuses(): Promise<ApiResponse<StreamStats[]>> {
    try {
      const response = await this.client.get("/streams/status");
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message, data: [] };
    }
  }

  // Get current inference scores
  async getCurrentScores(): Promise<ApiResponse<any[]>> {
    try {
      const response = await this.client.get("/inference/scores");
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message, data: [] };
    }
  }

  // Get violence events from RTSP service (PostgreSQL)
  async getEvents(filters?: {
    status?: string;
    stream_id?: number;
    limit?: number;
    offset?: number;
  }): Promise<{
    success: boolean;
    data: ViolenceEvent[];
    pagination?: { limit: number; offset: number; count: number };
  }> {
    try {
      const params = new URLSearchParams();
      if (filters?.status) params.append("status", filters.status);
      if (filters?.stream_id)
        params.append("stream_id", filters.stream_id.toString());
      if (filters?.limit) params.append("limit", filters.limit.toString());
      if (filters?.offset) params.append("offset", filters.offset.toString());

      const response = await this.client.get(`/events?${params.toString()}`);

      // Normalize events: ensure 'id' is populated from 'event_id' if missing
      if (response.data?.data) {
        response.data.data = response.data.data.map((event: any) => ({
          ...event,
          id: event.id || event.event_id,
          event_id: event.event_id || event.id,
        }));
      }

      return response.data;
    } catch (error: any) {
      return { success: false, data: [] };
    }
  }

  // Mark event as action executed (Yes - took action)
  async markActionExecuted(
    eventId: string | number,
  ): Promise<ApiResponse<void>> {
    try {
      const response = await this.client.post(
        `/events/${eventId}/action-executed`,
      );
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // Mark event as no action required (No - no action needed)
  async markNoActionRequired(
    eventId: string | number,
  ): Promise<ApiResponse<void>> {
    try {
      const response = await this.client.post(
        `/events/${eventId}/no-action-required`,
      );
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // Get clip URL (served from RTSP service)
  getClipUrl(clipFilename: string): string {
    if (!clipFilename) return "";
    return `${RTSP_SERVICE_URL}/api/v1/clips/${clipFilename}`;
  }

  // Get thumbnail URL (served from RTSP service)
  getThumbnailUrl(thumbnailFilename: string): string {
    if (!thumbnailFilename) return "";
    // Handle both "filename.jpg" and legacy "thumbnails/filename.jpg" formats
    const cleanFilename = thumbnailFilename.replace(/^thumbnails\//, "");
    return `${RTSP_SERVICE_URL}/api/v1/clips/thumbnails/${cleanFilename}`;
  }

  // Get person image URL (served from RTSP service)
  getPersonImageUrl(filename: string): string {
    return `${RTSP_SERVICE_URL}/api/v1/person-images/${filename}`;
  }

  // Get face/participant image URL (served from RTSP service)
  // Handles multiple formats:
  // - "face_participants/{event_id}/{filename.jpg}" - legacy filesystem
  // - "db:{face_id}" - database stored faces
  // - "{hex_filename}" - encrypted storage faces (need event_id)
  getFaceUrl(facePath: string, eventId?: string): string {
    if (!facePath) return "";
    
    // Handle database stored faces (format: "db:123")
    if (facePath.startsWith("db:")) {
      const faceId = facePath.slice(3);
      // Use the extracted-faces image endpoint
      return `${RTSP_SERVICE_URL}/api/v1/extracted-faces/${faceId}/image`;
    }
    
    // Handle encrypted storage faces (hex filenames like "5b2d87a29294d0afd494ff41939c700b")
    // These need the event_id to build the correct URL
    if (eventId && /^[a-f0-9]{32}$/i.test(facePath)) {
      return `${RTSP_SERVICE_URL}/api/v1/faces/${eventId}/secure/${facePath}`;
    }
    
    // Handle legacy face_participants format
    // facePath format: "face_participants/{event_id}/{filename.jpg}"
    // API expects: /api/v1/faces/{event_id}/{filename}
    const parts = facePath.replace(/^face_participants\//, "").split("/");
    if (parts.length >= 2) {
      return `${RTSP_SERVICE_URL}/api/v1/faces/${parts[0]}/${parts[1]}`;
    }
    return `${RTSP_SERVICE_URL}/api/v1/faces/${facePath}`;
  }

  // Extract faces from event clip (manual trigger)
  async extractFaces(
    eventId: string,
  ): Promise<
    ApiResponse<{ event_id: string; faces_count: number; stored_in_db: boolean; storage: string }>
  > {
    try {
      const response = await this.client.post(`/faces/${eventId}/extract`);
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  // Get faces for an event
  async getFaces(
    eventId: string,
  ): Promise<
    ApiResponse<{ event_id: string; faces: any[]; count: number; source: string }>
  > {
    try {
      const response = await this.client.get(`/faces/${eventId}`);
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }
}

// ============================================
// Event Service (Backend)
// ============================================

class EventService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: BACKEND_URL,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error(
          "Event Service Error:",
          error.response?.data || error.message,
        );
        return Promise.reject(error);
      },
    );
  }

  // Get events with filters
  async getEvents(filters?: EventFilters): Promise<{
    success: boolean;
    data: ViolenceEvent[];
    pagination?: {
      total: number;
      limit: number;
      offset: number;
      hasMore: boolean;
    };
  }> {
    try {
      const params = new URLSearchParams();
      if (filters?.status) params.append("status", filters.status);
      if (filters?.severity) params.append("severity", filters.severity);
      if (filters?.stream_id) params.append("stream_id", filters.stream_id);
      if (filters?.start_after)
        params.append("start_after", filters.start_after);
      if (filters?.start_before)
        params.append("start_before", filters.start_before);
      if (filters?.limit) params.append("limit", filters.limit.toString());
      if (filters?.offset) params.append("offset", filters.offset.toString());

      const response = await this.client.get(`/events?${params.toString()}`);
      return response.data;
    } catch (error: any) {
      return { success: false, data: [] };
    }
  }

  // Get pending events
  async getPendingEvents(
    limit: number = 20,
  ): Promise<ApiResponse<ViolenceEvent[]>> {
    try {
      const response = await this.client.get(`/events/pending?limit=${limit}`);
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message, data: [] };
    }
  }

  // Get single event
  async getEvent(id: string): Promise<ApiResponse<ViolenceEvent>> {
    const response = await this.client.get(`/events/${id}`);
    return response.data;
  }

  // Update event status
  async updateEventStatus(
    id: string,
    data: EventUpdateRequest,
  ): Promise<ApiResponse<ViolenceEvent>> {
    const response = await this.client.patch(`/events/${id}/status`, data);
    return response.data;
  }

  // Get event statistics
  async getEventStats(days: number = 7): Promise<ApiResponse<EventStats>> {
    try {
      const response = await this.client.get(`/events/stats?days=${days}`);
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
        data: {
          period_days: days,
          total_events: 0,
          by_status: {},
          by_severity: {},
          daily_breakdown: [],
          top_streams: [],
        },
      };
    }
  }

  // Get streams from backend (PostgreSQL)
  async getStreams(): Promise<ApiResponse<Stream[]>> {
    try {
      const response = await this.client.get("/streams");
      return response.data;
    } catch (error: any) {
      return { success: false, error: error.message, data: [] };
    }
  }

  // Get stream events
  async getStreamEvents(
    streamId: string,
    limit: number = 50,
    status?: string,
  ): Promise<ApiResponse<ViolenceEvent[]>> {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (status) params.append("status", status);

    const response = await this.client.get(
      `/streams/${streamId}/events?${params.toString()}`,
    );
    return response.data;
  }

  // Get clip URL (served from RTSP service)
  getClipUrl(clipFilename: string): string {
    if (!clipFilename) return "";
    return `${RTSP_SERVICE_URL}/api/v1/clips/${clipFilename}`;
  }

  // Get thumbnail URL (served from RTSP service)
  getThumbnailUrl(thumbnailFilename: string): string {
    if (!thumbnailFilename) return "";
    // Handle both "filename.jpg" and legacy "thumbnails/filename.jpg" formats
    const cleanFilename = thumbnailFilename.replace(/^thumbnails\//, "");
    return `${RTSP_SERVICE_URL}/api/v1/clips/thumbnails/${cleanFilename}`;
  }
}

// Export service instances
export const streamService = new StreamService();
export const eventService = new EventService();
