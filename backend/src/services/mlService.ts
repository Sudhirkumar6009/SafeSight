import axios, { AxiosInstance } from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import config from "../config";
import logger from "../utils/logger";

export interface InferenceRequest {
  videoPath: string;
  modelPath: string;
  architecture?: string;
  numFrames?: number;
}

export interface InferenceResponse {
  success: boolean;
  classification: "violence" | "non-violence";
  confidence: number;
  probabilities: {
    violence: number;
    nonViolence: number;
  };
  metrics: {
    inferenceTime: number;
    framesProcessed: number;
  };
  frameAnalysis?: {
    totalFrames: number;
    violentFrames: number;
    nonViolentFrames: number;
    frameScores: number[];
  };
  error?: string;
}

type RawInferenceResponse = Omit<
  InferenceResponse,
  "classification" | "probabilities"
> & {
  classification?: string;
  probabilities?: {
    violence?: number;
    nonViolence?: number;
    non_violence?: number;
    normal?: number;
  };
};

export interface ModelLoadRequest {
  modelPath: string;
  architecture: string;
}

export interface ModelLoadResponse {
  success: boolean;
  message: string;
  modelInfo?: {
    name: string;
    architecture: string;
    inputSize: {
      frames: number;
      height: number;
      width: number;
    };
    classes: string[];
  };
  error?: string;
}

export interface ModelStatusResponse {
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
}

export interface ModelMetricsResponse {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix?: {
    truePositive: number;
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
  };
  totalPredictions: number;
  avgInferenceTime: number;
}

class MLServiceClient {
  private client: AxiosInstance;
  private baseUrl: string;

  constructor() {
    this.baseUrl = config.mlService.url;
    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: config.mlService.timeout,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.client.interceptors.request.use(
      (config) => {
        logger.debug(
          `ML Service Request: ${config.method?.toUpperCase()} ${config.url}`,
        );
        return config;
      },
      (error) => {
        logger.error("ML Service Request Error:", error);
        return Promise.reject(error);
      },
    );

    this.client.interceptors.response.use(
      (response) => {
        logger.debug(`ML Service Response: ${response.status}`);
        return response;
      },
      (error) => {
        logger.error("ML Service Response Error:", error.message);
        return Promise.reject(error);
      },
    );
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get("/health");
      return response.data.status === "healthy";
    } catch (error) {
      logger.error("ML Service health check failed:", error);
      return false;
    }
  }

  async loadModel(request: ModelLoadRequest): Promise<ModelLoadResponse> {
    try {
      const response = await this.client.post<ModelLoadResponse>(
        "/model/load",
        request,
      );
      return response.data;
    } catch (error: any) {
      logger.error("Failed to load model:", error.message);
      return {
        success: false,
        message: "Failed to load model",
        error: error.message,
      };
    }
  }

  async getModelStatus(): Promise<ModelStatusResponse> {
    try {
      const response =
        await this.client.get<ModelStatusResponse>("/model/status");
      return response.data;
    } catch (error: any) {
      logger.error("Failed to get model status:", error.message);
      return {
        isLoaded: false,
        gpuAvailable: false,
      };
    }
  }

  async getModelMetrics(): Promise<ModelMetricsResponse | null> {
    try {
      const response =
        await this.client.get<ModelMetricsResponse>("/model/metrics");
      return response.data;
    } catch (error: any) {
      logger.error("Failed to get model metrics:", error.message);
      return null;
    }
  }

  async runInference(request: InferenceRequest): Promise<InferenceResponse> {
    try {
      // Check if ML service is remote (Hugging Face) - use file upload
      const isRemoteService =
        this.baseUrl.includes("hf.space") ||
        this.baseUrl.includes("huggingface");

      if (isRemoteService) {
        // Use file upload for remote services
        return await this.runInferenceWithUpload(request);
      }

      // Verify file exists before sending to ML service
      if (!fs.existsSync(request.videoPath)) {
        logger.error(`Video file does not exist: ${request.videoPath}`);
        return {
          success: false,
          classification: "non-violence",
          confidence: 0,
          probabilities: { violence: 0, nonViolence: 0 },
          metrics: { inferenceTime: 0, framesProcessed: 0 },
          error: `Video file not found: ${request.videoPath}`,
        };
      }

      logger.info(
        `Sending inference request to ML service: ${request.videoPath}`,
      );

      // Use path-based inference for local services
      const response = await this.client.post<RawInferenceResponse>(
        "/inference/predict",
        request,
      );
      return this.normalizeInferenceResponse(response.data);
    } catch (error: any) {
      // Capture detailed error from ML service response
      const mlServiceError =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        error.message;
      logger.error(`Inference failed: ${mlServiceError}`);
      logger.error(
        `Full error details: ${JSON.stringify(error.response?.data)}`,
      );
      return {
        success: false,
        classification: "non-violence",
        confidence: 0,
        probabilities: {
          violence: 0,
          nonViolence: 0,
        },
        metrics: {
          inferenceTime: 0,
          framesProcessed: 0,
        },
        error: mlServiceError,
      };
    }
  }

  async runInferenceWithUpload(
    request: InferenceRequest,
  ): Promise<InferenceResponse> {
    try {
      // Verify file exists
      if (!fs.existsSync(request.videoPath)) {
        throw new Error(`Video file not found: ${request.videoPath}`);
      }

      // Create form data with file upload
      const formData = new FormData();
      const fileStream = fs.createReadStream(request.videoPath);
      const filename = path.basename(request.videoPath);

      formData.append("video", fileStream, {
        filename: filename,
        contentType: "video/mp4",
      });

      if (request.numFrames) {
        formData.append("numFrames", request.numFrames.toString());
      }

      logger.info(`Uploading video to remote ML service: ${filename}`);

      const response = await axios.post<RawInferenceResponse>(
        `${this.baseUrl}/inference/predict-upload`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
          timeout: config.mlService.timeout,
          maxBodyLength: Infinity,
          maxContentLength: Infinity,
        },
      );

      return this.normalizeInferenceResponse(response.data);
    } catch (error: any) {
      logger.error("Inference with upload failed:", error.message);
      return {
        success: false,
        classification: "non-violence",
        confidence: 0,
        probabilities: {
          violence: 0,
          nonViolence: 0,
        },
        metrics: {
          inferenceTime: 0,
          framesProcessed: 0,
        },
        error: error.response?.data?.detail || error.message,
      };
    }
  }

  async unloadModel(): Promise<boolean> {
    try {
      const response = await this.client.post("/model/unload");
      return response.data.success;
    } catch (error: any) {
      logger.error("Failed to unload model:", error.message);
      return false;
    }
  }

  private normalizeClassification(label?: string): "violence" | "non-violence" {
    const normalized = (label || "").trim().toLowerCase();

    if (normalized === "violence" || normalized === "violent") {
      return "violence";
    }

    if (
      normalized === "non-violence" ||
      normalized === "non_violence" ||
      normalized === "non violence" ||
      normalized === "normal" ||
      normalized === "safe"
    ) {
      return "non-violence";
    }

    logger.warn(
      `Unknown ML classification label "${label}". Falling back to non-violence.`,
    );
    return "non-violence";
  }

  private normalizeInferenceResponse(
    response: RawInferenceResponse,
  ): InferenceResponse {
    const classification = this.normalizeClassification(
      response.classification,
    );
    const probabilities = response.probabilities || {};
    const nonViolenceProbability =
      probabilities.nonViolence ??
      probabilities.non_violence ??
      probabilities.normal ??
      0;

    return {
      success: response.success,
      classification,
      confidence: response.confidence,
      probabilities: {
        violence: probabilities.violence ?? 0,
        nonViolence: nonViolenceProbability,
      },
      metrics: response.metrics,
      frameAnalysis: response.frameAnalysis,
      error: response.error,
    };
  }
}

export default new MLServiceClient();
