import Video, { IVideo } from "../models/video";
import Prediction from "../models/prediction";
import ModelConfig from "../models/modelConfig";
import mlService, { InferenceResponse } from "./mlService";
import logger from "../utils/logger";
import mongoose from "mongoose";
import path from "path";
import fs from "fs";
import os from "os";
import { getGridFSBucket } from "../config/gridfs";

export interface AnalysisResult {
  success: boolean;
  prediction?: {
    id: string;
    classification: string;
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
    };
  };
  error?: string;
}

class VideoAnalysisService {
  // Extract video from GridFS to temporary file for ML inference
  private async extractVideoToTemp(video: IVideo): Promise<string> {
    const bucket = getGridFSBucket();
    const tempDir = os.tmpdir();
    const tempFilePath = path.join(tempDir, `safesight_${video.filename}`);

    logger.info(`Extracting video from GridFS to: ${tempFilePath}`);

    return new Promise((resolve, reject) => {
      const downloadStream = bucket.openDownloadStream(video.gridfsId);
      const writeStream = fs.createWriteStream(tempFilePath);

      downloadStream
        .pipe(writeStream)
        .on("error", (error) => {
          logger.error("Error extracting video from GridFS:", error);
          reject(error);
        })
        .on("finish", () => {
          logger.info(`Video extracted successfully: ${tempFilePath}`);
          resolve(tempFilePath);
        });
    });
  }

  // Clean up temporary file after inference
  private cleanupTempFile(filePath: string): void {
    try {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        logger.info(`Cleaned up temp file: ${filePath}`);
      }
    } catch (error) {
      logger.warn(`Failed to cleanup temp file: ${filePath}`, error);
    }
  }

  async analyzeVideo(videoId: string): Promise<AnalysisResult> {
    let tempFilePath: string | null = null;

    try {
      // Find the video
      const video = await Video.findById(videoId);
      if (!video) {
        return {
          success: false,
          error: "Video not found",
        };
      }

      // Get active model config
      const activeModel = await ModelConfig.findOne({ isActive: true });
      if (!activeModel) {
        return {
          success: false,
          error: "No active model configured. Please load a model first.",
        };
      }

      // Update video status to processing
      video.status = "processing";
      await video.save();

      // Create prediction record
      const prediction = new Prediction({
        videoId: video._id,
        modelId: activeModel._id,
        status: "running",
      });
      await prediction.save();

      // Extract video from GridFS to temporary file
      logger.info(`Starting inference for video: ${video.filename}`);
      tempFilePath = await this.extractVideoToTemp(video);

      logger.info(`Video path for inference: ${tempFilePath}`);

      const inferenceResult: InferenceResponse = await mlService.runInference({
        videoPath: tempFilePath,
        modelPath: activeModel.modelPath,
        architecture: activeModel.architecture,
        numFrames: activeModel.inputSize.frames,
      });

      // Clean up temp file after inference
      if (tempFilePath) {
        this.cleanupTempFile(tempFilePath);
        tempFilePath = null;
      }

      if (!inferenceResult.success) {
        prediction.status = "failed";
        prediction.error = inferenceResult.error || "Inference failed";
        await prediction.save();

        video.status = "failed";
        await video.save();

        return {
          success: false,
          error: inferenceResult.error || "Inference failed",
        };
      }

      // Update prediction with results
      prediction.classification = inferenceResult.classification;
      prediction.confidence = inferenceResult.confidence;
      prediction.probabilities = inferenceResult.probabilities;
      prediction.metrics = {
        inferenceTime: inferenceResult.metrics.inferenceTime,
      };

      if (inferenceResult.frameAnalysis) {
        prediction.frameAnalysis = inferenceResult.frameAnalysis;
      }

      prediction.status = "completed";
      prediction.completedAt = new Date();
      await prediction.save();

      // Update video status
      video.status = "completed";
      video.processedAt = new Date();
      await video.save();

      // Update model performance stats
      await this.updateModelPerformance(activeModel._id, inferenceResult);

      logger.info(`Analysis completed for video: ${video.filename}`);

      return {
        success: true,
        prediction: {
          id: prediction._id.toString(),
          classification: prediction.classification,
          confidence: prediction.confidence,
          probabilities: prediction.probabilities,
          metrics: prediction.metrics,
          frameAnalysis: prediction.frameAnalysis
            ? {
              totalFrames: prediction.frameAnalysis.totalFrames,
              violentFrames: prediction.frameAnalysis.violentFrames,
              nonViolentFrames: prediction.frameAnalysis.nonViolentFrames,
            }
            : undefined,
        },
      };
    } catch (error: any) {
      // Clean up temp file on error
      if (tempFilePath) {
        this.cleanupTempFile(tempFilePath);
      }
      logger.error("Video analysis failed:", error);
      return {
        success: false,
        error: error.message,
      };
    }
  }

  private async updateModelPerformance(
    modelId: mongoose.Types.ObjectId,
    result: InferenceResponse,
  ): Promise<void> {
    try {
      const model = await ModelConfig.findById(modelId);
      if (!model) return;

      const totalPredictions = (model.performance?.totalPredictions || 0) + 1;
      const currentAvgTime = model.performance?.avgInferenceTime || 0;
      const newAvgTime =
        (currentAvgTime * (totalPredictions - 1) +
          result.metrics.inferenceTime) /
        totalPredictions;

      await ModelConfig.findByIdAndUpdate(modelId, {
        $set: {
          "performance.totalPredictions": totalPredictions,
          "performance.avgInferenceTime": newAvgTime,
        },
      });
    } catch (error) {
      logger.error("Failed to update model performance:", error);
    }
  }

  async getVideoWithPrediction(videoId: string) {
    try {
      const video = await Video.findById(videoId);
      if (!video) {
        return null;
      }

      const prediction = await Prediction.findOne({ videoId: video._id })
        .sort({ createdAt: -1 })
        .populate("modelId", "name architecture");

      return {
        video,
        prediction,
      };
    } catch (error) {
      logger.error("Failed to get video with prediction:", error);
      return null;
    }
  }
}

export default new VideoAnalysisService();
