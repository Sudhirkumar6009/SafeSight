import express, { Express } from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import rateLimit from "express-rate-limit";
import path from "path";
import fs from "fs";
import passport from "passport";

import config from "./config";
import database from "./config/database";
import routes from "./routes";
import { errorHandler, notFoundHandler } from "./middleware";
import logger from "./utils/logger";
import mlService from "./services/mlService";
import ModelConfig from "./models/modelConfig";
import "./config/passport"; // Initialize passport strategies

class Server {
  private app: Express;
  private port: number;

  constructor() {
    this.app = express();
    this.port = config.port;

    this.initializeDirectories();
    this.initializeMiddleware();
    this.initializeRoutes();
    this.initializeErrorHandling();
  }

  private initializeDirectories(): void {
    // Create uploads directory if it doesn't exist
    const uploadsDir = config.upload.uploadDir;
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
      logger.info(`Created uploads directory: ${uploadsDir}`);
    }

    // Create logs directory if it doesn't exist
    const logsDir = path.join(__dirname, "../logs");
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
      logger.info(`Created logs directory: ${logsDir}`);
    }
  }

  private initializeMiddleware(): void {
    // Security middleware
    this.app.use(
      helmet({
        crossOriginResourcePolicy: { policy: "cross-origin" },
      }),
    );

    // CORS configuration
    const corsOrigins = config.cors.origin.split(",").map((o) => o.trim());
    this.app.use(
      cors({
        origin: (origin, callback) => {
          // Allow requests with no origin (like mobile apps, curl, etc.)
          if (!origin) return callback(null, true);

          // In development, allow all origins
          if (config.nodeEnv === "development") {
            return callback(null, true);
          }

          // Check if the origin is in the allowed list
          if (
            corsOrigins.includes("*") ||
            corsOrigins.includes(origin) ||
            corsOrigins.some((o) => origin.endsWith(o.replace("*", "")))
          ) {
            return callback(null, true);
          }

          // Also allow Vercel preview deployments
          if (origin.includes(".vercel.app")) {
            return callback(null, true);
          }

          logger.warn(`CORS blocked origin: ${origin}`);
          callback(new Error("Not allowed by CORS"));
        },
        methods: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allowedHeaders: ["Content-Type", "Authorization", "Range"],
        exposedHeaders: [
          "Content-Range",
          "Accept-Ranges",
          "Content-Length",
          "Content-Type",
        ],
        credentials: true,
      }),
    );

    // Rate limiting
    const limiter = rateLimit({
      windowMs: config.rateLimit.windowMs,
      max: config.rateLimit.maxRequests,
      message: {
        success: false,
        error: "Too many requests, please try again later.",
      },
    });
    this.app.use(limiter);

    // Body parsing
    this.app.use(express.json({ limit: "10mb" }));
    this.app.use(express.urlencoded({ extended: true, limit: "10mb" }));

    // Initialize Passport
    this.app.use(passport.initialize());

    // Request logging
    if (config.nodeEnv === "development") {
      this.app.use(morgan("dev"));
    } else {
      this.app.use(morgan("combined"));
    }

    // Serve static files (for video streaming)
    this.app.use("/uploads", express.static(config.upload.uploadDir));
  }

  private initializeRoutes(): void {
    // API routes
    this.app.use(`${config.api.prefix}/${config.api.version}`, routes);

    // Root route
    this.app.get("/", (req, res) => {
      res.json({
        success: true,
        message: "SafeSight API Server",
        version: "1.0.0",
        documentation: `${config.api.prefix}/${config.api.version}/health`,
        endpoints: {
          videos: `${config.api.prefix}/${config.api.version}/videos`,
          model: `${config.api.prefix}/${config.api.version}/model`,
          inference: `${config.api.prefix}/${config.api.version}/inference`,
          predictions: `${config.api.prefix}/${config.api.version}/predictions`,
          health: `${config.api.prefix}/${config.api.version}/health`,
        },
      });
    });
  }

  private initializeErrorHandling(): void {
    this.app.use(notFoundHandler);
    this.app.use(errorHandler);
  }

  private async initializeModel(): Promise<void> {
    const modelPath = config.model.defaultPath;
    const architecture = config.model.architecture;

    if (!modelPath) {
      logger.warn("No default model path configured. Skipping auto-load.");
      return;
    }

    logger.info(`Auto-loading model from environment configuration...`);
    logger.info(`Model Path: ${modelPath}`);
    logger.info(`Architecture: ${architecture}`);

    try {
      // Check if ML service is healthy
      const isHealthy = await mlService.healthCheck();
      if (!isHealthy) {
        logger.warn(
          "ML Service is not available. Model will be loaded when service is ready.",
        );
        return;
      }

      // Load the model
      const result = await mlService.loadModel({
        modelPath,
        architecture,
      });

      if (result.success) {
        logger.info(`✓ Model loaded successfully: ${modelPath}`);

        // Create or update model config in database
        await ModelConfig.updateMany(
          { isActive: true },
          { isActive: false, isLoaded: false },
        );

        let modelConfig = await ModelConfig.findOne({ modelPath });
        if (modelConfig) {
          modelConfig.isActive = true;
          modelConfig.isLoaded = true;
          modelConfig.loadedAt = new Date();
          await modelConfig.save();
        } else {
          modelConfig = new ModelConfig({
            name: path.basename(modelPath, path.extname(modelPath)),
            description: "Auto-loaded violence detection model",
            modelPath,
            architecture,
            version: "1.0.0",
            inputSize: result.modelInfo?.inputSize || {
              frames: 16,
              height: 224,
              width: 224,
            },
            classes: result.modelInfo?.classes || ["violence", "non-violence"],
            isActive: true,
            isLoaded: true,
            loadedAt: new Date(),
          });
          await modelConfig.save();
        }
        logger.info(`✓ Model config saved to database`);
      } else {
        logger.warn(
          `Model auto-load failed: ${result.error || "Unknown error"}`,
        );
      }
    } catch (error: any) {
      logger.warn(`Model auto-load failed: ${error.message}`);
    }
  }

  public async start(): Promise<void> {
    try {
      // Connect to MongoDB
      await database.connect();

      // Auto-load model from environment configuration
      await this.initializeModel();

      // Start server
      this.app.listen(this.port, () => {
        logger.info(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🎬 SafeSight API Server                                 ║
║                                                           ║
║   Status:      Running                                    ║
║   Environment: ${config.nodeEnv.padEnd(20)}                ║
║   Port:        ${this.port.toString().padEnd(20)}                ║
║   API:         ${config.api.prefix}/${config.api.version}                        ║
║                                                           ║
║   MongoDB:     Connected                                  ║
║   ML Service:  ${config.mlService.url}                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        `);
      });

      // Graceful shutdown
      process.on("SIGTERM", this.shutdown.bind(this));
      process.on("SIGINT", this.shutdown.bind(this));
    } catch (error) {
      logger.error("Failed to start server:", error);
      process.exit(1);
    }
  }

  private async shutdown(): Promise<void> {
    logger.info("Shutting down server...");
    await database.disconnect();
    process.exit(0);
  }
}

// Start server
const server = new Server();
server.start();

export default server;
