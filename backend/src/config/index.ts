import dotenv from "dotenv";
import path from "path";

dotenv.config();

interface IConfig {
  nodeEnv: string;
  port: number;
  mongodb: {
    uri: string;
    dbName: string;
  };
  postgres: {
    uri: string;
  };
  rtspService: {
    url: string;
  };
  mlService: {
    url: string;
    timeout: number;
  };
  model: {
    defaultPath: string;
    architecture: string;
  };
  upload: {
    maxFileSize: number;
    uploadDir: string;
    allowedExtensions: string[];
  };
  api: {
    version: string;
    prefix: string;
  };
  rateLimit: {
    windowMs: number;
    maxRequests: number;
  };
  cors: {
    origin: string;
  };
  logging: {
    level: string;
    file: string;
  };
  auth: {
    jwtSecret: string;
    jwtExpiresIn: string;
    google: {
      clientId: string;
      clientSecret: string;
      callbackUrl: string;
    };
    mongodbAtlasUri: string;
  };
}

const config: IConfig = {
  nodeEnv: process.env.NODE_ENV || "development",
  port: parseInt(process.env.PORT || "5000", 10),

  mongodb: {
    uri: process.env.MONGODB_URI || "mongodb://localhost:27017/SafeSight",
    dbName: process.env.MONGODB_DB_NAME || "SafeSight",
  },

  postgres: {
    uri:
      process.env.DATABASE_URL ||
      process.env.POSTGRES_URI ||
      "postgresql://postgres:password@localhost:5432/violencesense",
  },

  rtspService: {
    url: process.env.RTSP_SERVICE_URL || "http://localhost:8080",
  },

  mlService: {
    url: process.env.ML_SERVICE_URL || "http://localhost:8000",
    timeout: parseInt(process.env.ML_SERVICE_TIMEOUT || "300000", 10),
  },

  model: {
    defaultPath:
      process.env.DEFAULT_MODEL_PATH || "./models/best_violence_model.keras",
    architecture: process.env.MODEL_ARCHITECTURE || "keras-cnn",
  },

  upload: {
    maxFileSize: parseInt(process.env.MAX_FILE_SIZE || "500000000", 10),
    uploadDir: process.env.UPLOAD_DIR || path.join(__dirname, "../../uploads"),
    allowedExtensions: (
      process.env.ALLOWED_EXTENSIONS || "mp4,avi,mov,mkv"
    ).split(","),
  },

  api: {
    version: process.env.API_VERSION || "v1",
    prefix: process.env.API_PREFIX || "/api",
  },

  rateLimit: {
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || "900000", 10),
    maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || "100", 10),
  },

  cors: {
    origin: process.env.CORS_ORIGIN || "http://localhost:3000",
  },

  logging: {
    level: process.env.LOG_LEVEL || "debug",
    file: process.env.LOG_FILE || "./logs/app.log",
  },

  auth: {
    jwtSecret: process.env.JWT_SECRET || "safesight_jwt_secret_key_2024",
    jwtExpiresIn: process.env.JWT_EXPIRES_IN || "7d",
    google: {
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
      callbackUrl:
        process.env.GOOGLE_CALLBACK_URL ||
        "http://localhost:5000/api/auth/google/callback",
    },
    mongodbAtlasUri: process.env.MONGODB_ATLAS_URI || "",
  },
};

export { config };
export default config;
