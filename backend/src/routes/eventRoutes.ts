/**
 * SafeSight Backend - Event Routes
 * =====================================
 * REST API endpoints for violence event management.
 *
 * Endpoints:
 * - GET /api/events - List events with filters
 * - GET /api/events/:id - Get event by ID
 * - PATCH /api/events/:id/status - Update event status (confirm/dismiss)
 * - GET /api/events/stats - Get event statistics
 * - GET /api/streams - List all streams
 * - GET /api/streams/:id/events - Get events for a stream
 */

import { Router, Request, Response, NextFunction } from "express";
import { Pool } from "pg";
import logger from "../utils/logger";
import config from "../config";

const router = Router();

// PostgreSQL connection pool
let pool: Pool | null = null;

function getPool(): Pool {
  if (!pool) {
    const connectionString =
      config.postgres?.uri ||
      process.env.DATABASE_URL ||
      "postgresql://postgres:password@localhost:5432/violencesense";

    pool = new Pool({
      connectionString,
      max: 10,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    pool.on("error", (err) => {
      logger.error("PostgreSQL pool error:", err);
    });
  }
  return pool;
}

// Types
interface EventFilters {
  status?: string;
  severity?: string;
  stream_id?: string;
  start_after?: string;
  start_before?: string;
  limit?: number;
  offset?: number;
}

interface EventUpdate {
  status: "confirmed" | "dismissed";
  reviewed_by?: string;
  notes?: string;
}

// ============================================
// EVENT ENDPOINTS
// ============================================

/**
 * GET /api/events
 * List events with optional filters and pagination
 */
router.get(
  "/events",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const filters: EventFilters = {
        status: req.query.status as string,
        severity: req.query.severity as string,
        stream_id: req.query.stream_id as string,
        start_after: req.query.start_after as string,
        start_before: req.query.start_before as string,
        limit: parseInt(req.query.limit as string) || 50,
        offset: parseInt(req.query.offset as string) || 0,
      };

      // Build query
      let query = `
      SELECT 
        e.id, e.stream_id, e.stream_name, e.start_time, e.end_time,
        e.duration_seconds, e.max_confidence, e.avg_confidence, e.min_confidence,
        e.frame_count, e.severity, e.status, e.clip_path, e.clip_duration,
        e.thumbnail_path, e.reviewed_at, e.reviewed_by, e.notes,
        e.created_at, e.updated_at,
        s.location as stream_location
      FROM events e
      LEFT JOIN streams s ON e.stream_id = s.id
      WHERE 1=1
    `;
      const params: any[] = [];
      let paramIndex = 1;

      if (filters.status) {
        query += ` AND e.status = $${paramIndex++}`;
        params.push(filters.status);
      }

      if (filters.severity) {
        query += ` AND e.severity = $${paramIndex++}`;
        params.push(filters.severity);
      }

      if (filters.stream_id) {
        query += ` AND e.stream_id = $${paramIndex++}`;
        params.push(filters.stream_id);
      }

      if (filters.start_after) {
        query += ` AND e.start_time >= $${paramIndex++}`;
        params.push(filters.start_after);
      }

      if (filters.start_before) {
        query += ` AND e.start_time <= $${paramIndex++}`;
        params.push(filters.start_before);
      }

      query += ` ORDER BY e.start_time DESC`;
      query += ` LIMIT $${paramIndex++} OFFSET $${paramIndex++}`;
      params.push(filters.limit, filters.offset);

      const result = await pool.query(query, params);

      // Get total count
      let countQuery = `SELECT COUNT(*) FROM events e WHERE 1=1`;
      const countParams: any[] = [];
      let countParamIndex = 1;

      if (filters.status) {
        countQuery += ` AND e.status = $${countParamIndex++}`;
        countParams.push(filters.status);
      }
      if (filters.severity) {
        countQuery += ` AND e.severity = $${countParamIndex++}`;
        countParams.push(filters.severity);
      }
      if (filters.stream_id) {
        countQuery += ` AND e.stream_id = $${countParamIndex++}`;
        countParams.push(filters.stream_id);
      }

      const countResult = await pool.query(countQuery, countParams);
      const total = parseInt(countResult.rows[0].count);

      res.json({
        success: true,
        data: result.rows,
        pagination: {
          total,
          limit: filters.limit,
          offset: filters.offset,
          hasMore: filters.offset! + result.rows.length < total,
        },
      });
    } catch (error) {
      logger.error("Error listing events:", error);
      next(error);
    }
  },
);

/**
 * GET /api/events/pending
 * Get pending (new) events for dashboard
 */
router.get(
  "/events/pending",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const limit = parseInt(req.query.limit as string) || 20;

      const result = await pool.query(
        `
      SELECT 
        e.id, e.stream_id, e.stream_name, e.start_time, e.end_time,
        e.duration_seconds, e.max_confidence, e.avg_confidence,
        e.severity, e.status, e.clip_path, e.thumbnail_path,
        e.created_at, s.location as stream_location
      FROM events e
      LEFT JOIN streams s ON e.stream_id = s.id
      WHERE e.status = 'new'
      ORDER BY e.start_time DESC
      LIMIT $1
    `,
        [limit],
      );

      res.json({
        success: true,
        data: result.rows,
        count: result.rows.length,
      });
    } catch (error) {
      logger.error("Error getting pending events:", error);
      next(error);
    }
  },
);

/**
 * GET /api/events/stats
 * Get event statistics
 */
router.get(
  "/events/stats",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const days = parseInt(req.query.days as string) || 7;

      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - days);

      // Status counts
      const statusResult = await pool.query(
        `
      SELECT status, COUNT(*) as count
      FROM events
      WHERE created_at >= $1
      GROUP BY status
    `,
        [cutoff.toISOString()],
      );

      // Severity counts
      const severityResult = await pool.query(
        `
      SELECT severity, COUNT(*) as count
      FROM events
      WHERE created_at >= $1
      GROUP BY severity
    `,
        [cutoff.toISOString()],
      );

      // Daily counts
      const dailyResult = await pool.query(
        `
      SELECT 
        DATE(start_time) as date,
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE status = 'confirmed') as confirmed,
        AVG(max_confidence) as avg_confidence
      FROM events
      WHERE start_time >= $1
      GROUP BY DATE(start_time)
      ORDER BY date DESC
    `,
        [cutoff.toISOString()],
      );

      // Stream counts
      const streamResult = await pool.query(
        `
      SELECT 
        e.stream_name,
        COUNT(*) as event_count,
        MAX(e.max_confidence) as max_confidence
      FROM events e
      WHERE e.created_at >= $1
      GROUP BY e.stream_name
      ORDER BY event_count DESC
      LIMIT 10
    `,
        [cutoff.toISOString()],
      );

      const byStatus: Record<string, number> = {};
      statusResult.rows.forEach((row) => {
        byStatus[row.status] = parseInt(row.count);
      });

      const bySeverity: Record<string, number> = {};
      severityResult.rows.forEach((row) => {
        bySeverity[row.severity] = parseInt(row.count);
      });

      res.json({
        success: true,
        data: {
          period_days: days,
          total_events: Object.values(byStatus).reduce((a, b) => a + b, 0),
          by_status: byStatus,
          by_severity: bySeverity,
          daily_breakdown: dailyResult.rows,
          top_streams: streamResult.rows,
        },
      });
    } catch (error) {
      logger.error("Error getting event stats:", error);
      next(error);
    }
  },
);

/**
 * GET /api/events/:id
 * Get event by ID
 */
router.get(
  "/events/:id",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const { id } = req.params;

      const result = await pool.query(
        `
      SELECT 
        e.*, s.location as stream_location, s.url as stream_url
      FROM events e
      LEFT JOIN streams s ON e.stream_id = s.id
      WHERE e.id = $1
    `,
        [id],
      );

      if (result.rows.length === 0) {
        return res.status(404).json({
          success: false,
          error: "Event not found",
        });
      }

      res.json({
        success: true,
        data: result.rows[0],
      });
    } catch (error) {
      logger.error("Error getting event:", error);
      next(error);
    }
  },
);

/**
 * PATCH /api/events/:id/status
 * Update event status (confirm/dismiss)
 */
router.patch(
  "/events/:id/status",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const { id } = req.params;
      const update: EventUpdate = req.body;

      if (!["confirmed", "dismissed"].includes(update.status)) {
        return res.status(400).json({
          success: false,
          error: "Status must be 'confirmed' or 'dismissed'",
        });
      }

      const result = await pool.query(
        `
      UPDATE events
      SET 
        status = $1,
        reviewed_at = NOW(),
        reviewed_by = $2,
        notes = $3,
        updated_at = NOW()
      WHERE id = $4
      RETURNING *
    `,
        [update.status, update.reviewed_by || null, update.notes || null, id],
      );

      if (result.rows.length === 0) {
        return res.status(404).json({
          success: false,
          error: "Event not found",
        });
      }

      logger.info(
        `Event ${id} status updated to ${update.status} by ${update.reviewed_by || "unknown"}`,
      );

      res.json({
        success: true,
        data: result.rows[0],
        message: `Event ${update.status}`,
      });
    } catch (error) {
      logger.error("Error updating event status:", error);
      next(error);
    }
  },
);

// ============================================
// STREAM ENDPOINTS
// ============================================

/**
 * GET /api/streams
 * List all streams
 */
router.get(
  "/streams",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const activeOnly = req.query.active === "true";

      let query = `
      SELECT 
        s.*,
        (SELECT COUNT(*) FROM events e WHERE e.stream_id = s.id AND e.status = 'new') as pending_events
      FROM streams s
    `;

      if (activeOnly) {
        query += ` WHERE s.is_active = true`;
      }

      query += ` ORDER BY s.name`;

      const result = await pool.query(query);

      res.json({
        success: true,
        data: result.rows,
        count: result.rows.length,
      });
    } catch (error) {
      logger.error("Error listing streams:", error);
      next(error);
    }
  },
);

/**
 * GET /api/streams/:id
 * Get stream by ID
 */
router.get(
  "/streams/:id",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const { id } = req.params;

      const result = await pool.query(
        `
      SELECT * FROM streams WHERE id = $1
    `,
        [id],
      );

      if (result.rows.length === 0) {
        return res.status(404).json({
          success: false,
          error: "Stream not found",
        });
      }

      res.json({
        success: true,
        data: result.rows[0],
      });
    } catch (error) {
      logger.error("Error getting stream:", error);
      next(error);
    }
  },
);

/**
 * GET /api/streams/:id/events
 * Get events for a specific stream
 */
router.get(
  "/streams/:id/events",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const { id } = req.params;
      const limit = parseInt(req.query.limit as string) || 50;
      const status = req.query.status as string;

      let query = `
      SELECT * FROM events
      WHERE stream_id = $1
    `;
      const params: any[] = [id];
      let paramIndex = 2;

      if (status) {
        query += ` AND status = $${paramIndex++}`;
        params.push(status);
      }

      query += ` ORDER BY start_time DESC LIMIT $${paramIndex}`;
      params.push(limit);

      const result = await pool.query(query, params);

      res.json({
        success: true,
        data: result.rows,
        count: result.rows.length,
      });
    } catch (error) {
      logger.error("Error getting stream events:", error);
      next(error);
    }
  },
);

// ============================================
// CLIP ENDPOINTS
// ============================================

/**
 * GET /api/clips/:id
 * Proxy clip file for playback
 */
router.get(
  "/clips/:id",
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const pool = getPool();
      const { id } = req.params;

      // Get clip path from event
      const result = await pool.query(
        `
      SELECT clip_path FROM events WHERE id = $1
    `,
        [id],
      );

      if (result.rows.length === 0 || !result.rows[0].clip_path) {
        return res.status(404).json({
          success: false,
          error: "Clip not found",
        });
      }

      const clipPath = result.rows[0].clip_path;

      // Check if this is a URL (RTSP service) or local path
      if (clipPath.startsWith("http")) {
        // Redirect to RTSP service
        return res.redirect(clipPath);
      }

      // Send local file
      res.sendFile(clipPath, (err) => {
        if (err) {
          logger.error("Error sending clip file:", err);
          if (!res.headersSent) {
            res.status(404).json({
              success: false,
              error: "Clip file not found",
            });
          }
        }
      });
    } catch (error) {
      logger.error("Error getting clip:", error);
      next(error);
    }
  },
);

export default router;
