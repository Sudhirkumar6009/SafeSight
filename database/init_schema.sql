-- SafeSight Database Schema
-- PostgreSQL Migration Script
-- Version: 1.0.0
-- 
-- Run this script to initialize the safesight database:
-- psql -U postgres -d safesight -f init_schema.sql

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- 1. STREAMS TABLE - Camera/RTSP Source Registry
-- ============================================
-- Stores all connected video sources (CCTV, mobile, webcam, etc.)
-- Used by ingestion & monitoring services

CREATE TABLE IF NOT EXISTS streams (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                VARCHAR(255) NOT NULL,
    url                 TEXT NOT NULL,
    stream_type         VARCHAR(50) DEFAULT 'rtsp',     -- rtsp, rtmp, webcam, file
    location            VARCHAR(255),                    -- Physical location (Gate 3, Platform 2)
    is_active           BOOLEAN DEFAULT true,            -- Enable/disable stream
    status              VARCHAR(50) DEFAULT 'offline',   -- online, offline, error
    last_frame_at       TIMESTAMP WITH TIME ZONE,        -- Health monitoring
    error_message       TEXT,                            -- Debugging stream issues
    custom_threshold    FLOAT CHECK (custom_threshold >= 0 AND custom_threshold <= 1),
    custom_window_seconds INT CHECK (custom_window_seconds > 0),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for streams
CREATE INDEX IF NOT EXISTS idx_streams_is_active ON streams(is_active);
CREATE INDEX IF NOT EXISTS idx_streams_status ON streams(status);
CREATE INDEX IF NOT EXISTS idx_streams_stream_type ON streams(stream_type);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_streams_updated_at
    BEFORE UPDATE ON streams
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 2. INFERENCE_LOGS TABLE - Raw Model Output
-- ============================================
-- Stores raw sliding-window predictions
-- Useful for debugging, model evaluation, threshold tuning, retraining

CREATE TABLE IF NOT EXISTS inference_logs (
    id                  BIGSERIAL PRIMARY KEY,
    stream_id           UUID REFERENCES streams(id) ON DELETE CASCADE,
    timestamp           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    violence_score      FLOAT NOT NULL CHECK (violence_score >= 0 AND violence_score <= 1),
    non_violence_score  FLOAT NOT NULL CHECK (non_violence_score >= 0 AND non_violence_score <= 1),
    inference_time_ms   INT,                             -- Performance tracking
    frame_number        INT,                             -- Debug/trace
    window_start        TIMESTAMP WITH TIME ZONE,        -- Sliding window start
    window_end          TIMESTAMP WITH TIME ZONE         -- Sliding window end
);

-- Indexes for inference_logs (performance critical)
CREATE INDEX IF NOT EXISTS idx_inference_logs_stream_time ON inference_logs(stream_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_inference_logs_timestamp ON inference_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_inference_logs_violence_score ON inference_logs(violence_score);

-- Partition by day for better performance (optional, add if needed)
-- Note: Uncomment for production with high volume
/*
CREATE TABLE inference_logs_y2024m01 PARTITION OF inference_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
*/

-- ============================================
-- 3. EVENTS TABLE - Actual Violence Incidents
-- ============================================
-- Stores event-level, human-meaningful incidents
-- This is what operators, audits, and reports use

CREATE TYPE event_severity AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE event_status AS ENUM ('new', 'confirmed', 'dismissed', 'auto_dismissed');

CREATE TABLE IF NOT EXISTS events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_id           UUID REFERENCES streams(id) ON DELETE SET NULL,
    stream_name         VARCHAR(255) NOT NULL,           -- Denormalized for UI
    
    -- Timing
    start_time          TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time            TIMESTAMP WITH TIME ZONE,
    duration_seconds    INT,                             -- Event duration
    
    -- Confidence Scores
    max_confidence      FLOAT NOT NULL CHECK (max_confidence >= 0 AND max_confidence <= 1),
    avg_confidence      FLOAT NOT NULL CHECK (avg_confidence >= 0 AND avg_confidence <= 1),
    min_confidence      FLOAT NOT NULL CHECK (min_confidence >= 0 AND min_confidence <= 1),
    frame_count         INT DEFAULT 0,                   -- Frames involved
    
    -- Classification
    severity            event_severity DEFAULT 'medium',
    status              event_status DEFAULT 'new',
    
    -- Clip Information
    clip_path           TEXT,                            -- Stored clip path
    clip_duration       INT,                             -- Seconds
    thumbnail_path      TEXT,                            -- UI preview
    
    -- Human Review
    reviewed_at         TIMESTAMP WITH TIME ZONE,
    reviewed_by         VARCHAR(255),                    -- Operator
    notes               TEXT,                            -- Operator remarks
    
    -- Metadata
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for events
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_stream_id ON events(stream_id);
CREATE INDEX IF NOT EXISTS idx_events_stream_time ON events(stream_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at DESC);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_events_status_severity_time ON events(status, severity, start_time DESC);

-- Trigger to auto-update updated_at
CREATE TRIGGER update_events_updated_at
    BEFORE UPDATE ON events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 4. UTILITY FUNCTIONS
-- ============================================

-- Function to calculate severity based on confidence
CREATE OR REPLACE FUNCTION calculate_severity(confidence FLOAT)
RETURNS event_severity AS $$
BEGIN
    IF confidence >= 0.95 THEN
        RETURN 'critical';
    ELSIF confidence >= 0.85 THEN
        RETURN 'high';
    ELSIF confidence >= 0.75 THEN
        RETURN 'medium';
    ELSE
        RETURN 'low';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to cleanup old inference logs (call periodically)
CREATE OR REPLACE FUNCTION cleanup_old_inference_logs(hours_to_keep INT DEFAULT 24)
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM inference_logs
    WHERE timestamp < NOW() - (hours_to_keep || ' hours')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get event statistics for a stream
CREATE OR REPLACE FUNCTION get_stream_event_stats(p_stream_id UUID, p_days INT DEFAULT 7)
RETURNS TABLE (
    total_events BIGINT,
    confirmed_events BIGINT,
    dismissed_events BIGINT,
    avg_confidence FLOAT,
    max_confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_events,
        COUNT(*) FILTER (WHERE status = 'confirmed')::BIGINT as confirmed_events,
        COUNT(*) FILTER (WHERE status = 'dismissed')::BIGINT as dismissed_events,
        AVG(events.max_confidence)::FLOAT as avg_confidence,
        MAX(events.max_confidence)::FLOAT as max_confidence
    FROM events
    WHERE stream_id = p_stream_id
      AND created_at >= NOW() - (p_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 5. VIEWS FOR COMMON QUERIES
-- ============================================

-- View: Recent pending events (for dashboard)
CREATE OR REPLACE VIEW v_pending_events AS
SELECT 
    e.id,
    e.stream_id,
    e.stream_name,
    s.location as stream_location,
    e.start_time,
    e.end_time,
    e.duration_seconds,
    e.max_confidence,
    e.severity,
    e.status,
    e.clip_path,
    e.thumbnail_path,
    e.created_at
FROM events e
LEFT JOIN streams s ON e.stream_id = s.id
WHERE e.status = 'new'
ORDER BY e.start_time DESC;

-- View: Stream health status
CREATE OR REPLACE VIEW v_stream_health AS
SELECT 
    s.id,
    s.name,
    s.stream_type,
    s.location,
    s.status,
    s.last_frame_at,
    s.error_message,
    CASE 
        WHEN s.last_frame_at IS NULL THEN 'never'
        WHEN s.last_frame_at > NOW() - INTERVAL '30 seconds' THEN 'healthy'
        WHEN s.last_frame_at > NOW() - INTERVAL '5 minutes' THEN 'stale'
        ELSE 'dead'
    END as health_status,
    (SELECT COUNT(*) FROM events WHERE stream_id = s.id AND status = 'new') as pending_events
FROM streams s
WHERE s.is_active = true;

-- View: Daily event summary
CREATE OR REPLACE VIEW v_daily_event_summary AS
SELECT 
    DATE(start_time) as event_date,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE severity = 'critical') as critical_count,
    COUNT(*) FILTER (WHERE severity = 'high') as high_count,
    COUNT(*) FILTER (WHERE severity = 'medium') as medium_count,
    COUNT(*) FILTER (WHERE severity = 'low') as low_count,
    AVG(max_confidence) as avg_confidence,
    AVG(duration_seconds) as avg_duration_seconds
FROM events
WHERE start_time >= NOW() - INTERVAL '30 days'
GROUP BY DATE(start_time)
ORDER BY event_date DESC;

-- ============================================
-- 6. SAMPLE DATA (for testing)
-- ============================================

-- Insert sample streams
INSERT INTO streams (name, url, stream_type, location, status) VALUES
    ('Main Entrance', 'rtsp://192.168.1.100:554/stream1', 'rtsp', 'Building A - Front Gate', 'online'),
    ('Parking Lot', 'rtsp://192.168.1.101:554/stream1', 'rtsp', 'Parking Area - North', 'online'),
    ('Test Webcam', '0', 'webcam', 'Development Machine', 'offline'),
    ('Demo Stream', 'file:///path/to/demo.mp4', 'file', 'Test File', 'offline')
ON CONFLICT DO NOTHING;

-- ============================================
-- 7. MAINTENANCE PROCEDURES
-- ============================================

-- Comment: Run these periodically via cron or pg_cron

-- Cleanup old inference logs (run hourly)
-- SELECT cleanup_old_inference_logs(24);

-- Vacuum and analyze (run daily during low usage)
-- VACUUM ANALYZE streams;
-- VACUUM ANALYZE events;
-- VACUUM ANALYZE inference_logs;

COMMENT ON TABLE streams IS 'Camera/RTSP source registry - stores all connected video sources';
COMMENT ON TABLE inference_logs IS 'Raw sliding-window predictions - useful for debugging and model evaluation';
COMMENT ON TABLE events IS 'Actual violence incidents - human-meaningful events for operators and reports';

-- Done!
SELECT 'SafeSight database schema initialized successfully!' as status;
