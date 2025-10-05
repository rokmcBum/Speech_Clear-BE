-- initdb.sql

CREATE TABLE IF NOT EXISTS voices (
                                      id SERIAL PRIMARY KEY,
                                      filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    original_url TEXT NOT NULL,
    duration_sec FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

CREATE TABLE IF NOT EXISTS voice_segments (
                                              id SERIAL PRIMARY KEY,
                                              voice_id INT NOT NULL REFERENCES voices(id) ON DELETE CASCADE,
    order_no INT NOT NULL,
    text TEXT NOT NULL,
    start_time FLOAT,
    end_time FLOAT,
    segment_url TEXT,
    db FLOAT,
    pitch_mean_hz FLOAT,
    rate_wpm FLOAT,
    pause_ratio FLOAT,
    prosody_score FLOAT,
    feedback TEXT
    );

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_voice_segments_voice_id ON voice_segments(voice_id);
CREATE INDEX IF NOT EXISTS idx_voice_segments_order_no ON voice_segments(order_no);
