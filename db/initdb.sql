CREATE TABLE IF NOT EXISTS users (
                                     id SERIAL PRIMARY KEY,
                                     name TEXT NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    gender VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

CREATE TABLE IF NOT EXISTS voices (
  id SERIAL PRIMARY KEY,
  user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  category_id INT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    previous_voice_id INT REFERENCES voices(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
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
    part VARCHAR(50),
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

CREATE TABLE IF NOT EXISTS voice_segment_versions (
                                                      id SERIAL PRIMARY KEY,
                                                      segment_id INT NOT NULL REFERENCES voice_segments(id) ON DELETE CASCADE,
    version_no INT NOT NULL,
    text TEXT NOT NULL,
    segment_url TEXT NOT NULL,
    db FLOAT,
    pitch_mean_hz FLOAT,
    rate_wpm FLOAT,
    pause_ratio FLOAT,
    prosody_score FLOAT,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_segment_version UNIQUE (segment_id, version_no)
    );

CREATE INDEX IF NOT EXISTS idx_category_user_id ON categories(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_user_id ON voices(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_category_id ON voices(category_id);
CREATE INDEX IF NOT EXISTS idx_voice_prev_id ON voices(previous_voice_id);
CREATE INDEX IF NOT EXISTS idx_voice_segments_voice_id ON voice_segments(voice_id);
CREATE INDEX IF NOT EXISTS idx_voice_segments_order_no ON voice_segments(order_no);
CREATE INDEX IF NOT EXISTS idx_vsv_segment_id ON voice_segment_versions(segment_id);

-- 테스트 데이터 삽입
-- Users 
INSERT INTO users (id, name, email, password, gender, created_at) VALUES
(1, 'gAAAAABpGYfcRxIzg1NYff6Lp6AY793InZivw1496jI0P88AZJ150vGAQKp9YNGyB1AbF99KO3FHzgyWerFuiB-MXv8phgd58XKHN9s28cvwzAzjG3j-kNc=', 'test@a.com', '$2b$12$gChCSSqmQ3Fzt.lcOzFdLOAq3pBvCjWxiB/g0z0k/1Lk9MZAmSfcy', 'MALE', '2025-11-16 08:14:20.142')
ON CONFLICT (id) DO NOTHING;

-- Categories
INSERT INTO categories (id, user_id, name, created_at) VALUES
(1, 1, '프레젠테이션', '2025-11-16 08:15:26.658'),
(2, 1, '회의록', '2025-11-16 08:15:31.122')
ON CONFLICT (id) DO NOTHING;

-- 시퀀스 업데이트 (테스트 데이터 삽입 후 시퀀스를 최신 상태로 설정)
SELECT setval('users_id_seq', (SELECT MAX(id) FROM users));
SELECT setval('categories_id_seq', (SELECT MAX(id) FROM categories));