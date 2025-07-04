-- raw_data_schema.sql
CREATE TABLE IF NOT EXISTS raw_data (
    ID TEXT PRIMARY KEY,
    UUID TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('whatsapp', 'telegram')),
    sender_phone TEXT NOT NULL, -- Simplified: just the phone number
    is_group_message BOOLEAN NOT NULL DEFAULT 0,
    group_name TEXT, -- Group name if is_group_message is true
    channel_name TEXT, -- Channel name if message is from a channel
    content_type TEXT NOT NULL CHECK (content_type IN ('audio', 'video', 'pdf', 'image', 'text', 'document', 'link')),
    content_url TEXT, -- File storage path
    raw_text TEXT, -- For direct text submissions
    submission_timestamp DATETIME NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'pending',
    user_identifier TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal'
);

CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp ON raw_data(submission_timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_data_user ON raw_data(user_identifier);
CREATE INDEX IF NOT EXISTS idx_raw_data_status ON raw_data(processing_status);
CREATE INDEX IF NOT EXISTS idx_raw_data_sender_phone ON raw_data(sender_phone);
CREATE INDEX IF NOT EXISTS idx_raw_data_group ON raw_data(is_group_message);