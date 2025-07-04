-- Create processed table to store results from all extractors
CREATE TABLE IF NOT EXISTS processed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,                    -- Reference to original message ID
    processing_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_status TEXT DEFAULT 'pending',    -- pending, completed, failed
    
    -- Text Processing Results
    cleaned_text TEXT,                           -- from text_ext.py
    
    -- Image Processing Results  
    ocr_text TEXT,                              -- from ocr_ext.py
    ai_image_detection TEXT,                    -- from artificial_image_ext.py
    image_modification_detection TEXT,          -- from modification_ext.py
    
    -- Video Processing Results
    video_transcription TEXT,                   -- from video_transcriber.py
    
    -- PDF Processing Results
    pdf_text_extraction TEXT,                   -- from pdf_ext.py
    
    -- Link Processing Results
    link_content TEXT,                          -- from link_crawler.py
    
    -- Fact-Check Results
    fact_check_status TEXT DEFAULT 'pending',   -- pending, completed, failed, skipped
    fact_check_overall_verdict TEXT,            -- overall verdict from fact-checker
    fact_check_individual_claims TEXT,          -- JSON array of individual claim results
    fact_check_confidence_score REAL,           -- confidence score 0-1
    fact_check_sources_used TEXT,               -- JSON array of sources used
    fact_check_timestamp DATETIME,              -- when fact-check was completed
    fact_check_processing_time REAL,            -- time taken for fact-checking in seconds
    fact_check_error_message TEXT,              -- error message if fact-check failed
    
    -- General metadata
    processing_duration_seconds REAL,
    error_message TEXT,
    extractor_versions TEXT,                    -- JSON with version info of each extractor used
    
    -- Foreign key constraint
    FOREIGN KEY (message_id) REFERENCES raw_data(ID) ON DELETE CASCADE
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_processed_message_id ON processed(message_id);
CREATE INDEX IF NOT EXISTS idx_processed_timestamp ON processed(processing_timestamp);
CREATE INDEX IF NOT EXISTS idx_processed_status ON processed(processing_status);
CREATE INDEX IF NOT EXISTS idx_processed_fact_check_status ON processed(fact_check_status);
CREATE INDEX IF NOT EXISTS idx_processed_fact_check_timestamp ON processed(fact_check_timestamp);
