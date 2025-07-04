-- Create fact_check_results table for storing fact-checking results
CREATE TABLE IF NOT EXISTS fact_check_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,                    -- Reference to original message ID  
    fact_check_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Input text that was fact-checked
    input_text TEXT NOT NULL,                    -- The text that was analyzed (from OCR, transcription, raw text, etc.)
    input_source TEXT NOT NULL,                  -- Source of text: 'raw_text', 'ocr', 'transcription', 'pdf_extraction'
    
    -- Extracted claims
    claims_json TEXT,                            -- JSON array of extracted claims
    num_claims INTEGER DEFAULT 0,               -- Number of claims extracted
    
    -- Web search and crawling results
    search_queries_json TEXT,                    -- JSON array of search queries used
    crawled_urls_json TEXT,                      -- JSON array of URLs that were crawled
    num_sources INTEGER DEFAULT 0,              -- Number of sources crawled
    
    -- Summarization results
    summaries_json TEXT,                         -- JSON of summaries for each claim
    
    -- Individual claim verdicts
    claim_verdicts_json TEXT,                    -- JSON array of verdicts for each claim
    
    -- Overall verdict
    overall_verdict TEXT NOT NULL,               -- SUPPORTED, NOT_SUPPORTED, MIXED, INSUFFICIENT_INFO
    overall_confidence REAL DEFAULT 0.0,        -- Confidence score 0.0-1.0
    overall_reasoning TEXT,                      -- Explanation of overall verdict
    
    -- Processing metadata
    processing_duration_seconds REAL,
    num_llm_requests INTEGER DEFAULT 0,         -- Track LLM usage
    error_message TEXT,
    
    -- Foreign key constraint
    FOREIGN KEY (message_id) REFERENCES raw_data(ID) ON DELETE CASCADE
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_fact_check_message_id ON fact_check_results(message_id);
CREATE INDEX IF NOT EXISTS idx_fact_check_timestamp ON fact_check_results(fact_check_timestamp);
CREATE INDEX IF NOT EXISTS idx_fact_check_verdict ON fact_check_results(overall_verdict);
CREATE INDEX IF NOT EXISTS idx_fact_check_confidence ON fact_check_results(overall_confidence);
