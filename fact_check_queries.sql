-- SQL queries to work with fact-check results and trace back to original messages

-- 1. Get fact-check results with original message details
SELECT 
    r.ID as original_message_id,
    r.sender_phone,
    r.content_type,
    r.submission_timestamp,
    r.raw_text as original_text,
    f.overall_verdict,
    f.overall_confidence,
    f.num_claims,
    f.input_source,
    f.processing_duration_seconds,
    f.fact_check_timestamp
FROM raw_data r
JOIN fact_check_results f ON r.ID = f.message_id
ORDER BY f.fact_check_timestamp DESC;

-- 2. Get comprehensive view including processed results
SELECT 
    r.ID as message_id,
    r.sender_phone,
    r.content_type,
    r.submission_timestamp as received_time,
    p.processing_timestamp as processed_time,
    f.fact_check_timestamp,
    r.raw_text as original_text,
    f.input_text as fact_checked_text,
    f.input_source as text_source,
    f.overall_verdict,
    f.overall_confidence,
    f.num_claims,
    f.num_sources
FROM raw_data r
LEFT JOIN processed p ON r.ID = p.message_id
LEFT JOIN fact_check_results f ON r.ID = f.message_id
WHERE f.message_id IS NOT NULL
ORDER BY f.fact_check_timestamp DESC;

-- 3. Get fact-check results for a specific message ID
SELECT 
    r.ID as original_message_id,
    r.sender_phone,
    r.raw_text as original_text,
    f.input_text as analyzed_text,
    f.input_source,
    f.overall_verdict,
    f.overall_confidence,
    f.claims_json,
    f.claim_verdicts_json,
    f.crawled_urls_json,
    f.overall_reasoning,
    f.error_message
FROM raw_data r
JOIN fact_check_results f ON r.ID = f.message_id
WHERE r.ID = 'YOUR_MESSAGE_ID_HERE';

-- 4. Get summary statistics by sender
SELECT 
    r.sender_phone,
    COUNT(*) as messages_fact_checked,
    COUNT(CASE WHEN f.overall_verdict = 'SUPPORTED' THEN 1 END) as supported_claims,
    COUNT(CASE WHEN f.overall_verdict = 'NOT_SUPPORTED' THEN 1 END) as unsupported_claims,
    COUNT(CASE WHEN f.overall_verdict = 'MIXED' THEN 1 END) as mixed_verdicts,
    AVG(f.overall_confidence) as avg_confidence,
    AVG(f.num_claims) as avg_claims_per_message,
    AVG(f.processing_duration_seconds) as avg_processing_time
FROM raw_data r
JOIN fact_check_results f ON r.ID = f.message_id
GROUP BY r.sender_phone
ORDER BY messages_fact_checked DESC;

-- 5. Get recent fact-check activity
SELECT 
    r.ID as message_id,
    r.sender_phone,
    r.content_type,
    f.overall_verdict,
    f.num_claims,
    f.input_source,
    f.fact_check_timestamp,
    CASE 
        WHEN f.error_message IS NOT NULL THEN 'ERROR'
        WHEN f.overall_confidence >= 0.8 THEN 'HIGH_CONFIDENCE'
        WHEN f.overall_confidence >= 0.5 THEN 'MEDIUM_CONFIDENCE'
        ELSE 'LOW_CONFIDENCE'
    END as confidence_level
FROM raw_data r
JOIN fact_check_results f ON r.ID = f.message_id
ORDER BY f.fact_check_timestamp DESC
LIMIT 20;
