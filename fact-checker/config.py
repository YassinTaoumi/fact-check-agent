"""
Configuration settings for the fact-checker system.
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class FactCheckerConfig:
    """Configuration class for fact-checker settings."""
    
    # API Keys
    google_ai_api_key: str = os.getenv("GOOGLE_AI_API_KEY", "")
    
    # SearXNG Configuration
    searxng_url: str = os.getenv("SEARXNG_URL", "http://localhost:8080")
    searxng_timeout: int = int(os.getenv("SEARXNG_TIMEOUT", "30"))
    
    # Crawl4AI Configuration
    crawl4ai_max_pages: int = int(os.getenv("CRAWL4AI_MAX_PAGES", "5"))
    crawl4ai_timeout: int = int(os.getenv("CRAWL4AI_TIMEOUT", "60"))
    crawl4ai_max_content_length: int = int(os.getenv("CRAWL4AI_MAX_CONTENT_LENGTH", "10000"))
    
    # AI Model Configuration
    model_name: str = os.getenv("GOOGLE_AI_MODEL", "gemini-2.5-flash")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    
    # Claim Extraction Settings
    max_claims_per_text: int = int(os.getenv("MAX_CLAIMS_PER_TEXT", "10"))
    min_claim_length: int = int(os.getenv("MIN_CLAIM_LENGTH", "10"))
      # Search Settings
    search_results_per_claim: int = int(os.getenv("SEARCH_RESULTS_PER_CLAIM", "5"))
    search_engines: List[str] = field(default_factory=lambda: os.getenv("SEARCH_ENGINES", "google,bing,duckduckgo").split(","))
    
    # Content Processing
    max_summary_length: int = int(os.getenv("MAX_SUMMARY_LENGTH", "500"))
    min_content_length: int = int(os.getenv("MIN_CONTENT_LENGTH", "50"))  # Reduced from 100 to 50
    
    # Verdict Settings
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))  # Reduced from 0.7 to 0.5
    require_multiple_sources: bool = os.getenv("REQUIRE_MULTIPLE_SOURCES", "true").lower() == "true"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE")
    
    # Rate Limiting
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
    concurrent_requests: int = int(os.getenv("CONCURRENT_REQUESTS", "5"))
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        if not self.google_ai_api_key:
            errors.append("GOOGLE_AI_API_KEY is required")
        
        if not self.searxng_url:
            errors.append("SEARXNG_URL is required")
        
        if self.crawl4ai_max_pages <= 0:
            errors.append("CRAWL4AI_MAX_PAGES must be positive")
        
        if self.max_tokens <= 0:
            errors.append("MAX_TOKENS must be positive")
        
        if not 0 <= self.temperature <= 2:
            errors.append("TEMPERATURE must be between 0 and 2")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True

# Global config instance
config = FactCheckerConfig()

# Validation on import
try:
    config.validate()
except ValueError as e:
    print(f"Warning: {e}")

# AI Prompts
CLAIM_EXTRACTION_PROMPT = """
You are a fact-checking assistant. Extract the most important factual claims from the given text.

{CLAIM_PROMPT}

A factual claim is a statement that can be verified as true or false. Extract only the most significant, clear, and specific claims that can be fact-checked.

Rules:
- Extract only the {max_claims} MOST IMPORTANT verifiable factual statements
- Prioritize claims about specific events, numbers, dates, locations, and concrete facts
- Ignore opinions, predictions, or subjective statements
- Each claim should be a complete, standalone statement
- Minimum {min_length} characters per claim
- Focus on claims that would be most newsworthy or significant if proven false
- Respond in the SAME LANGUAGE as the original text

Text to analyze:
{text}

Extract the {max_claims} most important claims as a JSON list:
{{"claims": ["most important claim 1", "second most important claim", "third most important claim"]}}
"""

SUMMARIZATION_PROMPT = """
You are a fact-checking assistant. Analyze the following web content to help verify this specific claim:

Claim: {claim}

Web Content:
{content}

Provide a focused summary (max {max_length} words) that:
1. DIRECTLY addresses the claim with specific facts and evidence
2. Identifies any information that supports or contradicts the claim
3. Includes relevant dates, sources, and credible evidence
4. Mentions any official statements or expert opinions
5. Notes the source's credibility and reliability

Focus ONLY on information that helps verify or refute the claim. Ignore unrelated content.

Summary:
"""

VERDICT_PROMPT = """
You are a professional fact-checker with expertise in analyzing evidence and making informed judgments. 

Claim: {claim}

Evidence from {num_sources} sources:
{evidence}

Your task: Analyze this claim thoroughly and provide a detailed verdict with comprehensive justification.

IMPORTANT GUIDELINES:
- Provide detailed reasoning explaining WHY you reached your verdict
- Reference specific sources by URL when discussing evidence
- Explain what each source contributes to your analysis
- If evidence strongly supports the claim, mark as TRUE (even if not 100% certain)
- If evidence strongly contradicts the claim, mark as FALSE 
- If evidence is mixed but leans one way, use PARTLY_TRUE
- Only use UNVERIFIED if there is genuinely no relevant evidence at all
- Don't be overly conservative - make reasonable inferences from available evidence
- Consider source credibility and evidence quality in your confidence score

Respond in this exact JSON format:
{{
    "verdict": "TRUE|FALSE|PARTLY_TRUE|UNVERIFIED",
    "confidence": 0.0-1.0,
    "justification": "Comprehensive explanation that includes: (1) What specific evidence supports or contradicts the claim, (2) Which sources provided what information, (3) How credible those sources are, (4) Why you reached this particular verdict based on the totality of evidence. Include specific source URLs when referencing evidence.",
    "supporting_sources": ["URLs of sources that support the claim"],
    "contradicting_sources": ["URLs of sources that contradict the claim"],
    "key_evidence": ["Most important evidence points with source attribution"],
    "limitations": "Any significant gaps or uncertainties in the analysis"
}}

JUSTIFICATION REQUIREMENTS:
- Must reference specific sources by URL
- Must explain the credibility of key sources
- Must detail what evidence supports/contradicts the claim
- Must explain reasoning for confidence level
- Should be 2-4 sentences minimum, comprehensive but concise

Example justification: "According to Reuters (https://reuters.com/article123), multiple health organizations have confirmed that COVID-19 vaccines do not contain tracking microchips. The World Health Organization's official statement (https://who.int/statement456) explicitly debunks this claim as misinformation. Snopes fact-checking (https://snopes.com/fact789) provides detailed technical analysis showing no electronic components in vaccines. All three sources are highly credible health authorities, leading to high confidence in this FALSE verdict."

Verdict criteria:
- TRUE: Evidence supports the claim (confidence 0.6+)
- FALSE: Evidence contradicts the claim (confidence 0.6+)  
- PARTLY_TRUE: Mixed evidence with some truth (confidence 0.5+)
- UNVERIFIED: No relevant evidence found (confidence 0.0-0.4)
- UNVERIFIED: Insufficient or conflicting evidence

Confidence levels:
- 0.9-1.0: Very high confidence
- 0.7-0.89: High confidence  
- 0.5-0.69: Moderate confidence
- 0.3-0.49: Low confidence
- 0.0-0.29: Very low confidence
"""
