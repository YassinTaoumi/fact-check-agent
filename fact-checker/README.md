# Fact-Checker System

A comprehensive fact-checking system built with LangGraph, LangChain, SearXNG, and Crawl4AI.

## Features

- **Claim Extraction**: Automatically extracts factual claims from input text
- **Web Search**: Uses SearXNG for comprehensive web searches
- **Content Crawling**: Crawls top 5 webpages for each claim using Crawl4AI
- **AI Summarization**: Summarizes crawled content using Groq API
- **Fact Verification**: Provides verdicts and justifications for each claim

## Architecture

The system uses LangGraph to orchestrate the following workflow:
1. Extract claims from input text
2. For each claim:
   - Search web using SearXNG
   - Crawl top 5 relevant pages
   - Summarize content with AI
3. Analyze all summaries and provide final verdict

## Components

- `fact_checker.py` - Main fact-checker class and workflow
- `claim_extractor.py` - Extract factual claims from text
- `web_searcher.py` - Web search using SearXNG
- `content_crawler.py` - Web page crawling with Crawl4AI
- `summarizer.py` - AI-powered content summarization
- `verdict_analyzer.py` - Final verdict and justification
- `config.py` - Configuration and settings
- `requirements.txt` - Dependencies

## Usage

```python
from fact_checker import FactChecker

# Initialize fact checker
checker = FactChecker()

# Check facts in text
result = checker.check_facts("Your text with claims here...")

# Get results
for claim_result in result['claims']:
    print(f"Claim: {claim_result['claim']}")
    print(f"Verdict: {claim_result['verdict']}")
    print(f"Justification: {claim_result['justification']}")
```

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env`
3. Configure SearXNG instance
4. Run the fact checker

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key
SEARXNG_URL=http://localhost:8080
CRAWL4AI_MAX_PAGES=5
```
