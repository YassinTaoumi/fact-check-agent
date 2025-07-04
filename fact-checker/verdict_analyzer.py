"""
Verdict analyzer component for the fact-checker system.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from config import config, VERDICT_PROMPT
from rate_limiter import llm_rate_limiter

logger = logging.getLogger(__name__)

class VerdictAnalyzer:
    """Analyze summaries and provide fact-checking verdicts."""
    
    def __init__(self):
        """Initialize the verdict analyzer."""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=config.google_ai_api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        self.confidence_threshold = config.confidence_threshold
        logger.info("VerdictAnalyzer initialized with Google AI")
    
    async def analyze_claim(self, claim: str, summaries: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
        """
        Analyze a claim based on summarized evidence.
        
        Args:
            claim: The claim to fact-check
            summaries: List of content summaries
            language: ISO language code (en, ar, fr, es, etc.)
            
        Returns:
            Verdict analysis with justification
        """
        try:
            if not summaries:
                return self._create_no_evidence_verdict(claim, language)
            
            # Filter and rank summaries
            valid_summaries = self._filter_valid_summaries(summaries)
            
            if not valid_summaries:
                return self._create_insufficient_evidence_verdict(claim, summaries, language)
            
            # Prepare evidence for analysis
            evidence_text = self._prepare_evidence_text(valid_summaries)
            
            # Get AI verdict
            verdict_data = await self._generate_ai_verdict(claim, evidence_text, len(valid_summaries), language)
            
            # Process and validate verdict
            processed_verdict = self._process_verdict_response(verdict_data, claim, valid_summaries, language)
            
            logger.info(f"Generated verdict for claim: {claim[:50]}... - {processed_verdict.get('verdict', 'UNKNOWN')}")
            
            return processed_verdict
            
        except Exception as e:
            logger.error(f"Error analyzing claim '{claim}': {e}")
            return self._create_error_verdict(claim, str(e))
    
    async def _generate_ai_verdict(self, claim: str, evidence: str, num_sources: int, language: str = "en") -> str:
        """Generate AI verdict based on evidence."""
        try:
            # Import language utilities here to avoid circular imports
            from language_utils import get_language_templates, prepare_prompt
            
            # Add language instruction to the prompt
            lang_info = get_language_templates(language)
            language_instruction = ""
            if language == "ar":
                language_instruction = "\n\nIMPORTANT: يجب الرد باللغة العربية في قسم 'justification'. استخدم اللغة العربية في جميع التفسيرات والمبررات.\n"
            elif language == "fr":
                language_instruction = "\n\nIMPORTANT: Vous devez répondre en français dans la section 'justification'. Utilisez le français pour toutes les explications et justifications.\n"
            elif language == "es":
                language_instruction = "\n\nIMPORTANT: Debe responder en español en la sección 'justification'. Use español para todas las explicaciones y justificaciones.\n"
            
            # Prepare the prompt with language-specific templates
            prompt = prepare_prompt(
                VERDICT_PROMPT + language_instruction,
                language,
                claim=claim,
                evidence=evidence,
                num_sources=num_sources
            )
            
            # Get AI response
            messages = [
                SystemMessage(content="You are a professional fact-checker with expertise in evidence analysis and critical thinking."),
                HumanMessage(content=prompt)
            ]
            
            await llm_rate_limiter.wait_if_needed()
            response = self.llm(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI verdict: {e}")
            return json.dumps({
                "verdict": "UNVERIFIED",
                "confidence": 0.0,
                "justification": f"Error generating verdict: {str(e)}",
                "supporting_sources": [],
                "contradicting_sources": [],
                "key_evidence": [],
                "limitations": "Technical error occurred during analysis"
            })
    
    def _prepare_evidence_text(self, summaries: List[Dict[str, Any]]) -> str:
        """Prepare evidence text from summaries."""
        evidence_parts = []
        
        for i, summary in enumerate(summaries, 1):
            source_info = f"Source {i}: {summary.get('source_title', 'Unknown Title')}"
            source_url = summary.get('source_url', 'No URL')
            credibility = summary.get('credibility_score', 0)
            relevance = summary.get('relevance_score', 0)
            content = summary.get('summary', 'No summary available')
            
            evidence_part = f"""
{source_info}
URL: {source_url}
Credibility Score: {credibility:.2f}/1.0
Relevance Score: {relevance:.2f}/1.0
Content: {content}
"""
            evidence_parts.append(evidence_part)
        
        return "\n" + "="*50 + "\n".join(evidence_parts)
    
    def _process_verdict_response(self, verdict_response: str, claim: str, summaries: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
        """Process and validate the AI verdict response."""
        try:
            # Try to parse JSON response
            verdict_data = self._parse_verdict_json(verdict_response, language)
            
            # Validate and enhance verdict
            processed_verdict = {
                'claim': claim,
                'verdict': self._validate_verdict_type(verdict_data.get('verdict', 'UNVERIFIED')),
                'confidence': self._validate_confidence(verdict_data.get('confidence', 0.0)),
                'justification': verdict_data.get('justification', 'No justification provided'),
                'supporting_sources': verdict_data.get('supporting_sources', []),
                'contradicting_sources': verdict_data.get('contradicting_sources', []),
                'key_evidence': verdict_data.get('key_evidence', []),
                'limitations': verdict_data.get('limitations', ''),
                
                # Additional metadata
                'analysis_metadata': {
                    'sources_analyzed': len(summaries),
                    'avg_credibility': sum(s.get('credibility_score', 0) for s in summaries) / len(summaries) if summaries else 0,
                    'avg_relevance': sum(s.get('relevance_score', 0) for s in summaries) / len(summaries) if summaries else 0,
                    'has_high_credibility_sources': any(s.get('credibility_score', 0) > 0.8 for s in summaries),
                    'has_multiple_sources': len(summaries) > 1,
                    'analysis_timestamp': self._get_current_timestamp()
                },
                
                # Source details
                'source_details': [
                    {
                        'url': s.get('source_url', ''),
                        'title': s.get('source_title', ''),
                        'credibility_score': s.get('credibility_score', 0),
                        'relevance_score': s.get('relevance_score', 0)
                    }
                    for s in summaries
                ]
            }
            
            # Add quality assessment
            processed_verdict['quality_assessment'] = self._assess_verdict_quality(processed_verdict)
            
            return processed_verdict
            
        except Exception as e:
            logger.error(f"Error processing verdict response: {e}")
            return self._create_error_verdict(claim, f"Error processing verdict: {str(e)}")
    
    def _parse_verdict_json(self, response: str, language: str = "en") -> Dict[str, Any]:
        """Parse JSON from the verdict response."""
        try:
            import re
            
            # First, try to find JSON in fenced markdown code blocks (with optional 'json' label)
            markdown_json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if markdown_json_match:
                json_str = markdown_json_match.group(1)
                logger.info(f"Found JSON in markdown block: {json_str[:100]}...")
                return json.loads(json_str)
            
            # Skipping simple JSON extraction here to avoid truncated captures; proceed to robust brace matching
            logger.info("Attempting robust brace matching for JSON extraction")
            for start_idx, char in enumerate(response):
                if char == '{':
                    brace_count = 1
                    for end_idx in range(start_idx + 1, len(response)):
                        if response[end_idx] == '{':
                            brace_count += 1
                        elif response[end_idx] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                candidate = response[start_idx:end_idx+1]
                                try:
                                    return json.loads(candidate)
                                except Exception:
                                    break
            
            # As a last resort, extract substring between first and last braces
            logger.info("Attempting to extract JSON substring from response")
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                substring = response[start:end+1]
                try:
                    return json.loads(substring)
                except Exception as e:
                    logger.warning(f"Failed to parse extracted JSON substring: {e}")
            # Attempt JSON decoding at any position using JSONDecoder
            logger.info("Attempting to decode JSON via raw_decode scanning")
            decoder = json.JSONDecoder()
            for idx in range(len(response)):
                if response[idx] != '{':
                    continue
                try:
                    obj, end_idx = decoder.raw_decode(response[idx:])
                    return obj
                except json.JSONDecodeError:
                    continue
            # Try parsing entire response as JSON as final fallback
            logger.info("Attempting to parse entire response as JSON")
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verdict JSON: {e}")
            logger.warning(f"Response was: {response[:500]}...")
            
            import re
            # Try to extract any 'justification' field text from raw response
            just_text = None
            match = re.search(r'"justification"\s*:\s*"(.+?)"', response, re.DOTALL)
            if match:
                raw_val = match.group(1)
                # unescape JSON string
                try:
                    just_text = json.loads(f'"{raw_val}"')
                except Exception:
                    just_text = raw_val
                logger.info("Extracted 'justification' from raw response despite JSON parse error.")
            # Create fallback messages per language
            if language == "ar":
                default_msg = "لا يمكن إكمال تحليل التحقق من الحقائق بسبب خطأ تقني في تحليل الاستجابة."
                limitations = "غير قادر على تحليل الاستجابة المنظمة بسبب خطأ في التنسيق"
            elif language == "fr":
                default_msg = "L'analyse de vérification des faits n'a pas pu être terminée en raison d'une erreur technique d'analyse."
                limitations = "Impossible d'analyser la réponse structurée en raison d'une erreur de format"
            elif language == "es":
                default_msg = "El análisis de verificación de hechos no se pudo completar debido a un error técnico de análisis."
                limitations = "No se puede analizar la respuesta estructurada debido a un error de formato"
            else:
                default_msg = "The fact-checking analysis could not be completed due to a technical parsing error."
                limitations = "Unable to parse structured verdict response due to format error"
            justification = just_text if just_text else f"{default_msg} Response preview: {response[:150]}..."
            # Return fallback verdict preserving any extracted justification
            return {
                "verdict": "UNVERIFIED",
                "confidence": 0.1,
                "justification": justification,
                "supporting_sources": [],
                "contradicting_sources": [],
                "key_evidence": [],
                "limitations": limitations
            }
    
    def _validate_verdict_type(self, verdict: str) -> str:
        """Validate and normalize verdict type."""
        valid_verdicts = ["TRUE", "FALSE", "PARTLY_TRUE", "UNVERIFIED"]
        verdict_upper = str(verdict).upper()
        
        if verdict_upper in valid_verdicts:
            return verdict_upper
        
        # Try to map common variations
        verdict_mapping = {
            "CORRECT": "TRUE",
            "ACCURATE": "TRUE",
            "VERIFIED": "TRUE",
            "CONFIRMED": "TRUE",
            "INCORRECT": "FALSE",
            "INACCURATE": "FALSE",
            "WRONG": "FALSE",
            "DEBUNKED": "FALSE",
            "PARTIALLY_TRUE": "PARTLY_TRUE",
            "MIXED": "PARTLY_TRUE",
            "INCONCLUSIVE": "UNVERIFIED",
            "UNKNOWN": "UNVERIFIED",
            "INSUFFICIENT": "UNVERIFIED"
        }
        
        return verdict_mapping.get(verdict_upper, "UNVERIFIED")
    
    def _validate_confidence(self, confidence: Any) -> float:
        """Validate and normalize confidence score."""
        try:
            conf = float(confidence)
            return max(0.0, min(1.0, conf))  # Clamp between 0 and 1
        except (ValueError, TypeError):
            return 0.0
    
    def _filter_valid_summaries(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter summaries to only include valid ones with content."""
        valid_summaries = []
        
        for summary in summaries:
            # Skip summaries with errors
            if summary.get('error'):
                continue
            
            # Skip summaries without content
            if not summary.get('summary', '').strip():
                continue
            
            # Skip very low relevance summaries
            if summary.get('relevance_score', 0) < 0.1:
                continue
            
            valid_summaries.append(summary)
        
        # Sort by quality (credibility + relevance)
        valid_summaries.sort(
            key=lambda s: (s.get('credibility_score', 0) + s.get('relevance_score', 0)) / 2,
            reverse=True
        )
        
        return valid_summaries
    
    def _assess_verdict_quality(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the verdict analysis."""
        metadata = verdict.get('analysis_metadata', {})
        
        quality_factors = {
            'source_count': min(metadata.get('sources_analyzed', 0) / 3, 1.0),  # Normalize to 3+ sources
            'credibility': metadata.get('avg_credibility', 0),
            'relevance': metadata.get('avg_relevance', 0),
            'confidence': verdict.get('confidence', 0),
            'has_high_credibility': 1.0 if metadata.get('has_high_credibility_sources', False) else 0.5,
            'has_multiple_sources': 1.0 if metadata.get('has_multiple_sources', False) else 0.7
        }
        
        # Calculate overall quality score
        weights = {
            'source_count': 0.2,
            'credibility': 0.25,
            'relevance': 0.2,
            'confidence': 0.15,
            'has_high_credibility': 0.1,
            'has_multiple_sources': 0.1
        }
        
        quality_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = "HIGH"
        elif quality_score >= 0.6:
            quality_level = "MEDIUM"
        elif quality_score >= 0.4:
            quality_level = "LOW"
        else:
            quality_level = "VERY_LOW"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'quality_factors': quality_factors,
            'recommendations': self._get_quality_recommendations(quality_factors, verdict)
        }
    
    def _get_quality_recommendations(self, quality_factors: Dict[str, float], verdict: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving verdict quality."""
        recommendations = []
        
        if quality_factors['source_count'] < 0.5:
            recommendations.append("Consider finding more sources to strengthen the analysis")
        
        if quality_factors['credibility'] < 0.6:
            recommendations.append("Try to include more credible sources (news organizations, academic sources)")
        
        if quality_factors['relevance'] < 0.5:
            recommendations.append("Search for more directly relevant content related to the claim")
        
        if not quality_factors['has_high_credibility']:
            recommendations.append("Include authoritative sources like fact-checking sites or academic publications")
        
        if verdict.get('confidence', 0) < self.confidence_threshold:
            recommendations.append("Low confidence - consider this verdict preliminary pending additional evidence")
        
        return recommendations
    
    def _create_no_evidence_verdict(self, claim: str, language: str = "en") -> Dict[str, Any]:
        """Create verdict when no evidence is available."""
        # Import language utilities here to avoid circular imports
        from language_utils import translate_verdict, get_language_templates
        
        # Get language-specific templates and translations
        lang_templates = get_language_templates(language)
        
        # Default justification in the right language
        if language == "ar":
            justification = "لم يتم العثور على أدلة لتقييم هذا الادعاء"
        elif language == "fr":
            justification = "Aucune preuve trouvée pour évaluer cette affirmation"
        elif language == "es":
            justification = "No se encontraron pruebas para evaluar esta afirmación"
        else:
            justification = "No evidence found to evaluate this claim"
            
        # Default limitations in the right language
        if language == "ar":
            limitations = "لم تكن هناك مصادر متاحة للتحليل"
        elif language == "fr":
            limitations = "Aucune source n'était disponible pour l'analyse"
        elif language == "es":
            limitations = "No había fuentes disponibles para el análisis"
        else:
            limitations = "No sources were available for analysis"
        
        return {
            'claim': claim,
            'verdict': 'UNVERIFIED',
            'confidence': 0.0,
            'justification': justification,
            'supporting_sources': [],
            'contradicting_sources': [],
            'key_evidence': [],
            'limitations': limitations,
            'analysis_metadata': {
                'sources_analyzed': 0,
                'avg_credibility': 0,
                'avg_relevance': 0,
                'has_high_credibility_sources': False,
                'has_multiple_sources': False,
                'analysis_timestamp': self._get_current_timestamp(),
                'language': language
            },
            'source_details': [],
            'quality_assessment': {
                'quality_score': 0.0,
                'quality_level': 'VERY_LOW',
                'recommendations': ['Find sources that discuss this claim', 'Use more specific search terms']
            }
        }
    
    def _create_insufficient_evidence_verdict(self, claim: str, summaries: List[Dict[str, Any]], language: str = "en") -> Dict[str, Any]:
        """Create verdict when evidence is insufficient but try to make reasonable inferences."""
        # Try to extract any available information from summaries
        partial_info = []
        total_relevance = 0
        total_credibility = 0
        
        for summary in summaries:
            if summary.get('summary') and not summary.get('error'):
                partial_info.append(summary['summary'])
                total_relevance += summary.get('relevance_score', 0)
                total_credibility += summary.get('credibility_score', 0)
        
        # If we have some information, try to make a cautious inference
        if partial_info and len(partial_info) > 0:
            avg_relevance = total_relevance / len(summaries) if summaries else 0
            avg_credibility = total_credibility / len(summaries) if summaries else 0
            
            # If we have some decent relevance and credibility, be less conservative
            if avg_relevance >= 0.3 and avg_credibility >= 0.4:
                # Analyze the partial information to see if we can infer anything
                combined_info = " ".join(partial_info)
                
                # Simple keyword analysis for obvious false claims
                false_indicators = ['debunked', 'false', 'fake', 'hoax', 'misinformation', 'no evidence', 'unsubstantiated']
                true_indicators = ['confirmed', 'verified', 'official', 'documented', 'evidence shows', 'studies show']
                
                false_count = sum(1 for indicator in false_indicators if indicator in combined_info.lower())
                true_count = sum(1 for indicator in true_indicators if indicator in combined_info.lower())
                
                if false_count > true_count and false_count > 0:
                    return {
                        'claim': claim,
                        'verdict': 'FALSE',
                        'confidence': min(0.6, 0.4 + (false_count * 0.1)),
                        'justification': f'Available evidence suggests the claim is false. Found {false_count} indicators suggesting the claim is debunked or unsubstantiated.',
                        'supporting_sources': [],
                        'contradicting_sources': [s.get('source_url', '') for s in summaries if s.get('source_url')],
                        'key_evidence': [f"Evidence suggests: {info[:100]}..." for info in partial_info[:3]],
                        'limitations': f'Based on limited evidence from {len(summaries)} sources',
                        'analysis_metadata': {
                            'sources_analyzed': len(summaries),
                            'avg_credibility': avg_credibility,
                            'avg_relevance': avg_relevance,
                            'has_high_credibility_sources': any(s.get('credibility_score', 0) > 0.7 for s in summaries),
                            'has_multiple_sources': len(summaries) > 1,
                            'analysis_timestamp': self._get_current_timestamp()
                        },
                        'source_details': summaries[:3],
                        'quality_assessment': {
                            'quality_score': 0.5,
                            'quality_level': 'MEDIUM',
                            'recommendations': ['Confidence could be improved with more comprehensive sources']
                        }
                    }
                elif true_count > false_count and true_count > 0:
                    return {
                        'claim': claim,
                        'verdict': 'TRUE',
                        'confidence': min(0.6, 0.4 + (true_count * 0.1)),
                        'justification': f'Available evidence suggests the claim has merit. Found {true_count} indicators supporting the claim.',
                        'supporting_sources': [s.get('source_url', '') for s in summaries if s.get('source_url')],
                        'contradicting_sources': [],
                        'key_evidence': [f"Evidence suggests: {info[:100]}..." for info in partial_info[:3]],
                        'limitations': f'Based on limited evidence from {len(summaries)} sources',
                        'analysis_metadata': {
                            'sources_analyzed': len(summaries),
                            'avg_credibility': avg_credibility,
                            'avg_relevance': avg_relevance,
                            'has_high_credibility_sources': any(s.get('credibility_score', 0) > 0.7 for s in summaries),
                            'has_multiple_sources': len(summaries) > 1,
                            'analysis_timestamp': self._get_current_timestamp()
                        },
                        'source_details': summaries[:3],
                        'quality_assessment': {
                            'quality_score': 0.5,
                            'quality_level': 'MEDIUM',
                            'recommendations': ['Confidence could be improved with more comprehensive sources']
                        }
                    }
        
        # Default to unverified only if we really have no useful information
        return {
            'claim': claim,
            'verdict': 'UNVERIFIED',
            'confidence': 0.2,
            'justification': f'Limited evidence available. Analyzed {len(summaries)} sources but could not find sufficient information to make a determination.',
            'supporting_sources': [],
            'contradicting_sources': [],
            'key_evidence': [f"Partial info: {info[:100]}..." for info in partial_info[:2]] if partial_info else [],
            'limitations': f'Only {len(summaries)} sources found, none contained comprehensive relevant information',
            'analysis_metadata': {
                'sources_analyzed': len(summaries),
                'avg_credibility': total_credibility / len(summaries) if summaries else 0,
                'avg_relevance': total_relevance / len(summaries) if summaries else 0,
                'has_high_credibility_sources': any(s.get('credibility_score', 0) > 0.7 for s in summaries),
                'has_multiple_sources': len(summaries) > 1,
                'analysis_timestamp': self._get_current_timestamp()
            },
            'source_details': summaries[:3],
            'quality_assessment': {
                'quality_score': 0.2,
                'quality_level': 'LOW',
                'recommendations': ['Find more relevant sources', 'Refine search terms', 'Check for alternative phrasings of the claim']
            }
        }
    
    def _create_error_verdict(self, claim: str, error_message: str) -> Dict[str, Any]:
        """Create verdict when an error occurs."""
        return {
            'claim': claim,
            'verdict': 'UNVERIFIED',
            'confidence': 0.0,
            'justification': f'Analysis failed due to technical error: {error_message}',
            'supporting_sources': [],
            'contradicting_sources': [],
            'key_evidence': [],
            'limitations': 'Technical error prevented proper analysis',
            'analysis_metadata': {
                'sources_analyzed': 0,
                'avg_credibility': 0,
                'avg_relevance': 0,
                'has_high_credibility_sources': False,
                'has_multiple_sources': False,
                'analysis_timestamp': self._get_current_timestamp()
            },
            'source_details': [],
            'quality_assessment': {
                'quality_score': 0.0,
                'quality_level': 'ERROR',
                'recommendations': ['Retry the analysis', 'Check system configuration']
            },
            'error': error_message
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def batch_analyze_claims(self, claims_with_summaries: Dict[str, List[Dict[str, Any]]], language: str = "en") -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple claims with their respective summaries.
        
        Args:
            claims_with_summaries: Dictionary mapping claims to their summaries
            language: ISO language code (en, ar, fr, es, etc.)
            
        Returns:
            Dictionary mapping claims to their verdicts
        """
        verdicts = {}
        
        logger.info(f"Analyzing {len(claims_with_summaries)} claims in language: {language}")
        
        for claim, summaries in claims_with_summaries.items():
            logger.info(f"Analyzing claim: {claim[:50]}...")
            verdict = await self.analyze_claim(claim, summaries, language)
            verdicts[claim] = verdict
        
        return verdicts
    
    async def generate_overall_verdict(self, input_text: str, claims: List[str], 
                                 individual_verdicts: Dict[str, Dict[str, Any]], 
                                 language: str = "en") -> Dict[str, Any]:
        """
        Generate an overall verdict for the entire input text based on individual claim verdicts.
        
        Args:
            input_text: The original input text
            claims: List of extracted claims
            individual_verdicts: Dictionary of individual claim verdicts
            language: ISO language code (en, ar, fr, es, etc.)
            
        Returns:
            Overall verdict for the entire text
        """
        try:
            # Import language utilities here to avoid circular imports
            from language_utils import translate_verdict, get_language_templates
            
            if not individual_verdicts:
                # Provide language-specific response for no verdict
                if language == "ar":
                    justification = "لم يكن من الممكن تحليل أي ادعاءات قابلة للتحقق بسبب نقص الأدلة."
                elif language == "fr":
                    justification = "Aucune allégation vérifiable n'a pu être analysée en raison du manque de preuves."
                elif language == "es":
                    justification = "No se pudieron analizar afirmaciones verificables debido a la falta de evidencia."
                else:
                    justification = "No verifiable claims could be analyzed due to lack of evidence."
                
                return {
                    'overall_verdict': 'UNVERIFIED',
                    'confidence': 0.1,
                    'justification': justification,
                    'claim_breakdown': {},
                    'methodology': 'Unable to verify claims due to insufficient evidence'
                }
            
            # Prepare overall verdict prompt
            verdict_summary = self._prepare_overall_verdict_prompt(input_text, claims, individual_verdicts)
            
            # Generate overall verdict using AI
            overall_verdict = await self._generate_overall_ai_verdict(verdict_summary)
            
            return overall_verdict
            
        except Exception as e:
            logger.error(f"Error generating overall verdict: {e}")
            return {
                'overall_verdict': 'ERROR',
                'confidence': 0.0,
                'justification': f'Error analyzing text: {str(e)}',
                'claim_breakdown': {},
                'methodology': 'Analysis failed due to technical error'
            }
    
    def _prepare_overall_verdict_prompt(self, input_text: str, claims: List[str], individual_verdicts: Dict[str, Dict[str, Any]]) -> str:
        """Prepare the prompt for overall verdict generation."""
        claim_summaries = []
        
        for i, claim in enumerate(claims, 1):
            verdict_data = individual_verdicts.get(claim, {})
            verdict = verdict_data.get('verdict', 'UNVERIFIED')
            confidence = verdict_data.get('confidence', 0.0)
            justification = verdict_data.get('justification', 'No justification available')
            
            claim_summaries.append(f"""
Claim {i}: {claim}
Verdict: {verdict}
Confidence: {confidence:.2f}
Justification: {justification}
""")
        
        return f"""
Original Text:
{input_text}

Individual Claim Analysis:
{''.join(claim_summaries)}

Total Claims Analyzed: {len(claims)}
"""
    
    async def _generate_overall_ai_verdict(self, verdict_summary: str) -> Dict[str, Any]:
        """Generate overall verdict using AI analysis."""
        try:
            prompt = f"""
You are a fact-checking expert providing an overall assessment of a text based on individual claim analysis.

Review the original text and the individual claim verdicts below, then provide an overall assessment.

{verdict_summary}

Provide a comprehensive overall verdict for the ENTIRE TEXT considering:
1. How many claims were verified as TRUE vs FALSE vs UNVERIFIED
2. The credibility and sources that support or contradict claims
3. The overall reliability of the text based on evidence found
4. Whether the text contains misinformation that could mislead readers

IMPORTANT: Your overall_reasoning must include:
- Summary of how many claims were found true/false/unverified with specific counts
- Specific mention of key authoritative sources that supported or contradicted claims (with URLs)
- Names of credible organizations/institutions that provided evidence (e.g., WHO, Reuters, medical journals)
- Overall assessment of the text's reliability based on source quality and evidence strength
- Any important context about the analysis quality, source diversity, or limitations

Overall verdict options:
- SUPPORTED: Most claims are verified as true by credible sources
- NOT_SUPPORTED: Most significant claims are false or contradicted by evidence  
- MIXED: Mix of true and false claims, with evidence on both sides
- INSUFFICIENT_INFO: Cannot determine truthfulness due to lack of evidence

Respond with JSON:
{{
    "overall_verdict": "verdict_here",
    "overall_confidence": 0.0-1.0,
    "overall_reasoning": "Comprehensive 4-6 sentence explanation that MUST include: (1) Summary of individual claim results with exact counts (e.g., '2 of 3 claims were false'), (2) Specific authoritative sources that were consulted with URLs or organization names (e.g., 'WHO official statements', 'Reuters fact-checking', 'medical journals'), (3) What evidence these sources provided, (4) Overall assessment of text reliability and potential for misinformation, (5) Quality assessment of the analysis and any limitations in the evidence base.",
    "claim_breakdown": {{
        "total_claims": number,
        "true_claims": number,
        "false_claims": number,
        "partly_true_claims": number, 
        "unverified_claims": number
    }},
    "key_sources_referenced": ["URLs of most important sources that influenced the verdict"],
    "methodology": "Brief description of how the analysis was conducted and evidence quality"
}}

EXCELLENT EXAMPLE of overall_reasoning: "Analysis of 3 claims found 2 definitively false and 1 unverified. The primary claim about microchips in COVID vaccines was thoroughly debunked by multiple authoritative medical sources including WHO official statements (who.int/fact-sheets), Reuters Health fact-checking (reuters.com/fact-check), and peer-reviewed research from medical journals. These sources consistently showed no electronic components exist in any approved vaccines. The second false claim about vaccine side effects was contradicted by clinical trial data from FDA and CDC surveillance systems. The third claim about vaccine development timeline could not be verified due to insufficient specific data. Overall, this text contains significant medical misinformation that contradicts established scientific consensus from the world's leading health authorities, making it potentially harmful and misleading to readers."

BAD EXAMPLE: "Majority of claims are false (2/3 false). Sources found contradictory evidence. Text appears unreliable."
"""
            
            await llm_rate_limiter.wait_if_needed()
            messages = [
                SystemMessage(content="You are a professional fact-checking expert providing comprehensive analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            
            # Parse the response
            try:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    overall_data = json.loads(json_match.group())
                else:
                    overall_data = json.loads(response.content)
                
                return overall_data
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse overall verdict JSON")
                return {
                    'overall_verdict': 'INSUFFICIENT_INFO',
                    'overall_confidence': 0.3,
                    'overall_reasoning': 'Could not generate comprehensive overall analysis due to parsing error.',
                    'claim_breakdown': {'total_claims': 0, 'true_claims': 0, 'false_claims': 0, 'partly_true_claims': 0, 'unverified_claims': 0},
                    'key_sources_referenced': [],
                    'methodology': 'Analysis incomplete due to technical issues'
                }
                
        except Exception as e:
            logger.error(f"Error generating overall AI verdict: {e}")
            return {
                'overall_verdict': 'INSUFFICIENT_INFO',
                'overall_confidence': 0.0,
                'overall_reasoning': f'Analysis failed due to error: {str(e)}',
                'claim_breakdown': {'total_claims': 0, 'true_claims': 0, 'false_claims': 0, 'partly_true_claims': 0, 'unverified_claims': 0},
                'key_sources_referenced': [],
                'methodology': 'Analysis failed due to technical error'
            }

    def _parse_fallback_verdict(self, response_text: str) -> Dict[str, Any]:
        """Parse verdict response when JSON parsing fails."""
        # Simple fallback parsing
        verdict = 'UNVERIFIED'
        confidence = 0.5
        justification = response_text
        
        # Try to extract verdict from text
        response_upper = response_text.upper()
        if 'MOSTLY_TRUE' in response_upper:
            verdict = 'MOSTLY_TRUE'
            confidence = 0.8
        elif 'MOSTLY_FALSE' in response_upper:
            verdict = 'MOSTLY_FALSE'
            confidence = 0.8
        elif 'PARTIALLY_TRUE' in response_upper:
            verdict = 'PARTIALLY_TRUE'
            confidence = 0.6
        elif 'FALSE' in response_upper and 'MOSTLY' not in response_upper:
            verdict = 'FALSE'
            confidence = 0.9
        elif 'TRUE' in response_upper:
            verdict = 'MOSTLY_TRUE'
            confidence = 0.7
        
        return {
            'overall_verdict': verdict,
            'confidence': confidence,
            'justification': justification[:500],
            'claim_breakdown': {},
            'methodology': 'Fallback text parsing'
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_verdict_analyzer():
        analyzer = VerdictAnalyzer()
        
        # Test summaries
        test_summaries = [
            {
                'source_url': 'https://example.com/study1',
                'source_title': 'COVID-19 Vaccine Efficacy Study',
                'summary': 'Clinical trials show 95% efficacy in preventing symptomatic COVID-19',
                'credibility_score': 0.9,
                'relevance_score': 0.95
            },
            {
                'source_url': 'https://example.com/news1',
                'source_title': 'Health Authority Confirms Vaccine Effectiveness',
                'summary': 'Health officials report high effectiveness of vaccines in real-world conditions',
                'credibility_score': 0.8,
                'relevance_score': 0.9
            }
        ]
        
        claim = "COVID-19 vaccines are highly effective"
        
        verdict = await analyzer.analyze_claim(claim, test_summaries)
        
        print(f"Verdict for: {claim}")
        print(f"Result: {verdict['verdict']}")
        print(f"Confidence: {verdict['confidence']:.2f}")
        print(f"Quality: {verdict['quality_assessment']['quality_level']}")
        print(f"Justification: {verdict['justification']}")
        
        # Test overall verdict
        claims = [s['summary'] for s in test_summaries]
        individual_verdicts = {claim: verdict}
        
        overall_verdict = await analyzer.generate_overall_verdict(" ".join(claims), claims, individual_verdicts)
        
        print(f"\nOverall Verdict: {overall_verdict['overall_verdict']}")
        print(f"Confidence: {overall_verdict['confidence']:.2f}")
        print(f"Justification: {overall_verdict['justification']}")
    
    # Run test
    asyncio.run(test_verdict_analyzer())
