"""
Language detection utilities for the fact-checking pipeline.
Enables the fact-checking system to identify and respond in the user's language.
"""

import re
import logging
from typing import Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Language information with prompt translation templates
LANGUAGE_INFO = {
    "en": {
        "name": "English",
        "claim_prompt": "Extract factual claims from this text that can be fact-checked:",
        "verdict_label": {"TRUE": "TRUE", "FALSE": "FALSE", "PARTLY_TRUE": "PARTLY TRUE", "UNVERIFIED": "UNVERIFIED"},
        "justification_prefix": "Justification:",
        "confidence_labels": {
            "high": "High confidence",
            "medium": "Medium confidence",
            "low": "Low confidence"
        },
        "summary_prefix": "Summary:",
        "sources_prefix": "Sources:",
        "overall_verdict_prefix": "Overall Analysis:"
    },
    "ar": {
        "name": "Arabic",
        "claim_prompt": "استخرج الادعاءات الواقعية من هذا النص التي يمكن التحقق منها:",
        "verdict_label": {"TRUE": "صحيح", "FALSE": "خاطئ", "PARTLY_TRUE": "صحيح جزئياً", "UNVERIFIED": "غير مؤكد"},
        "justification_prefix": "التبرير:",
        "confidence_labels": {
            "high": "ثقة عالية",
            "medium": "ثقة متوسطة",
            "low": "ثقة منخفضة"
        },
        "summary_prefix": "الملخص:",
        "sources_prefix": "المصادر:",
        "overall_verdict_prefix": "التحليل الشامل:",
        "justification_templates": {
            "no_evidence": "لم يتم العثور على معلومات ذات صلة للتحقق من هذا الادعاء. هذا لا يعني أن الادعاء خاطئ، فقط أنه لا يمكن العثور على مصادر موثوقة لتأكيده أو دحضه.",
            "insufficient_evidence": "الأدلة المتاحة غير كافية للوصول إلى نتيجة قاطعة حول هذا الادعاء.",
            "mixed_evidence": "الأدلة مختلطة، حيث تدعم بعض المصادر الادعاء بينما تناقضه مصادر أخرى."
        },
        "error_messages": {
            "analysis_error": "حدث خطأ في التحليل",
            "no_claims": "لم يتم استخراج ادعاءات محددة للتحقق منها",
            "processing_failed": "فشل في المعالجة"
        }
    },
    "fr": {
        "name": "French",
        "claim_prompt": "Extrayez les affirmations factuelles de ce texte qui peuvent être vérifiées:",
        "verdict_label": {"TRUE": "VRAI", "FALSE": "FAUX", "PARTLY_TRUE": "PARTIELLEMENT VRAI", "UNVERIFIED": "NON VÉRIFIÉ"},
        "justification_prefix": "Justification:",
        "confidence_labels": {
            "high": "Confiance élevée",
            "medium": "Confiance moyenne",
            "low": "Confiance faible"
        },
        "summary_prefix": "Résumé:",
        "sources_prefix": "Sources:",
        "overall_verdict_prefix": "Analyse globale:",
        "justification_templates": {
            "no_evidence": "Aucune information pertinente n'a été trouvée pour vérifier cette affirmation. Cela ne signifie pas que l'affirmation est fausse, seulement qu'aucune source fiable n'a pu être trouvée pour la confirmer ou la réfuter.",
            "insufficient_evidence": "Les preuves disponibles sont insuffisantes pour parvenir à une conclusion définitive sur cette affirmation.",
            "mixed_evidence": "Les preuves sont mitigées, certaines sources soutenant l'affirmation tandis que d'autres la contredisent."
        },
        "error_messages": {
            "analysis_error": "Erreur d'analyse survenue",
            "no_claims": "Aucune allégation spécifique n'a pu être extraite pour vérification",
            "processing_failed": "Échec du traitement"
        }
    },
    "es": {
        "name": "Spanish",
        "claim_prompt": "Extrae las afirmaciones factuales de este texto que pueden ser verificadas:",
        "verdict_label": {"TRUE": "VERDADERO", "FALSE": "FALSO", "PARTLY_TRUE": "PARCIALMENTE VERDADERO", "UNVERIFIED": "NO VERIFICADO"},
        "justification_prefix": "Justificación:",
        "confidence_labels": {
            "high": "Alta confianza",
            "medium": "Confianza media",
            "low": "Baja confianza"
        },
        "summary_prefix": "Resumen:",
        "sources_prefix": "Fuentes:",
        "overall_verdict_prefix": "Análisis general:",
        "justification_templates": {
            "no_evidence": "No se encontró información relevante para verificar esta afirmación. Esto no significa que la afirmación sea falsa, solo que no se pudieron encontrar fuentes confiables para confirmarla o refutarla.",
            "insufficient_evidence": "La evidencia disponible es insuficiente para llegar a una conclusión definitiva sobre esta afirmación.",
            "mixed_evidence": "La evidencia es mixta, con algunas fuentes apoyando la afirmación mientras otras la contradicen."
        },
        "error_messages": {
            "analysis_error": "Ocurrió un error de análisis",
            "no_claims": "No se pudieron extraer afirmaciones específicas para verificación",
            "processing_failed": "Falló el procesamiento"
        }
    }
}

# Default to English if language detection fails
DEFAULT_LANGUAGE = "en"

def detect_language(text: str) -> str:
    """
    Detect the language of input text using character patterns and common words
    
    Args:
        text: Input text to analyze
        
    Returns:
        str: ISO language code (en, ar, fr, es, etc.)
    """
    if not text or len(text.strip()) < 10:
        logger.warning("Text too short for reliable language detection")
        return DEFAULT_LANGUAGE
    
    # Language scores
    scores = defaultdict(int)
    
    # Arabic script detection (scores heavily)
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    if arabic_chars > len(text) * 0.3:
        scores["ar"] += 100
    
    # Basic word-based detection for Latin scripts
    # French indicators
    french_words = ['le ', 'la ', 'les ', 'une ', 'des ', 'est ', 'sont ', 'et ', 'pour ', 'avec ', 'vous ', 'nous ']
    for word in french_words:
        if word in ' ' + text.lower() + ' ':
            scores["fr"] += 5
    
    # Spanish indicators
    spanish_words = ['el ', 'la ', 'los ', 'las ', 'una ', 'es ', 'son ', 'y ', 'para ', 'con ', 'de ', 'en ']
    for word in spanish_words:
        if word in ' ' + text.lower() + ' ':
            scores["es"] += 5
    
    # English indicators
    english_words = ['the ', 'and ', 'is ', 'are ', 'to ', 'of ', 'in ', 'for ', 'with ', 'that ', 'this ']
    for word in english_words:
        if word in ' ' + text.lower() + ' ':
            scores["en"] += 5
    
    # Consider length of text in each script
    latin_chars = sum(1 for c in text if ord(c) < 256 and c.isalpha())
    if latin_chars > len(text) * 0.5:
        # For Latin script text, consider punctuation patterns
        if "," in text and "?" in text:
            scores["fr"] += 2  # French often uses these
        if "¿" in text or "¡" in text:
            scores["es"] += 10  # Strong Spanish indicators
    
    # Find the most likely language
    if scores:
        detected_lang = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Detected language: {detected_lang} (scores: {dict(scores)})")
        return detected_lang
    
    # Default to English for Latin scripts or unknown
    if all(ord(char) < 256 for char in text if char.isalpha()):
        return "en"
    
    logger.warning("Could not reliably detect language, defaulting to English")
    return DEFAULT_LANGUAGE

def get_language_templates(language_code: str) -> Dict[str, Any]:
    """
    Get language-specific templates for the detected language
    
    Args:
        language_code: ISO language code (en, ar, fr, es)
        
    Returns:
        Dict of language templates
    """
    if language_code in LANGUAGE_INFO:
        return LANGUAGE_INFO[language_code]
    
    logger.warning(f"Language code {language_code} not supported, using English")
    return LANGUAGE_INFO[DEFAULT_LANGUAGE]

def translate_verdict(verdict: str, language_code: str) -> str:
    """
    Translate a verdict into the target language
    
    Args:
        verdict: Original verdict in uppercase English
        language_code: Target language code
        
    Returns:
        Translated verdict
    """
    if language_code == "en" or language_code not in LANGUAGE_INFO:
        return verdict
    
    # Get translation mapping
    verdict_map = LANGUAGE_INFO[language_code]["verdict_label"]
    
    # Map standard verdict keys
    norm_verdict = verdict.upper()
    if norm_verdict in verdict_map:
        return verdict_map[norm_verdict]
    
    # Handle variations
    if "SUPPORTED" in norm_verdict:
        return verdict_map["TRUE"]
    elif "NOT_SUPPORTED" in norm_verdict:
        return verdict_map["FALSE"]
    elif "MIXED" in norm_verdict or "PARTLY" in norm_verdict:
        return verdict_map["PARTLY_TRUE"]
    elif "INSUFFICIENT" in norm_verdict or "UNVERIFIED" in norm_verdict:
        return verdict_map["UNVERIFIED"]
    
    # Fallback
    logger.warning(f"Could not translate verdict: {verdict}")
    return verdict

def prepare_prompt(template: str, language_code: str, **kwargs) -> str:
    """
    Prepare a prompt in the target language with proper instructions
    
    Args:
        template: The base template with placeholders
        language_code: Target language code
        **kwargs: Values to substitute in the template
        
    Returns:
        Formatted prompt in target language
    """
    # Get language templates
    lang_info = get_language_templates(language_code)
    
    # Replace language-specific placeholders
    template = template.replace("{CLAIM_PROMPT}", lang_info["claim_prompt"])
    template = template.replace("{JUSTIFICATION_PREFIX}", lang_info["justification_prefix"])
    template = template.replace("{SUMMARY_PREFIX}", lang_info["summary_prefix"])
    template = template.replace("{SOURCES_PREFIX}", lang_info["sources_prefix"])
    template = template.replace("{OVERALL_VERDICT_PREFIX}", lang_info["overall_verdict_prefix"])
    
    # Fill in the regular placeholders
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder in template:
            template = template.replace(placeholder, str(value))
    
    return template
