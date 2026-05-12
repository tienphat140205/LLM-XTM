import torch
import numpy as np
from collections import Counter, defaultdict
import re
import time
from typing import List, Dict, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_topic_task(topic_idx: int, worker_id: int, prompt: str, api_key: str, provider: str, model_name: str, base_url: str, original_words_en: str = "", original_words_cn: str = "") -> Dict:
    """
    Standalone helper function to run in a separate process.
    Configures the OpenAI client locally for this process.
    """
    import time
    
    # Log the topic being processed with its content
    print(f"Worker {worker_id} processing Topic {topic_idx}: EN={original_words_en[:50]}... CN={original_words_cn[:50]}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if provider == 'gemini':
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.7,
                        "max_output_tokens": 1024,
                    },
                )
                response_text = getattr(response, 'text', '')
            else:
                from openai import OpenAI
                client = OpenAI(base_url=base_url, api_key=api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.7,
                    max_tokens=1024,
                )
                response_text = response.choices[0].message.content
            
            # Simple parsing logic duplicated here to avoid dependency on class instance
            lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
            parsed_topic = None
            
            # We expect exactly one topic in the output
            i = 0
            while i < len(lines):
                m = re.match(r"^Topic\s+(\d+)\s*:\s*(.*)$", lines[i])
                if m:
                    tid = int(m.group(1))
                    theme = m.group(2).strip()
                    en_words = []
                    cn_words = []
                    if i + 1 < len(lines) and lines[i+1].startswith("EN:"):
                        en_line = lines[i+1][3:].strip()
                        if ' - ' in en_line:
                            en_words = [w.strip() for w in en_line.split(' - ') if w.strip()]
                        else:
                            en_words = [w.strip() for w in en_line.split(',') if w.strip()]
                    if i + 2 < len(lines) and lines[i+2].startswith("CN:"):
                        cn_line = lines[i+2][3:].strip()
                        if ' - ' in cn_line:
                            cn_words = [w.strip() for w in cn_line.split(' - ') if w.strip()]
                        else:
                            cn_words = [w.strip() for w in cn_line.split(',') if w.strip()]
                    
                    if en_words and cn_words:
                        parsed_topic = {
                            'topic_id': tid,
                            'topic_theme': theme,
                            'refined_words_en': en_words,
                            'refined_words_cn': cn_words
                        }
                        break # Found it
                    i += 3
                else:
                    i += 1
            
            if parsed_topic:
                # Ensure ID matches what we requested, or force it if it's the only one
                if parsed_topic['topic_id'] == topic_idx:
                    return parsed_topic
                else:
                    # If ID mismatch but valid content, trust the content and fix ID
                    parsed_topic['topic_id'] = topic_idx
                    return parsed_topic
            
            # If parsing failed but we got text, maybe retry?
            # print(f"Worker {worker_id}: Failed to parse response for topic {topic_idx}")
            
        except Exception as e:
            # print(f"Worker {worker_id}: Error refining topic {topic_idx}: {e}")
            time.sleep(1 + attempt)
            
    return None

class CrossLingualTopicRefiner:
    def __init__(self, api_keys: List[str], provider: str = "openai", model_name: str = "qwen/qwen3-coder-480b-a35b-instruct", base_url: str = "https://integrate.api.nvidia.com/v1"):
        """
        Initialize the cross-lingual topic refiner with OpenAI compatible API (NVIDIA)
        
        Args:
            api_keys: List of API keys
            model_name: Model name to use
        """
        self.api_keys = api_keys
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        
    def _get_model(self, index: int):
        """
        Get a model instance for a specific index (round-robin)
        """
        # Not used in parallel flow, but kept for compatibility
        return None

    
    def create_single_topic_refinement_prompt(self, target_topic_idx: int, topic_words_en: List[str], topic_words_cn: List[str]) -> str:
        """
        Create prompt for refining a SINGLE specific topic, while providing all other topics as context.
        
        Args:
            target_topic_idx: Index of the topic to refine
            topic_words_en: List of English topic word strings for ALL topics
            topic_words_cn: List of Chinese topic word strings for ALL topics
            
        Returns:
            Formatted prompt string
        """
        num_topics = len(topic_words_en)
        
        # Construct context of all topics
        all_topics_context = ""
        for k in range(num_topics):
            top_15_en = topic_words_en[k].split()
            top_15_cn = topic_words_cn[k].split()
            words_en_str = ", ".join(top_15_en)
            words_cn_str = ", ".join(top_15_cn)
            
            marker = " (TARGET TOPIC)" if k == target_topic_idx else ""
            all_topics_context += f"Topic {k}{marker}:\nEN: {words_en_str}\nCN: {words_cn_str}\n\n"

        prompt = f"""You are an expert in cross-lingual topic modeling. We have extracted {num_topics} topics from a bilingual corpus (English and Chinese).
Your task is to REFINE ONLY ONE specific topic (Topic {target_topic_idx}), ensuring it is coherent, distinct, and high-quality.

Here is the list of all {num_topics} topics for context (to avoid overlap):
{all_topics_context}

----------------------------------------------------------------
YOUR TASK: REFINE TOPIC {target_topic_idx}
----------------------------------------------------------------

Original Top Words for Topic {target_topic_idx}:
EN: {topic_words_en[target_topic_idx]}
CN: {topic_words_cn[target_topic_idx]}

Instructions:
1. Identify the core semantic theme of Topic {target_topic_idx}.
2. Select the best 20 single words for English and 20 single words for Chinese that represent this theme.
3. Remove noise (irrelevant words) and generic words.
4. Ensure DISTINCTIVENESS: Do not use words that clearly belong to other topics listed in the context.
5. Output format must be strict plain text.

Output Format:
Topic {target_topic_idx}: <Short Theme Description>
EN: word1 - word2 - ... - word20
CN: word1 - word2 - ... - word20

Rules:
- Use ONLY single words (no phrases).
- Exactly 20 words per language.
- Separated by " - ".
- Do not output anything else.
"""
        return prompt
    
    def _parse_plain_response(self, response_text: str, expected_num_topics: int) -> List[Dict]:
        """Parse plain-text Topic/EN/CN response into a list of topic dicts."""
        topics = []
        # Split by lines and iterate assembling blocks per topic
        lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
        i = 0
        while i < len(lines):
            # Expect: Topic k: theme
            m = re.match(r"^Topic\s+(\d+)\s*:\s*(.*)$", lines[i])
            if not m:
                i += 1
                continue
            topic_id = int(m.group(1))
            theme = m.group(2).strip()
            en_words = []
            cn_words = []
            if i + 1 < len(lines) and lines[i+1].startswith("EN:"):
                en_line = lines[i+1][3:].strip()
                # Support both hyphen-separated and comma-separated
                if ' - ' in en_line:
                    en_words = [w.strip() for w in en_line.split(' - ') if w.strip()]
                else:
                    en_words = [w.strip() for w in en_line.split(',') if w.strip()]
            if i + 2 < len(lines) and lines[i+2].startswith("CN:"):
                cn_line = lines[i+2][3:].strip()
                if ' - ' in cn_line:
                    cn_words = [w.strip() for w in cn_line.split(' - ') if w.strip()]
                else:
                    cn_words = [w.strip() for w in cn_line.split(',') if w.strip()]
            if en_words and cn_words:
                topics.append({
                    'topic_id': topic_id,
                    'topic_theme': theme,
                    'refined_words_en': en_words,
                    'refined_words_cn': cn_words
                })
                i += 3
            else:
                i += 1
        # Basic validation
        topics = sorted(topics, key=lambda t: t['topic_id'])
        topics = [t for t in topics if 0 <= t['topic_id'] < expected_num_topics]
        return topics if topics else None

    def _check_word_counts(self, topics: List[Dict], expected_count: int = 20) -> bool:
        """Return True if each topic has at least expected_count EN and CN words."""
        if not topics:
            return False
        ok = True
        for t in topics:
            en = t.get('refined_words_en', [])
            cn = t.get('refined_words_cn', [])
            if len(en) < expected_count or len(cn) < expected_count:
                tid = t.get('topic_id')
                print(f"Format check failed for topic {tid}: EN={len(en)}, CN={len(cn)} (expected at least {expected_count}).")
                ok = False
        return ok

    def call_gemini_api_single_topic(self, prompt: str, topic_idx: int, worker_id: int, max_retries: int = 3) -> Dict:
        """
        Call Gemini API for a single topic with retry logic.
        This method is kept for compatibility or sequential testing but is superseded by process_single_topic_task.
        """
        # This method is no longer used in the parallel flow but kept for reference
        pass
    
    def self_consistent_refinement(self,
                                   topic_words_en: List[str],
                                   topic_words_cn: List[str],
                                   R: int = 3) -> List[Dict]:
        """
        Self-Consistent Refinement: Ask Gemini R times to refine topics in PARALLEL using ProcessPoolExecutor
        """
        num_topics = len(topic_words_en)
        
        # Initialize topic data structures
        refined_topics = []
        for k in range(num_topics):
            refined_topics.append({
                'topic_id': k,
                'word_counts_en': defaultdict(int),
                'word_counts_cn': defaultdict(int),
                'refinement_rounds_completed': 0
            })
        
        print(f"Starting PARALLEL refinement for {num_topics} topics with {R} rounds using ProcessPoolExecutor...")
        
        # Max workers based on user request (50) or number of keys
        # User asked for 50 processes.
        max_workers = 50 
        
        for r in range(R):
            print(f"--- Round {r+1}/{R} ---")
            # Use ProcessPoolExecutor for true parallelism and API key isolation
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_topic = {}
                for k in range(num_topics):
                    prompt = self.create_single_topic_refinement_prompt(k, topic_words_en, topic_words_cn)
                    worker_id = k
                    # Pick key for this task
                    api_key = self.api_keys[worker_id % len(self.api_keys)]
                    
                    future = executor.submit(
                        process_single_topic_task, 
                        topic_idx=k, 
                        worker_id=worker_id, 
                        prompt=prompt, 
                        api_key=api_key, 
                        provider=self.provider,
                        model_name=self.model_name,
                        base_url=self.base_url,
                        original_words_en=topic_words_en[k],
                        original_words_cn=topic_words_cn[k]
                    )
                    future_to_topic[future] = k

                for future in as_completed(future_to_topic):
                    k = future_to_topic[future]
                    try:
                        result = future.result()
                        if result:
                            topic_data = refined_topics[k]
                            self._update_word_counts(topic_data['word_counts_en'], result.get('refined_words_en', []))
                            self._update_word_counts(topic_data['word_counts_cn'], result.get('refined_words_cn', []))
                            topic_data['refinement_rounds_completed'] += 1
                            # print(f"  Topic {k} refined.")
                        else:
                            print(f"  Topic {k} failed to refine.")
                    except Exception as exc:
                        print(f"  Topic {k} generated an exception: {exc}")

        return refined_topics

    # _process_single_topic method removed as we use standalone function
    
    def _is_valid_topic_result(self, topic_result: Dict, num_topics: int) -> bool:
        """Validate topic result structure"""
        return (isinstance(topic_result, dict) and 
                'topic_id' in topic_result and 
                topic_result['topic_id'] < num_topics)
    
    def _update_word_counts(self, word_counts: defaultdict, words: List[str]) -> None:
        """Update word counts efficiently"""
        for word in words:
            word_counts[word] += 1
    
    
    def get_high_confidence_words(self, 
                                  refined_topics: List[Dict], 
                                  top_k: int = 15) -> List[Dict]:
        """
        Get top-k words by count across refinement rounds
        
        Args:
            refined_topics: List of refined topic dictionaries with word counts
            top_k: Number of top words to return per topic (default 15)
            
        Returns:
            List with top words and their raw counts
        """
        results = []
        
        for topic_data in refined_topics:
            en_word_counts = topic_data.get('word_counts_en', {})
            cn_word_counts = topic_data.get('word_counts_cn', {})
            
            # Get top_k words by count (highest first)
            en_top_items = sorted(en_word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            cn_top_items = sorted(cn_word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            results.append({
                'topic_id': topic_data['topic_id'],
                'high_confidence_words_en': [word for word, count in en_top_items],
                'high_confidence_words_cn': [word for word, count in cn_top_items],
                'word_counts_en': {word: count for word, count in en_top_items},
                'word_counts_cn': {word: count for word, count in cn_top_items}
            })
        
        return results
    
    def calculate_confidence_word_probabilities(self, high_confidence_topics: List[Dict]) -> List[Dict]:
        """
        Calculate probabilities for high confidence words from their counts
        
        Args:
            high_confidence_topics: Topics with high confidence words and their counts
            
        Returns:
            Topics with added probability distributions for high confidence words
        """
        topics_with_probs = []
        
        for topic_data in high_confidence_topics:
            topic_with_probs = topic_data.copy()
            
            # Calculate probabilities for English high confidence words
            en_counts = topic_data.get('word_counts_en', {})
            en_total = sum(en_counts.values())
            if en_total > 0:
                topic_with_probs['word_probs_en'] = {
                    word: count / en_total for word, count in en_counts.items()
                }
            else:
                topic_with_probs['word_probs_en'] = {}
            
            # Calculate probabilities for Chinese high confidence words  
            cn_counts = topic_data.get('word_counts_cn', {})
            cn_total = sum(cn_counts.values())
            if cn_total > 0:
                topic_with_probs['word_probs_cn'] = {
                    word: count / cn_total for word, count in cn_counts.items()
                }
            else:
                topic_with_probs['word_probs_cn'] = {}
                
            topics_with_probs.append(topic_with_probs)
            
        return topics_with_probs

    def validate_words_against_vocab(self, refined_topics: List[Dict], vocab_en: List[str], vocab_cn: List[str]) -> List[Dict]:
        """
        Validate refined words against actual vocabulary files and discard invalid words
        
        Args:
            refined_topics: List of refined topic dictionaries
            vocab_en: English vocabulary list from TextData
            vocab_cn: Chinese vocabulary list from TextData
            
        Returns:
            List of validated refined topics with only vocab-valid words
        """
        vocab_en_set = set(vocab_en)
        vocab_cn_set = set(vocab_cn)
        
        validated_topics = []
        
        for topic_data in refined_topics:
            topic_id = topic_data['topic_id']
            
            # Get refined word counts
            word_counts_en = topic_data.get('word_counts_en', {})
            word_counts_cn = topic_data.get('word_counts_cn', {})
            
            # Filter words that exist in vocabulary
            valid_word_counts_en = {word: count for word, count in word_counts_en.items() 
                                   if word in vocab_en_set}
            valid_word_counts_cn = {word: count for word, count in word_counts_cn.items() 
                                   if word in vocab_cn_set}
            
            # Count discarded words for logging
            discarded_en = len(word_counts_en) - len(valid_word_counts_en)
            discarded_cn = len(word_counts_cn) - len(valid_word_counts_cn)
            
            # Create validated topic data
            validated_topic = topic_data.copy()
            validated_topic['word_counts_en'] = valid_word_counts_en
            validated_topic['word_counts_cn'] = valid_word_counts_cn
            validated_topic['discarded_words_en'] = discarded_en
            validated_topic['discarded_words_cn'] = discarded_cn
            
            validated_topics.append(validated_topic)
        
        return validated_topics


def refine_cross_lingual_topics(topic_words_en: List[str],
                                topic_words_cn: List[str], 
                                topic_probas_en: torch.Tensor,
                                topic_probas_cn: torch.Tensor,
                                vocab_en: List[str],
                                vocab_cn: List[str],
                                api_key: Union[str, List[str]],
                                provider: str = "openai",
                                model_name: str = "qwen/qwen3-coder-480b-a35b-instruct",
                                base_url: str = "https://integrate.api.nvidia.com/v1",
                                R: int = 3) -> Tuple[List[Dict], List[Dict]]:
    """
    Main function to perform cross-lingual topic refinement
    """
    # Handle single or multiple API keys
    if isinstance(api_key, str):
        api_keys = [k.strip() for k in api_key.split(',') if k.strip()]
    else:
        api_keys = api_key

    if not api_keys:
        raise ValueError("No LLM API key provided for topic refinement")
        
    refiner = CrossLingualTopicRefiner(api_keys, provider=provider, model_name=model_name, base_url=base_url)
    
    print(f"Starting batch refinement for {len(topic_words_en)} topics with {R} rounds each...")
    
    # Process all topics together in each refinement round
    refined_topics = refiner.self_consistent_refinement(topic_words_en, topic_words_cn, R=R)
    
    # Validate refined words against actual vocabulary
    print("Validating refined words against vocabulary...")
    validated_topics = refiner.validate_words_against_vocab(refined_topics, vocab_en, vocab_cn)
    
    # Extract high-confidence words based on frequency from validated topics
    high_confidence_topics = refiner.get_high_confidence_words(
        validated_topics, top_k=15
    )
    
    # Calculate probabilities for high confidence words
    print("Calculating probabilities for high confidence words...")
    high_confidence_topics_with_probs = refiner.calculate_confidence_word_probabilities(high_confidence_topics)
    
    return validated_topics, high_confidence_topics_with_probs
