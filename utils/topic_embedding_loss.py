import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


def create_topic_embeddings(high_confidence_topics: List[Dict], 
                           encoder_model=None,
                           model_name: str = "BAAI/bge-m3") -> torch.Tensor:
    """
    Create topic embeddings by concatenating English and Chinese top words for each topic
    and encoding them using an external encoder (default: BGE-M3)
    
    Args:
        high_confidence_topics: Topics with high confidence words
        encoder_model: Pre-loaded encoder model (optional)
        model_name: Name of the encoder model to use (default: BAAI/bge-m3)
        
    Returns:
        topic_embeddings: Tensor of shape [num_topics, embedding_dim]
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    # Load encoder model if not provided
    if encoder_model is None:
        print(f"Loading encoder model: {model_name}")
        encoder_model = SentenceTransformer(model_name)
    
    num_topics = len(high_confidence_topics)
    topic_texts = []
    
    # Create concatenated topic strings
    for i, topic_data in enumerate(high_confidence_topics):
        # Get high confidence words
        en_words = topic_data.get('high_confidence_words_en', [])
        cn_words = topic_data.get('high_confidence_words_cn', [])
        
        # Concatenate English and Chinese words
        # Format: "english_word1 english_word2 ... chinese_word1 chinese_word2 ..."
        topic_text = " ".join(en_words + cn_words)
        topic_texts.append(topic_text)
        
        if i < 3:  # Print first 3 topics for debugging
            print(f"Topic {i} text: {topic_text}")
    
    # Encode all topic texts at once
    print(f"Encoding {num_topics} topic texts using {model_name}...")
    topic_embeddings = encoder_model.encode(topic_texts, convert_to_tensor=True)
    
    print(f"Created topic embeddings: {topic_embeddings.shape}")
    return topic_embeddings


def compute_topic_similarity_loss(doc_embeddings_en: torch.Tensor,
                                 doc_embeddings_cn: torch.Tensor,
                                 topic_embeddings: torch.Tensor,
                                 theta_en: torch.Tensor,
                                 theta_cn: torch.Tensor,
                                 temperature: float = 0.1) -> torch.Tensor:
    """
    Compute KL divergence loss between encoder theta and softmax similarity with topic embeddings
    
    Mathematical Framework:
    1. Compute cosine similarity between document embeddings and topic embeddings
    2. Apply softmax with temperature to get topic probability distribution
    3. Compute KL divergence between encoder theta and similarity-based distribution
    
    Args:
        doc_embeddings_en: English document embeddings [batch_size, embedding_dim]
        doc_embeddings_cn: Chinese document embeddings [batch_size, embedding_dim]  
        topic_embeddings: Topic embeddings from refined words [num_topics, embedding_dim]
        theta_en: English topic distributions from encoder [batch_size, num_topics]
        theta_cn: Chinese topic distributions from encoder [batch_size, num_topics]
        temperature: Temperature for softmax scaling (default: 0.1)
        
    Returns:
        kl_loss: KL divergence loss between encoder theta and similarity distributions
    """
    device = theta_en.device
    topic_embeddings = topic_embeddings.to(device)
    
    # Normalize embeddings for cosine similarity
    doc_embeddings_en_norm = F.normalize(doc_embeddings_en, p=2, dim=1)
    doc_embeddings_cn_norm = F.normalize(doc_embeddings_cn, p=2, dim=1) 
    topic_embeddings_norm = F.normalize(topic_embeddings, p=2, dim=1)
    
    # Compute cosine similarity: [batch_size, num_topics]
    sim_en = torch.matmul(doc_embeddings_en_norm, topic_embeddings_norm.T)
    sim_cn = torch.matmul(doc_embeddings_cn_norm, topic_embeddings_norm.T)
    
    # Apply temperature scaling and softmax to get probability distributions
    sim_probs_en = F.softmax(sim_en / temperature, dim=1)
    sim_probs_cn = F.softmax(sim_cn / temperature, dim=1)
    
    # Compute KL divergence: KL(theta || sim_probs)
    # KL(P || Q) = sum(P * log(P / Q))
    kl_loss_en = F.kl_div(
        F.log_softmax(theta_en, dim=1), 
        sim_probs_en, 
        reduction='batchmean'
    )
    
    kl_loss_cn = F.kl_div(
        F.log_softmax(theta_cn, dim=1), 
        sim_probs_cn, 
        reduction='batchmean'
    )
    
    # Combine losses from both languages
    kl_loss = (kl_loss_en + kl_loss_cn) / 2.0
    
    return kl_loss


