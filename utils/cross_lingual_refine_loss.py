import torch
import torch.nn.functional as F
import numpy as np


def get_word_embedding(word, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn):
    """Get word embedding for a word from either English or Chinese vocabulary"""
    if word in vocab_en:
        idx = vocab_en.index(word)
        vec = word_embeddings_en[idx]
    elif word in vocab_cn:
        idx = vocab_cn.index(word)
        vec = word_embeddings_cn[idx]
    else:
        # This shouldn't happen after validation, but just in case
        raise ValueError(f"Word '{word}' not found in vocabularies")

    # Ensure we always return a torch.Tensor on CPU with float dtype
    if isinstance(vec, np.ndarray):
        vec = torch.from_numpy(vec)
    elif not torch.is_tensor(vec):
        vec = torch.tensor(vec)

    return vec.detach().cpu().float()


def _pairwise_cosine_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    X: [n, d], Y: [m, d] (assumed float32/float64 on same device)
    returns D[i,j] = sqrt( max(0, 2 - 2*cos_sim(x_i,y_j)) )
    """
    Xn = F.normalize(X, dim=-1)
    Yn = F.normalize(Y, dim=-1)
    cos = Xn @ Yn.T                      # [n, m]
    # turn cosine similarity into Euclidean distance on the unit sphere
    d2 = torch.clamp(2.0 - 2.0 * cos, min=0.0)
    return torch.sqrt(d2 + 1e-12)


def _median_bandwidth(dist_matrix: torch.Tensor) -> torch.Tensor:
    """
    Median heuristic on pooled pairwise distances (flattened, excluding zeros).
    """
    v = dist_matrix.flatten()
    # remove zeros (same-point pairs), keep positive entries
    v = v[v > 0]
    if v.numel() == 0:
        return torch.tensor(0.5, device=dist_matrix.device, dtype=dist_matrix.dtype)
    return torch.median(v)


def _gaussian_kernel_from_dist(D: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # k(x,y) = exp( - d(x,y)^2 / (2 * sigma^2) )
    return torch.exp(-(D * D) / (2.0 * (sigma * sigma + 1e-12)))


def _weighted_mmd2(X: torch.Tensor, wX: torch.Tensor,
                   Y: torch.Tensor, wY: torch.Tensor,
                   sigma: float = None) -> torch.Tensor:
    """
    X: [n,d], wX: [n]  (sum ~ 1)
    Y: [m,d], wY: [m]  (sum ~ 1)
    Returns scalar MMD^2 with Gaussian kernel on cosine distance.
    """
    device = X.device
    dtype = X.dtype

    # distances
    DX = _pairwise_cosine_distance(X, X)   # [n,n]
    DY = _pairwise_cosine_distance(Y, Y)   # [m,m]
    DXY = _pairwise_cosine_distance(X, Y)  # [n,m]

    # single-bandwidth (median heuristic if None)
    if sigma is None:
        pooled_distances = torch.cat([DX.flatten(), DY.flatten(), DXY.flatten()])
        sig = _median_bandwidth(pooled_distances.reshape(-1, 1).squeeze())
    else:
        sig = torch.tensor(float(sigma), device=device, dtype=dtype)

    KX = _gaussian_kernel_from_dist(DX, sig)
    KY = _gaussian_kernel_from_dist(DY, sig)
    KXY = _gaussian_kernel_from_dist(DXY, sig)
    
    return (wX @ (KX @ wX)) + (wY @ (KY @ wY)) - 2.0 * (wX @ (KXY @ wY))


def get_embedding_matrix(words, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn):
    """Get embedding matrix for a list of words"""
    embeddings = []
    for word in words:
        try:
            emb = get_word_embedding(word, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn)
            embeddings.append(emb)
        except ValueError:
            continue
    
    if not embeddings:
        return None
    
    return torch.stack(embeddings)

def compute_cross_lingual_refine_loss(topic_probas_en, topic_probas_cn,
                                      topic_indices_en, topic_indices_cn,
                                      refined_topics, high_confidence_topics,
                                      vocab_en, vocab_cn,
                                      word_embeddings_en, word_embeddings_cn,
                                      sinkhorn_reg: float = 0.1):
    """
    Compute MMD² loss between original and refined topic distributions
    
    Args:
        topic_probas_en: Original English probabilities [num_topics, 15]
        topic_probas_cn: Original Chinese probabilities [num_topics, 15] 
        topic_indices_en: Original English word indices [num_topics, 15]
        topic_indices_cn: Original Chinese word indices [num_topics, 15]
        high_confidence_topics: Refined words and counts from refinement
        vocab_en: English vocabulary list
        vocab_cn: Chinese vocabulary list
        word_embeddings_en: English word embeddings tensor
        word_embeddings_cn: Chinese word embeddings tensor
        sinkhorn_reg: Deprecated parameter (kept for compatibility)
    
    Returns:
        MMD² distance between original and refined distributions
    """

    
    device = topic_probas_en.device
    num_topics = topic_probas_en.shape[0]
    total_mmd_distance = torch.zeros((), device=device, dtype=torch.float32)
    used_topics = 0
    
    for i in range(num_topics):
        if i >= len(high_confidence_topics):
            continue
            
        # 1. Get original words and probabilities
        orig_en_indices = topic_indices_en[i]
        orig_cn_indices = topic_indices_cn[i]
        orig_en_words = [vocab_en[idx.item()] for idx in orig_en_indices]
        orig_cn_words = [vocab_cn[idx.item()] for idx in orig_cn_indices]
        orig_en_probs = topic_probas_en[i]
        orig_cn_probs = topic_probas_cn[i]
        
        # 2. Get refined words and probabilities
        refined_en_words = high_confidence_topics[i]['high_confidence_words_en']
        refined_cn_words = high_confidence_topics[i]['high_confidence_words_cn']
        refined_en_probs = high_confidence_topics[i].get('word_probs_en', {})
        refined_cn_probs = high_confidence_topics[i].get('word_probs_cn', {})
        
        # Fallback to counts if probabilities not available
        if not refined_en_probs and not refined_cn_probs:
            refined_en_counts = high_confidence_topics[i].get('word_counts_en', {})
            refined_cn_counts = high_confidence_topics[i].get('word_counts_cn', {})
        else:
            refined_en_counts = refined_en_probs
            refined_cn_counts = refined_cn_probs
        
        # Skip if no refined words
        if not refined_en_words and not refined_cn_words:
            continue
        
        # 3. Build original distribution data - keep as tensors
        orig_words = orig_en_words + orig_cn_words
        orig_probs_combined = torch.cat([0.5 * orig_en_probs, 0.5 * orig_cn_probs])
        
        # 4. Build refined distribution data
        ref_words = refined_en_words + refined_cn_words
        ref_probs_list = []
        for word in refined_en_words:
            ref_probs_list.append(0.5 * float(refined_en_counts.get(word, 0)))
        for word in refined_cn_words:
            ref_probs_list.append(0.5 * float(refined_cn_counts.get(word, 0)))
        ref_probs_combined = torch.tensor(ref_probs_list, device=device, dtype=torch.float32)
        
        # Skip if empty distributions
        if len(orig_words) < 1 or len(ref_words) < 1:
            continue
            
        # 5. Get embeddings
        orig_embeddings = get_embedding_matrix(orig_words, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn)
        ref_embeddings = get_embedding_matrix(ref_words, vocab_en, vocab_cn, word_embeddings_en, word_embeddings_cn)
        
        if orig_embeddings is None or ref_embeddings is None:
            continue
            
        # 6. Prepare tensors
        X = orig_embeddings.to(device=device, dtype=torch.float32)
        Y = ref_embeddings.to(device=device, dtype=torch.float32)
        
        # Normalize probabilities
        orig_sum = torch.clamp(orig_probs_combined.sum(), min=1e-8)
        ref_sum = torch.clamp(ref_probs_combined.sum(), min=1e-8)
        
        wX = orig_probs_combined / orig_sum
        wY = ref_probs_combined / ref_sum
        
        # 7. Compute MMD²

        mmd2 = _weighted_mmd2(X, wX, Y, wY, sigma=None)
        total_mmd_distance = total_mmd_distance + mmd2
        used_topics += 1

    
    return total_mmd_distance / max(1, used_topics)