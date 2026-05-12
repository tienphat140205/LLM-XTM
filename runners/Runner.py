import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.InfoCTM import InfoCTM
from models.NMTM import NMTM
from models.XTRA import XTRA
from utils.cross_lingual_refinement import refine_cross_lingual_topics, CrossLingualTopicRefiner
from utils.cross_lingual_refine_loss import compute_cross_lingual_refine_loss
from utils.topic_embedding_loss import create_topic_embeddings, compute_topic_similarity_loss


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = self._create_model(args)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.device}" if args.device is not None else "cuda:0")
            self.model = self.model.to(self.device)

    def _create_model(self, args):
        """Create model instance based on args.model"""
        if args.model == 'InfoCTM':
            model = InfoCTM(
                trans_e2c=args.trans_matrix_en,
                pretrain_word_embeddings_en=args.pretrained_WE_en,
                pretrain_word_embeddings_cn=args.pretrained_WE_cn,
                vocab_size_en=args.vocab_size_en,
                vocab_size_cn=args.vocab_size_cn,
                num_topics=args.num_topic,
                en_units=args.en1_units,
                dropout=args.dropout,
                temperature=args.temperature,
                pos_threshold=args.pos_threshold,
                weight_MI=args.weight_MI  
            )
        elif args.model == 'NMTM':
            model = NMTM(
                Map_en2cn=args.Map_en2cn,
                Map_cn2en=args.Map_cn2en,
                vocab_size_en=args.vocab_size_en,
                vocab_size_cn=args.vocab_size_cn,
                num_topics=args.num_topic,
                en_units=args.ens_unit,
                dropout=args.dropout,
                lam=args.lam
            )
        elif args.model == 'XTRA':
            model = XTRA(args)
        else:
            raise ValueError(f"Unsupported model: {args.model}. Supported models: InfoCTM, NMTM, XTRA")
        
        # Add required attributes to the model for compatibility with existing Runner code
        model.vocab_en = args.vocab_en
        model.vocab_cn = args.vocab_cn
        model.word_embeddings_en = args.word_embeddings_en
        model.word_embeddings_cn = args.word_embeddings_cn
        
        return model

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.learning_rate
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma)
        else:
            raise NotImplementedError(self.args.lr_scheduler)

        return lr_scheduler

    def train(self, data_loader):

        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()

        # Add these parameters
        max_grad_norm = 1.0  # Adjust this value as needed
 
        if 'lr_scheduler' in self.args:
            lr_scheduler = self.make_lr_scheduler(optimizer)

        for epoch in range(1, self.args.epochs + 1):
            # Phase 2: Check if we should extract topic words
            print(self.args.warmStep)
            if epoch >= self.args.warmStep:
                # Extract topic words from current beta
                beta_en, beta_cn = self.model.get_beta()
                topic_words_en, topic_words_cn = self.get_topic_words(beta_en, beta_cn, topk_refine=15)
                print(f"Phase 2 - Epoch {epoch}: Extracted topic words")
                print(f"English topic words: {len(topic_words_en)} topics")
                print(f"Chinese topic words: {len(topic_words_cn)} topics")
                
                # Step 1: Extract top-k words using torch.topk for each topic
                topk = 15  # Number of top words to keep
                
                # For English topics - store indices for later fresh computation
                top_values_en, top_indices_en = torch.topk(beta_en, topk, dim=1)
                
                # For Chinese topics - store indices for later fresh computation  
                top_values_cn, top_indices_cn = torch.topk(beta_cn, topk, dim=1)
                
                # Store vocabulary indices for loss computation
                self.topic_indices_en = top_indices_en
                self.topic_indices_cn = top_indices_cn
                
                print(f"Extracted top {topk} word indices for each topic")
                print(f"English topic indices shape: {top_indices_en.shape}")
                print(f"Chinese topic indices shape: {top_indices_cn.shape}")
                
                # Cross-lingual topic refinement using Gemini API (multiple times during training)
                refined_topics, high_confidence_topics = None, None
                
                # Determine if we should refine this epoch
                should_refine = False
                refine_frequency = getattr(self.args, 'refine_frequency', 5)  # Default every 50 epochs
                
                # First refinement after warmStep
                if epoch == self.args.warmStep:
                    should_refine = True
                    print("🔄 First refinement after warmStep...")
                    
                    # Reset learning rate for Phase 2
                    phase2_lr = self.args.learning_rate * 0.3  # 30% of original LR
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = phase2_lr
                    print(f"📈 Reset learning rate to {phase2_lr:.6f} for Phase 2 (refinement phase)")
                # Subsequent refinements at regular intervals
                elif (epoch > self.args.warmStep and 
                      (epoch - self.args.warmStep) % refine_frequency == 0):
                    should_refine = True
                    print(f"🔄 Periodic refinement at epoch {epoch} (every {refine_frequency} epochs)...")
                
                if should_refine and hasattr(self.args, 'gemini_api_key') and self.args.gemini_api_key:
                    print("Starting cross-lingual topic refinement...")
                    
                    # Compute probabilities for refinement (detached for API call)
                    topic_probas_en_for_refinement = torch.div(top_values_en, top_values_en.sum(dim=1, keepdim=True)).detach()
                    topic_probas_cn_for_refinement = torch.div(top_values_cn, top_values_cn.sum(dim=1, keepdim=True)).detach()

                    refined_topics, high_confidence_topics = refine_cross_lingual_topics(
                        topic_words_en=topic_words_en,
                        topic_words_cn=topic_words_cn,
                        topic_probas_en=topic_probas_en_for_refinement,
                        topic_probas_cn=topic_probas_cn_for_refinement,
                        vocab_en=self.model.vocab_en,
                        vocab_cn=self.model.vocab_cn,
                        api_key=self.args.gemini_api_key,
                        provider=getattr(self.args, 'llm_provider', 'openai'),
                        model_name=getattr(self.args, 'llm_model', 'qwen/qwen3-coder-480b-a35b-instruct'),
                        base_url=getattr(self.args, 'llm_base_url', 'https://integrate.api.nvidia.com/v1'),
                        R=getattr(self.args, 'refinement_rounds', 5)
                    )

                    print(f"Refined {len(refined_topics)} topics using cross-lingual refinement")
                    
                    # Print summary of refined topics
                    for i, (refined, high_conf) in enumerate(zip(refined_topics, high_confidence_topics)):
                        total_words = len(high_conf['high_confidence_words_en']) + len(high_conf['high_confidence_words_cn'])
                        print(f"Topic {i}: {total_words} high-confidence words ({len(high_conf['high_confidence_words_en'])} EN, {len(high_conf['high_confidence_words_cn'])} CN)")
                        sample_words = high_conf['high_confidence_words_en'][:3] + high_conf['high_confidence_words_cn'][:3]
                        print(f"  Sample words: {', '.join(sample_words[:5])}...")
                    
                    # Store refined topics for evaluation
                    self.refined_topics = refined_topics
                    self.high_confidence_topics = high_confidence_topics
                    
                    # Create topic embeddings from refined words using BGE-M3
                    self.topic_embeddings = create_topic_embeddings(
                        high_confidence_topics=high_confidence_topics,
                        encoder_model=None,  # Will load BGE-M3 automatically
                        model_name="BAAI/bge-m3"
                    )
                    print(f"Created topic embeddings with shape: {self.topic_embeddings.shape}")
                    
                elif should_refine and (not hasattr(self.args, 'gemini_api_key') or not self.args.gemini_api_key):
                    print("No Gemini API key provided, skipping cross-lingual refinement")

                # Ensure refined topics persist after warmStep for loss computation
                refined_topics = getattr(self, 'refined_topics', None)
                high_confidence_topics = getattr(self, 'high_confidence_topics', None)

            sum_loss = 0.

            loss_rst_dict = defaultdict(float)
            print(epoch)


            self.model.train()
            for batch_data in data_loader:
                batch_bow_en = batch_data['bow_en']
                batch_bow_cn = batch_data['bow_cn']
                document_info = {
                'doc_embedding_en': batch_data.get('doc_embedding_en'),
                'doc_embedding_cn': batch_data.get('doc_embedding_cn')
                }
                cluster_info = {
                'cluster_en': batch_data.get('cluster_en'),
                'cluster_cn': batch_data.get('cluster_cn')
                }
                # Get theta from model for topic similarity loss
                theta_en, _, _ = self.model.get_theta(batch_bow_en, lang='en')
                theta_cn, _, _ = self.model.get_theta(batch_bow_cn, lang='cn')
                
                # Forward pass
                if isinstance(self.model, XTRA):
                    rst_dict = self.model(batch_bow_en, batch_bow_cn, document_info=document_info, cluster_info=cluster_info)
                else:
                    rst_dict = self.model(batch_bow_en, batch_bow_cn)
                batch_loss = rst_dict['loss']
                
                # Add refinement loss if we have refined topics
                if (epoch >= self.args.warmStep and
                    refined_topics is not None and
                    high_confidence_topics is not None and
                    hasattr(self.args, 'refine_weight') and
                    self.args.refine_weight > 0):

                    # Compute fresh probabilities from current beta for this batch
                    dev = self.device
                    current_beta_en, current_beta_cn = self.model.get_beta()
                    
                    # Extract probabilities for the same top-k indices identified during refinement
                    current_topic_probas_en = torch.gather(current_beta_en, 1, self.topic_indices_en)
                    current_topic_probas_cn = torch.gather(current_beta_cn, 1, self.topic_indices_cn)
                    
                    # Renormalize to ensure they sum to 1
                    current_topic_probas_en = torch.div(current_topic_probas_en, current_topic_probas_en.sum(dim=1, keepdim=True))
                    current_topic_probas_cn = torch.div(current_topic_probas_cn, current_topic_probas_cn.sum(dim=1, keepdim=True))
                    
                    # Ensure consistent device/dtype for OT inputs
                    topic_probas_en_ot = current_topic_probas_en.to(dev, dtype=torch.float32)
                    topic_probas_cn_ot = current_topic_probas_cn.to(dev, dtype=torch.float32)
                    
                    refine_loss = compute_cross_lingual_refine_loss(
                        topic_probas_en=topic_probas_en_ot,
                        topic_probas_cn=topic_probas_cn_ot,
                        topic_indices_en=self.topic_indices_en,
                        topic_indices_cn=self.topic_indices_cn,
                        refined_topics=refined_topics,
                        high_confidence_topics=high_confidence_topics,
                        vocab_en=self.model.vocab_en,
                        vocab_cn=self.model.vocab_cn,
                        word_embeddings_en=self.model.word_embeddings_en,
                        word_embeddings_cn=self.model.word_embeddings_cn
                    )
                    
                    # Always apply refinement loss with non-zero weight
                    weighted_refine_loss = self.args.refine_weight * refine_loss
                    batch_loss = batch_loss + weighted_refine_loss
                    rst_dict['weighted_refine_loss'] = weighted_refine_loss.detach()

                # Add topic embedding similarity loss from Phase 2 onwards
                if (epoch >= self.args.warmStep and 
                    hasattr(self, 'topic_embeddings') and
                    hasattr(self.args, 'topic_sim_weight') and
                    self.args.topic_sim_weight > 0):
                    
                    # Compute topic similarity loss
                    topic_sim_loss = compute_topic_similarity_loss(
                        doc_embeddings_en=document_info['doc_embedding_en'],
                        doc_embeddings_cn=document_info['doc_embedding_cn'],
                        topic_embeddings=self.topic_embeddings,
                        theta_en=theta_en,
                        theta_cn=theta_cn,
                        temperature=getattr(self.args, 'temperature', 0.1)
                    )
                    
                    # Add weighted topic similarity loss
                    weighted_topic_sim_loss = self.args.topic_sim_weight * topic_sim_loss
                    batch_loss = batch_loss + weighted_topic_sim_loss
                    rst_dict['weighted_topic_sim_loss'] = weighted_topic_sim_loss.detach()

                for key in rst_dict:
                    if 'loss' in key:
                        val = rst_dict[key]
                        loss_rst_dict[key] += float(val) if torch.is_tensor(val) else float(val)

                optimizer.zero_grad()
                batch_loss.backward()
                # Add gradient clipping before optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                sum_loss += batch_loss.item() * len(batch_bow_en)

            if 'lr_scheduler' in self.args:
                lr_scheduler.step()

            sum_loss /= data_size

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / num_batch :.3f}'

            print(output_log)


        beta_en, beta_cn = self.model.get_beta()
        beta_en = beta_en.detach().cpu().numpy()
        beta_cn = beta_cn.detach().cpu().numpy()
        return beta_en, beta_cn


    def get_theta(self, bow, lang):
        """Get topic distribution from BOW"""
        theta_list = list()
        data_size = bow.shape[0]
        all_idx = torch.split(torch.arange(data_size,), self.args.batch_size)
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = bow[idx]
                # Đảm bảo batch_bow ở đúng device
                if isinstance(batch_bow, np.ndarray):
                    batch_bow = torch.tensor(batch_bow, dtype=torch.float)
                if hasattr(batch_bow, 'device') and batch_bow.device != self.device:
                    batch_bow = batch_bow.to(self.device)
                theta = self.model.get_theta(batch_bow, lang=lang)
                theta_list.extend(theta.detach().cpu().numpy().tolist())

        return np.asarray(theta_list)

    # Keep existing code and modify Runner.py:
    def test(self, dataset):
        theta_en = self.get_theta(dataset.bow_en, lang='en')
        theta_cn = self.get_theta(dataset.bow_cn, lang='cn')
        return theta_en, theta_cn

    def get_topic_words(self, beta_en, beta_cn, topk_refine=15, topk_loss=15):
        """Extract top words for each topic from beta matrices

        Args:
            beta_en: English beta matrix
            beta_cn: Chinese beta matrix
            topk_refine: Number of words for refinement vocabulary (default: 15)
            topk_loss: Number of words for loss computation (default: 15)

        Returns:
            topic_words_en, topic_words_cn: Lists of topic word strings
        """
        # Convert CUDA tensors to numpy arrays if necessary
        if torch.is_tensor(beta_en):
            beta_en = beta_en.detach().cpu().numpy()
        if torch.is_tensor(beta_cn):
            beta_cn = beta_cn.detach().cpu().numpy()

        topic_words_en = []
        topic_words_cn = []

        # Get vocabularies from the model
        vocab_en = self.model.vocab_en
        vocab_cn = self.model.vocab_cn

        # Extract top words for each topic in English
        for i, topic_dist in enumerate(beta_en):
            top_word_indices = np.argsort(topic_dist)[:-(topk_refine + 1):-1]
            topic_words = np.array(vocab_en)[top_word_indices]
            topic_str = ' '.join(topic_words)
            topic_words_en.append(topic_str)

        # Extract top words for each topic in Chinese
        for i, topic_dist in enumerate(beta_cn):
            top_word_indices = np.argsort(topic_dist)[:-(topk_refine + 1):-1]
            topic_words = np.array(vocab_cn)[top_word_indices]
            topic_str = ' '.join(topic_words)
            topic_words_cn.append(topic_str)

        return topic_words_en, topic_words_cn
