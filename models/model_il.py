import torch.nn as nn
import torch
import torch.nn.functional as F

from .conch import tokenize, get_tokenizer


def merge_parameter(cfg, param):
    if cfg['opt_name'] == 'adam':
        merged_param = torch.cat([param[i] for i in range(cfg['pool_size'])], dim=0)
    else:
        raise NotImplementedError('Invalid optimizer')

    return merged_param


class TunableVLearner(nn.Module):
    """
    Tunable Vector
    Adapted from TaskRes: https://github.com/geekyutao/TaskRes
    """
    def __init__(self, cfg, tunable_v):
        """
        :param cfg: the dict of configuration parameters.
        :param tunable_v: the trainable parameter, which is predefined in manager/manager.py.
        """
        super().__init__()

        self.cfg = cfg
        self.alpha = cfg['alpha']
        self.tunable_v = tunable_v # Trainable parameters

    def forward(self, class_ensemble_feature):
        """
        :param class_ensemble_feature: Tensor(num_cls, C).
        :return: enhanced class feature: Tensor(num_cls, C).
        """
        return class_ensemble_feature + self.alpha * torch.cat([self.tunable_v[i] for i in range(len(self.cfg["seen_task_dict"]))], dim=0) # [num_cls, C]


class CONCHPromptEncoder(nn.Module):
    """
    CONCH (CONCH custom API at github.com/mahmoodlab/CONCH)
    """
    def __init__(self, coca_model):
        super().__init__()
        # load text model and text projection
        coca_text_model = coca_model.text

        # Config of clip text model
        self.pad_id = coca_text_model.pad_id
        assert self.pad_id == 0, "Assume pad_id = 0 in CONCH to encode prompts as expected."
        self.heads = coca_text_model.heads

        # Embedding
        self.positional_embedding = coca_text_model.positional_embedding

        self.attn_mask = coca_text_model.attn_mask
        # Transformer
        self.transformer = coca_text_model.transformer
        self.ln_final = coca_text_model.ln_final

        self.cls_emb = coca_text_model.cls_emb

        # Text projection
        self.text_projection = coca_text_model.text_projection

        # Additional attributes used for downstream calls
        self.token_embedding = coca_text_model.token_embedding
        self.text_config = {
            'max_num_tokens': 127,
            'embedding_dim': self.token_embedding.embedding_dim,
            'embedding_dtype': self.token_embedding.weight.dtype
        }

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, prompts_embedding, prompts_pseudo_tokens):
        """
        Adapted from CONCH API:
            https://github.com/mahmoodlab/CONCH/blob/main/conch/open_clip_custom/transformer.py#L418

        input:
            prompts_embedding: the embedding of ranking prompts, with shape [n_batch, length_ctx, dim_embedding].
                Note that this input differs from the text_encoder input of HuggingFace CLIP.
            prompts_pseudo_tokens: the pseudo-tokens of ranking prompts, with shape [n_batch, length_ctx].
        """
        cast_dtype = self.transformer.get_cast_dtype()
        device = prompts_embedding.device
        seq_len = prompts_embedding.shape[1] # max_length - 1 = 128 - 1
        x = prompts_embedding.to(cast_dtype)  # [batch_size, n_ctx, d_model]

        prompts_pseudo_tokens = prompts_pseudo_tokens.to(device)
        attn_mask = self.attn_mask.to(device)
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(prompts_pseudo_tokens, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), prompts_pseudo_tokens.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        return pooled


class AlignMIL(nn.Module):
    """
    AlignMIL-NS: Aligning Prototypes and Updating Null Spaces for Continual WSI Learning (MICCAI 2026).
    Prototype-based MIL with learnable prototype vectors (prompt) and Prototype-Class Feature Alignment (PCFA).
    The Null Space Class Feature Update is performed in manager/manager.py.
    """
    def __init__(self, cfg, base_model, device, key, prompt, tunable_v, current_ensemble_classes, train_key_frequency):

        super().__init__()
        self.cfg = cfg
        self.dtype = base_model.dtype
        self.logit_scale = base_model.logit_scale
        self.device = device

        """
        Trainable parameters:
            - Vision branch: key-prompt pairs in Prototype Pool (key, prompt)
            - Language branch: Tunable Vector (tunable_v)
        """
        self.key = key
        self.prompt = prompt
        self.tunable_v_learner = TunableVLearner(cfg, tunable_v) # Tunable Vector
        if cfg['base_model_arch'] == 'CONCH':
            self.text_encoder = CONCHPromptEncoder(base_model)
        else:
            raise NotImplementedError("Please specify a valid architecture.")
        self.class_ensemble_feature = self._get_ensemble_feature(base_model, current_ensemble_classes) # [num_cls, C]

        if cfg['task_num'] > 1:
            self.penalty_table = self._get_penalty_table(train_key_frequency)
        else:
            self.penalty_table = None

    def _get_penalty_table(self, train_key_frequency):
        """
        train_key_frequency: {'dataset_name': torch.zeros(self.cfg['pool_size'])}
        """
        penalty_table = None
        train_key_frequency_v_list = list(train_key_frequency.values())
        n_seen_task = min(len(self.cfg["seen_task_dict"]), self.cfg['task_num'] - 1)
        for i in range(n_seen_task):
            one_task_frequency = train_key_frequency_v_list[i][-1]
            total_count = torch.sum(one_task_frequency)
            frequency_table = one_task_frequency / total_count
            if penalty_table is None:
                penalty_table = frequency_table
            else:
                penalty_table += frequency_table
        penalty_table = penalty_table / n_seen_task  # Update to use n_seen_task instead of self.cfg['task_num'] - 1
        penalty_table = penalty_table.type(self.dtype).to(self.device)

        return penalty_table

    def _query_prototype_pool(self, x_list, mini_batch, eval):
        """
        Prototype Pool
        Adapted from L2P: https://github.com/JH-LEE-KR/l2p-pytorch

        :param x_list: [Tensor(B=1, N, C), ], list of multiple WSI samples.
        :param mini_batch: the mini-batch size.
        :param eval:
            True: evaluation mode.
            False: training mode.
        :return:
            matching indices: Tensor(MB, match_size).
            matching loss.
        """
        q_vec_list = []
        for i in range(mini_batch):
            x_list[i] = x_list[i].type(self.dtype)
            q_vec, _ = torch.max(x_list[i], dim=1)  # [B, N, C] -> [B, C]: B = 1
            q_vec_list.append(q_vec)
        q_vecs = torch.cat(q_vec_list, dim=0) # [MB, C]
        q_vecs = q_vecs / q_vecs.norm(dim=-1, keepdim=True)

        merged_key = merge_parameter(self.cfg, self.key) # [pool_size, C]
        merged_key = merged_key / merged_key.norm(dim=-1, keepdim=True)
        cos_sim = q_vecs @ merged_key.t()  # Cosine similarity [MB, pool_size]
        if not eval and self.cfg['task_num'] > 1:
            cos_dist = 1 - cos_sim
            punished_cos_dist = cos_dist * self.penalty_table
            _, indices = punished_cos_dist.topk(k=self.cfg['match_size'], dim=1, largest=False) # [MB, match_size]
        else:
            _, indices = cos_sim.topk(k=self.cfg['match_size'], dim=1, largest=True) # [MB, match_size]
        key_id, id_counts = torch.unique(indices, return_counts=True, sorted=True)
        _, major_idx = torch.topk(id_counts, k=self.cfg['match_size'])
        indices = key_id[major_idx]  # minibatch-wise indices [match_size, ]
        indices = indices.expand(mini_batch, -1) # [MB, match_size]

        # matching loss
        if eval:
            matching_loss = None
        else:
            matched_key = merged_key[indices]
            matched_key = matched_key / matched_key.norm(dim=-1, keepdim=True)
            q_vecs = q_vecs.unsqueeze(1).repeat(1, self.cfg['match_size'], 1)
            matching_loss = 1 - ((q_vecs * matched_key) / (mini_batch * self.cfg['match_size'])).sum()

        return indices, matching_loss

    def _get_ensemble_feature(self, base_model, current_ensemble_classes):

        def get_embedding(base_model, device, classes):
            prompts = [name + '.' for name in classes]
            if self.cfg['base_model_arch'] == 'CONCH':
                tokenized_prompts = tokenize(get_tokenizer(), prompts)[:, :-1]
                with torch.no_grad():
                    embedding = base_model.text.token_embedding(tokenized_prompts.to(device)).type(base_model.dtype)
            else:
                raise NotImplementedError("Please specify a valid architecture.")

            return tokenized_prompts, embedding
        
        def get_feature(class_feature_matrix, count):
            count_t = []
            t = 0
            for i in range(len(count)):
                t += count[i]
                count_t.append(t)

            index = 0
            feature_list = []
            for i in range(len(count)):
                feature = torch.mean(class_feature_matrix[index:count_t[i]], dim=0, keepdim=True)
                feature_list.append(feature)
                index = count_t[i]

            return torch.cat(feature_list, dim=0)

        count = current_ensemble_classes['count']
        tokenized_prompts, embedding = get_embedding(base_model, self.device, classes=current_ensemble_classes['ensemble_classes'])
        with torch.no_grad():
            class_feature_matrix = self.text_encoder(embedding, tokenized_prompts)
        class_ensemble_feature = get_feature(class_feature_matrix, count)
        return class_ensemble_feature

    def get_prototype_features(self, prototype_idx):
        """
        Get the prototype features.
        :param prototype_idx: Tensor(num_proto, ).
        :return: prototype features: Tensor(num_proto, C).
        """
        prototype_features = merge_parameter(self.cfg, self.prompt) # [pool_size, C]
        prototype_features = prototype_features[prototype_idx] # [num_proto, C]
        prototype_features = prototype_features / prototype_features.norm(dim=-1, keepdim=True)
        return prototype_features
    
    def compute_prototype_class_loss(self, dataset2prototype_idx, current_dataset_name, class_indices):
        """
        Prototype-Class Feature Alignment (PCFA) loss, Eq. 5.
        :param dataset2prototype_idx: {'dataset_name': [idx1, idx2, ...]}, matched prototypes of each seen task.
        :param current_dataset_name: name of the current task.
        :param class_indices: {'dataset_name': [idx1, idx2, ...]}, class indices of each seen task.
        """
        enhanced_class_feature = self.tunable_v_learner(self.class_ensemble_feature)
        enhanced_class_feature = enhanced_class_feature / enhanced_class_feature.norm(dim=-1, keepdim=True)

        # prototypes of all seen tasks and their task labels
        all_prototypes = []
        proto_task_labels = []
        task_name_to_id = {name: i for i, name in enumerate(dataset2prototype_idx.keys())}
        for task_id, prototype_idx in enumerate(dataset2prototype_idx.values()):
            with torch.no_grad():
                prototypes = self.get_prototype_features(prototype_idx)
            all_prototypes.append(prototypes)
            proto_task_labels.extend([task_id] * prototypes.shape[0])
        all_prototypes = torch.cat(all_prototypes, dim=0) # [N_p, C]
        proto_task_labels = torch.tensor(proto_task_labels, dtype=torch.long, device=self.device)

        # class features of the current task
        current_task_id = task_name_to_id[current_dataset_name]
        current_features = enhanced_class_feature[class_indices[current_dataset_name]] # [N_f, C]

        # positives: prototypes of the current task; negatives: prototypes of previous tasks
        sim_matrix = torch.matmul(all_prototypes, current_features.T) # [N_p, N_f]
        denominator_per_feature = torch.sum(torch.exp(self.logit_scale * sim_matrix), dim=0) # [N_f]
        positive_mask = (proto_task_labels == current_task_id).unsqueeze(1).expand(-1, current_features.shape[0])
        log_prob = self.logit_scale * sim_matrix - torch.log(denominator_per_feature.unsqueeze(0))
        pc_loss = -log_prob[positive_mask].mean()
        return pc_loss

    def _get_bag_feature(self, x_list, indices, mini_batch):
        prototype_features = self.get_prototype_features(indices) # [match_size, C]
        # repeat to match the mini_batch size [mini_batch, match_size, C]

        bag_feature_list = []
        for i, x in enumerate(x_list):
            x = x.squeeze()
            x_norm = x / x.norm(dim=-1, keepdim=True)
            scaled_cos_sim_matrix = self.cfg['csm_logit_scale'] * x_norm @ prototype_features[i].t()
            weighted_aggregation_matrix = torch.softmax(scaled_cos_sim_matrix, dim=0)
            bag_feature = torch.mean(weighted_aggregation_matrix.t() @ x, dim=0, keepdim=True)
            bag_feature_list.append(bag_feature)
        bag_feature = torch.cat(bag_feature_list, dim=0)
        bag_feature = bag_feature / bag_feature.norm(dim=-1, keepdim=True)
        bag_feature = bag_feature.unsqueeze(1) # [MB, 1, C]
        return bag_feature

    def _compute_class_sim_loss(self, enhanced_class_feature): # class similarity loss
        features = enhanced_class_feature[0]
        n_cls = features.shape[0]
        cos_sim = features @ features.permute(1, 0)
        cos_sim += 1
        class_sim_loss = cos_sim[~torch.eye(n_cls, dtype=torch.bool, device=self.device)].mean()

        return class_sim_loss

    def get_enhanced_class_feature(self):
        """
        Get the enhanced class feature.
        :return: enhanced class feature: Tensor(num_cls, C).
        """
        enhanced_class_feature = self.tunable_v_learner(self.class_ensemble_feature)
        enhanced_class_feature = enhanced_class_feature / enhanced_class_feature.norm(dim=-1, keepdim=True)
        return enhanced_class_feature


    def forward(self, x_list, eval=False, current_dataset=None, class_indices=None):
        """
        C: the feature dimension.
        num_cls: the number of classes seen so far.
        MB: the mini-batch size.
        :param x_list: [Tensor(B=1, N, C), ], list of multiple WSI samples.
        :param eval:
            True: evaluation mode.
            False: training mode.
        :param current_dataset: Optional; the current dataset index.
        :param class_indices: {'dataset_name': [idx1, idx2, ...]}
        :return:
            evaluate:
                classification logits: Tensor(MB, num_cls).
                matching indices: Tensor(MB, match_size).
            train:
                classification logits: Tensor(MB, num_cls).
                loss dict: the matching loss and class similarity loss.
                matching indices: Tensor(MB, match_size).
                feature dict: the image (bag-level) feature and text (enhanced class) feature.
        """
        mini_batch = len(x_list)

        # Vision Branch
        indices, matching_loss = self._query_prototype_pool(x_list, mini_batch, eval)
        bag_feature = self._get_bag_feature(x_list, indices, mini_batch)
        # Language Branch (CFE)
        enhanced_class_feature = self.tunable_v_learner(self.class_ensemble_feature)
        if current_dataset is not None and class_indices is not None:
            current_class_indices = class_indices[current_dataset]
            enhanced_class_feature = torch.stack([feature if i in current_class_indices else feature.detach() for i, feature in enumerate(enhanced_class_feature)], dim=0)
        enhanced_class_feature = enhanced_class_feature.unsqueeze(0).repeat(mini_batch, 1, 1)
        enhanced_class_feature = enhanced_class_feature / enhanced_class_feature.norm(dim=-1, keepdim=True)
        # compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (bag_feature * enhanced_class_feature).sum(-1)
        if eval:
            return logits.float(), indices
        else:
            # class similarity loss
            class_sim_loss = self._compute_class_sim_loss(enhanced_class_feature)
            loss_dict = {
                'matching_loss': matching_loss.float(),
                'class_sim_loss': class_sim_loss.float()
            }

            feature_dict = {
                'image': bag_feature.float().squeeze(1),
                'text': enhanced_class_feature.float()
            }

            return logits.float(), loss_dict, indices, feature_dict