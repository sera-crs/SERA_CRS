

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-8


def _standardize(scores: torch.Tensor) -> torch.Tensor:

    centered = scores - scores.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(EPS)
    return centered / scale


def _binary_entropy(probabilities: torch.Tensor) -> torch.Tensor:

    p = probabilities.clamp(EPS, 1.0 - EPS)
    return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.normalize(left.float(), dim=-1) @ F.normalize(
        right.float(), dim=-1
    ).transpose(-1, -2)


def _pool_observed_items(
    dialogue_rep: torch.Tensor,
    entity_vectors: torch.Tensor,
    polarity_mask: torch.Tensor,
) -> torch.Tensor:

    if entity_vectors.ndim != 3:
        raise ValueError("entity_vectors must be [B,E,H]")
    if polarity_mask.shape != entity_vectors.shape[:2]:
        raise ValueError("polarity_mask must match the first two entity dimensions")
    attention = torch.einsum("bh,beh->be", dialogue_rep, entity_vectors)
    attention = attention.masked_fill(~polarity_mask, -1e4)
    weights = F.softmax(attention, dim=-1) * polarity_mask.to(attention.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.einsum("be,beh->bh", weights, entity_vectors)


def _candidate_mask(
    candidate_ids: torch.Tensor,
    observed_mask: torch.Tensor,
    num_candidates: int,
) -> torch.Tensor:

    valid = observed_mask & candidate_ids.ge(0) & candidate_ids.lt(num_candidates)
    result = torch.zeros(
        candidate_ids.shape[0],
        num_candidates,
        dtype=torch.float,
        device=candidate_ids.device,
    )
    result.scatter_add_(
        1,
        candidate_ids.clamp(0, num_candidates - 1),
        valid.to(result.dtype),
    )
    return result.clamp_max(1.0)


@dataclass
class SERAOutput:
    final_logits: torch.Tensor
    standardized_base_scores: torch.Tensor
    preference_scores: torch.Tensor
    evidence_scores: torch.Tensor
    positive_preference_logits: torch.Tensor
    negative_preference_logits: torch.Tensor
    positive_preferences: torch.Tensor
    negative_preferences: torch.Tensor
    preference_representation: torch.Tensor
    retrieved_evidence_representation: torch.Tensor
    recommendation_summary: torch.Tensor
    generation_condition: torch.Tensor
    gate_probability: torch.Tensor
    effective_gate: torch.Tensor
    gate_features: torch.Tensor
    preference_uncertainty: torch.Tensor
    preference_contrast: torch.Tensor
    ranking_confidence: torch.Tensor
    preference_coverage: torch.Tensor
    retrieval_logits: torch.Tensor
    retrieval_pool_mask: torch.Tensor
    positive_evidence_mask: torch.Tensor
    retrieved_scene_ids: torch.Tensor
    retrieved_scene_weights: torch.Tensor
    current_positive_items: torch.Tensor
    current_negative_items: torch.Tensor

    hard_logits: torch.Tensor
    scene_logits: torch.Tensor
    tag_logits: torch.Tensor
    tag_probabilities: torch.Tensor
    negative_tag_probabilities: torch.Tensor
    tag_group_weights: torch.Tensor
    scene_gate: torch.Tensor
    scene_reliability: torch.Tensor
    entity_preference_logits: Optional[torch.Tensor]
    predicted_positive_mask: torch.Tensor


class StructuredPreferenceModel(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        num_attributes: int,
        negative_penalty: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_size = hidden_size * 3
        self.positive_head = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_attributes),
        )
        self.negative_head = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_attributes),
        )
        self.negative_penalty = float(negative_penalty)

    def forward(
        self,
        dialogue_rep: torch.Tensor,
        entity_vectors: torch.Tensor,
        positive_entity_mask: torch.Tensor,
        negative_entity_mask: torch.Tensor,
        item_attributes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_evidence = _pool_observed_items(
            dialogue_rep, entity_vectors, positive_entity_mask
        )
        negative_evidence = _pool_observed_items(
            dialogue_rep, entity_vectors, negative_entity_mask
        )
        joint = torch.cat(
            [dialogue_rep, positive_evidence, negative_evidence], dim=-1
        )
        positive_logits = self.positive_head(joint)
        negative_logits = self.negative_head(joint)
        positive = torch.sigmoid(positive_logits)
        negative = torch.sigmoid(negative_logits)
        attributes = item_attributes.to(positive.dtype)
        positive_score = positive @ attributes.transpose(0, 1)
        positive_score = positive_score / positive.sum(dim=-1, keepdim=True).clamp_min(EPS)
        negative_score = negative @ attributes.transpose(0, 1)
        negative_score = negative_score / negative.sum(dim=-1, keepdim=True).clamp_min(EPS)
        score = positive_score - self.negative_penalty * negative_score
        return positive_logits, negative_logits, positive, negative, score


class EvidenceSufficiencyGate(nn.Module):


    FEATURE_NAMES = (
        "preference_uncertainty",
        "one_minus_preference_contrast",
        "one_minus_ranking_confidence",
        "one_minus_preference_coverage",
    )

    def __init__(
        self,
        group_ids: torch.Tensor,
        group_is_exclusive: Optional[torch.Tensor] = None,
        coverage_top_k: int = 10,
        preference_threshold: float = 0.0,
        coverage_smoothness: float = 0.1,
        gate_threshold: float = 0.5,
        hidden_size: int = 32,
    ):
        super().__init__()
        group_ids = torch.as_tensor(group_ids, dtype=torch.long)
        if group_ids.ndim != 1:
            raise ValueError("group_ids must be [D]")
        unique = torch.unique(group_ids, sorted=True)
        if not torch.equal(unique, torch.arange(unique.numel())):
            raise ValueError("group_ids must be contiguous and start at zero")
        if group_is_exclusive is None:
            group_is_exclusive = torch.zeros(unique.numel(), dtype=torch.bool)
        group_is_exclusive = torch.as_tensor(group_is_exclusive, dtype=torch.bool)
        if group_is_exclusive.shape != (unique.numel(),):
            raise ValueError("group_is_exclusive must contain one value per group")
        self.register_buffer("group_ids", group_ids)
        self.register_buffer("group_is_exclusive", group_is_exclusive)
        self.num_groups = int(unique.numel())
        self.coverage_top_k = int(coverage_top_k)
        self.preference_threshold = float(preference_threshold)
        self.coverage_smoothness = float(coverage_smoothness)
        self.gate_threshold = float(gate_threshold)
        self.network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def _contrast_one_polarity(self, probabilities: torch.Tensor) -> torch.Tensor:
        values = []
        for group in range(self.num_groups):
            current = probabilities[:, self.group_ids.eq(group)]
            if bool(self.group_is_exclusive[group]):
                top = torch.topk(current, k=min(2, current.shape[-1]), dim=-1).values
                contrast = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else 0.0)
            else:

                contrast = 1.0 - _binary_entropy(current).mean(dim=-1)
            values.append(contrast)
        return torch.stack(values, dim=-1).mean(dim=-1)

    def diagnostics(
        self,
        positive_preferences: torch.Tensor,
        negative_preferences: torch.Tensor,
        base_logits: torch.Tensor,
        preference_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uncertainty = 0.5 * (
            _binary_entropy(positive_preferences).mean(dim=-1)
            + _binary_entropy(negative_preferences).mean(dim=-1)
        )
        contrast = 0.5 * (
            self._contrast_one_polarity(positive_preferences)
            + self._contrast_one_polarity(negative_preferences)
        )
        ranking = F.softmax(base_logits, dim=-1)
        top = torch.topk(ranking, k=min(2, ranking.shape[-1]), dim=-1).values
        confidence = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else 0.0)
        k = min(self.coverage_top_k, base_logits.shape[-1])
        top_ids = torch.topk(base_logits, k=k, dim=-1).indices
        top_preference = torch.gather(preference_scores, 1, top_ids)
        coverage = torch.sigmoid(
            (top_preference - self.preference_threshold)
            / max(self.coverage_smoothness, EPS)
        ).mean(dim=-1)
        return uncertainty, contrast, confidence, coverage

    def forward(
        self,
        positive_preferences: torch.Tensor,
        negative_preferences: torch.Tensor,
        base_logits: torch.Tensor,
        preference_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uncertainty, contrast, confidence, coverage = self.diagnostics(
            positive_preferences,
            negative_preferences,
            base_logits,
            preference_scores,
        )
        features = torch.stack(
            [uncertainty, 1.0 - contrast, 1.0 - confidence, 1.0 - coverage],
            dim=-1,
        )
        probability = torch.sigmoid(self.network(features)).squeeze(-1)
        return probability, features, uncertainty, contrast, confidence, coverage


class HistoricalEvidenceMemory(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        scene_embeddings: torch.Tensor,
        scene_tag_profiles: torch.Tensor,
        scene_movie_incidence: torch.Tensor,
        scene_conversation_ids: torch.Tensor,
        top_k: int = 3,
        tag_threshold: float = 0.5,
        semantic_weight: float = 1.0,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        temperature: float = 0.1,
        scene_degree_power: float = 1.0,
        item_degree_power: float = 1.0,
    ):
        super().__init__()
        if scene_embeddings.ndim != 2:
            raise ValueError("scene_embeddings must be [M,Ds]")
        num_scenes = scene_embeddings.shape[0]
        if scene_tag_profiles.shape[0] != num_scenes:
            raise ValueError("scene_tag_profiles must share the scene dimension")
        if scene_conversation_ids.shape != (num_scenes,):
            raise ValueError("scene_conversation_ids must be [M]")
        incidence = scene_movie_incidence.float()
        if not incidence.is_sparse:
            incidence = incidence.to_sparse_coo()
        incidence = incidence.coalesce()
        if incidence.shape[0] != num_scenes:
            raise ValueError("scene_movie_incidence must share the scene dimension")
        self.register_buffer(
            "scene_embeddings", F.normalize(scene_embeddings.float(), dim=-1)
        )
        self.register_buffer(
            "scene_attributes", scene_tag_profiles.float().ge(tag_threshold).float()
        )
        self.register_buffer("scene_movie_incidence", incidence)
        self.register_buffer("scene_conversation_ids", scene_conversation_ids.long())
        self.query_projection = nn.Linear(
            hidden_size, scene_embeddings.shape[1], bias=False
        )
        self.top_k = int(top_k)
        self.semantic_weight = float(semantic_weight)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.temperature = float(temperature)
        self.scene_degree_power = float(scene_degree_power)
        self.item_degree_power = float(item_degree_power)

        rows, columns = incidence.indices()
        values = incidence.values()
        counts = torch.bincount(rows, minlength=num_scenes)
        max_items = int(counts.max().item())
        padded_ids = torch.zeros(num_scenes, max_items, dtype=torch.long)
        padded_values = torch.zeros(num_scenes, max_items, dtype=torch.float)
        starts = torch.cumsum(counts, dim=0) - counts
        offsets = torch.arange(rows.numel()) - torch.repeat_interleave(starts, counts)
        padded_ids[rows, offsets] = columns
        padded_values[rows, offsets] = values
        self.register_buffer("scene_movie_ids", padded_ids)
        self.register_buffer("scene_movie_values", padded_values)
        scene_degrees = torch.zeros(num_scenes, dtype=torch.float)
        scene_degrees.scatter_add_(0, rows, values)
        item_degrees = torch.zeros(incidence.shape[1], dtype=torch.float)
        item_degrees.scatter_add_(0, columns, values)
        self.register_buffer("scene_degrees", scene_degrees.clamp_min(1.0))
        self.register_buffer("item_degrees", item_degrees.clamp_min(1.0))

    def forward(
        self,
        dialogue_rep: torch.Tensor,
        positive_preferences: torch.Tensor,
        negative_preferences: torch.Tensor,
        current_positive_items: torch.Tensor,
        conversation_ids: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        semantic = _cosine(self.query_projection(dialogue_rep), self.scene_embeddings)
        positive_match = _cosine(positive_preferences, self.scene_attributes)
        negative_conflict = _cosine(negative_preferences, self.scene_attributes)
        retrieval_logits = (
            self.semantic_weight * semantic
            + self.positive_weight * positive_match
            - self.negative_weight * negative_conflict
        )

        eligible = torch.ones_like(retrieval_logits, dtype=torch.bool)

        if self.training and conversation_ids is not None:
            eligible = eligible & conversation_ids[:, None].ne(
                self.scene_conversation_ids[None, :]
            )
        overlap = torch.sparse.mm(
            self.scene_movie_incidence,
            current_positive_items.float().transpose(0, 1),
        ).transpose(0, 1).gt(0)
        anchor_pool = eligible & overlap
        has_anchor_pool = anchor_pool.any(dim=-1, keepdim=True)
        retrieval_pool = torch.where(has_anchor_pool, anchor_pool, eligible)
        if active_mask is not None:
            retrieval_pool = retrieval_pool & active_mask[:, None]

        masked_logits = retrieval_logits.masked_fill(
            ~retrieval_pool, torch.finfo(retrieval_logits.dtype).min
        )
        k = min(self.top_k, masked_logits.shape[-1])
        top_values, top_ids = torch.topk(masked_logits, k=k, dim=-1)
        top_valid = torch.gather(retrieval_pool, 1, top_ids)
        top_values = top_values.masked_fill(~top_valid, -1e4)
        weights = F.softmax(top_values / max(self.temperature, EPS), dim=-1)
        weights = weights * top_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)

        selected_ids = self.scene_movie_ids[top_ids]
        selected_values = self.scene_movie_values[top_ids]
        selected_values = selected_values / (
            self.scene_degrees[top_ids].unsqueeze(-1) + EPS
        ).pow(self.scene_degree_power)
        selected_values = selected_values / (
            self.item_degrees[selected_ids] + EPS
        ).pow(self.item_degree_power)
        contributions = weights.unsqueeze(-1) * selected_values
        evidence_scores = weights.new_zeros(
            weights.shape[0], self.scene_movie_incidence.shape[1]
        )
        evidence_scores.scatter_add_(
            1, selected_ids.flatten(1), contributions.flatten(1)
        )
        evidence_rep = torch.einsum(
            "bk,bkd->bd", weights, self.scene_embeddings[top_ids]
        )

        positive_evidence = torch.zeros_like(retrieval_pool)
        if target_ids is not None:
            target_in_scene = (
                self.scene_movie_ids.unsqueeze(0).eq(target_ids[:, None, None])
                & self.scene_movie_values.unsqueeze(0).gt(0)
            ).any(dim=-1)
            positive_evidence = retrieval_pool & overlap & target_in_scene
        reliability = weights.max(dim=-1).values
        return (
            evidence_scores,
            evidence_rep,
            retrieval_logits,
            retrieval_pool,
            positive_evidence,
            top_ids,
            weights,
        )


class EnhancedRecommender(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        item_tags: torch.Tensor,
        tag_group_ids: Optional[torch.Tensor] = None,
        group_is_exclusive: Optional[torch.Tensor] = None,
        scene_assets: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        negative_penalty: float = 1.0,
        scene_top_k: int = 3,
        scene_tag_threshold: float = 0.5,
        scene_semantic_weight: float = 1.0,
        scene_positive_weight: float = 1.0,
        scene_negative_weight: float = 1.0,
        scene_temperature: float = 0.1,
        scene_edge_degree_power: float = 1.0,
        scene_item_degree_power: float = 1.0,
        coverage_top_k: int = 10,
        preference_threshold: float = 0.0,
        coverage_smoothness: float = 0.1,
        gate_threshold: float = 0.5,
        hard_threshold: float = 0.8,
        learn_fusion_weights: bool = False,
        use_hard: bool = True,
        use_scene: bool = True,
        **legacy_kwargs,
    ):
        super().__init__()
        if item_tags.ndim != 2:
            raise ValueError("item_tags must be [N,D]")
        self.register_buffer("item_tags", item_tags.float())
        if tag_group_ids is None:
            tag_group_ids = torch.zeros(item_tags.shape[1], dtype=torch.long)
        self.register_buffer(
            "tag_group_ids", torch.as_tensor(tag_group_ids, dtype=torch.long)
        )
        self.preference_model = StructuredPreferenceModel(
            hidden_size,
            item_tags.shape[1],
            negative_penalty=negative_penalty,
        )
        self.gate = EvidenceSufficiencyGate(
            self.tag_group_ids,
            group_is_exclusive=group_is_exclusive,
            coverage_top_k=coverage_top_k,
            preference_threshold=preference_threshold,
            coverage_smoothness=coverage_smoothness,
            gate_threshold=gate_threshold,
        )
        self.scene_memory = None
        if scene_assets is not None:
            self.scene_memory = HistoricalEvidenceMemory(
                hidden_size=hidden_size,
                top_k=scene_top_k,
                tag_threshold=scene_tag_threshold,
                semantic_weight=scene_semantic_weight,
                positive_weight=scene_positive_weight,
                negative_weight=scene_negative_weight,
                temperature=scene_temperature,
                scene_degree_power=scene_edge_degree_power,
                item_degree_power=scene_item_degree_power,
                **scene_assets,
            )
            evidence_size = scene_assets["scene_embeddings"].shape[1]
        else:
            evidence_size = hidden_size
        self.preference_projection = nn.Linear(item_tags.shape[1] * 2, hidden_size)
        self.evidence_projection = nn.Linear(evidence_size, hidden_size)
        self.response_conditioner = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(float(alpha)).log(), requires_grad=learn_fusion_weights
        )
        self.log_beta = nn.Parameter(
            torch.tensor(float(beta)).log(), requires_grad=learn_fusion_weights
        )
        self.hard_threshold = float(hard_threshold)
        self.use_hard = bool(use_hard)
        self.use_scene = bool(use_scene)

    @property
    def tag_head(self):

        return self.preference_model

    def _empty_memory(
        self, base_logits: torch.Tensor, dialogue_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = base_logits.shape[0]
        empty_long = torch.empty(batch, 0, dtype=torch.long, device=base_logits.device)
        empty_float = torch.empty(batch, 0, dtype=base_logits.dtype, device=base_logits.device)
        return (
            torch.zeros_like(base_logits),
            torch.zeros_like(dialogue_rep),
            empty_float,
            torch.zeros(batch, 0, dtype=torch.bool, device=base_logits.device),
            torch.zeros(batch, 0, dtype=torch.bool, device=base_logits.device),
            empty_long,
            empty_float,
        )

    def forward(
        self,
        base_logits: torch.Tensor,
        dialogue_rep: torch.Tensor,
        entity_vectors: torch.Tensor,
        entity_mask: torch.Tensor,
        entity_candidate_ids: torch.Tensor,
        positive_entity_mask: Optional[torch.Tensor] = None,
        negative_entity_mask: Optional[torch.Tensor] = None,
        conversation_ids: Optional[torch.Tensor] = None,
        candidate_item_embeddings: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        hard_inference: bool = False,
        always_on_gate: bool = False,
    ) -> SERAOutput:
        if hard_inference and always_on_gate:
            raise ValueError("hard_inference and always_on_gate are mutually exclusive")
        if positive_entity_mask is None:
            positive_entity_mask = torch.zeros_like(entity_mask)
        if negative_entity_mask is None:
            negative_entity_mask = torch.zeros_like(entity_mask)
        positive_entity_mask = positive_entity_mask & entity_mask
        negative_entity_mask = negative_entity_mask & entity_mask
        num_candidates = self.item_tags.shape[0]
        current_positive = _candidate_mask(
            entity_candidate_ids, positive_entity_mask, num_candidates
        )
        current_negative = _candidate_mask(
            entity_candidate_ids, negative_entity_mask, num_candidates
        )
        (
            positive_logits,
            negative_logits,
            positive_preferences,
            negative_preferences,
            raw_preference_scores,
        ) = self.preference_model(
            dialogue_rep,
            entity_vectors,
            positive_entity_mask,
            negative_entity_mask,
            self.item_tags,
        )
        (
            gate_probability,
            gate_features,
            uncertainty,
            contrast,
            confidence,
            coverage,
        ) = self.gate(
            positive_preferences,
            negative_preferences,
            base_logits,
            raw_preference_scores,
        )
        if always_on_gate:
            active = torch.ones_like(gate_probability, dtype=torch.bool)
            effective_gate = torch.ones_like(gate_probability)
        elif hard_inference:
            group_confidence = []
            for group in range(self.gate.num_groups):
                mask = self.tag_group_ids.eq(group)
                group_confidence.append(
                    positive_preferences[:, mask].max(dim=-1).values
                )
            hard_confidence = torch.stack(group_confidence, dim=-1).mean(dim=-1)
            effective_gate = hard_confidence.lt(self.hard_threshold).to(
                gate_probability.dtype
            )
            active = effective_gate.bool()
        elif self.training:
            active = torch.ones_like(gate_probability, dtype=torch.bool)
            effective_gate = gate_probability
        else:
            active = gate_probability.ge(self.gate.gate_threshold)
            effective_gate = gate_probability * active.to(gate_probability.dtype)

        if self.scene_memory is None or not self.use_scene:
            memory = self._empty_memory(base_logits, dialogue_rep)
        else:
            memory = self.scene_memory(
                dialogue_rep,
                positive_preferences,
                negative_preferences,
                current_positive,
                conversation_ids=conversation_ids,
                active_mask=active,
                target_ids=target_ids,
            )
        (
            raw_evidence_scores,
            evidence_rep,
            retrieval_logits,
            retrieval_pool,
            positive_evidence,
            scene_ids,
            scene_weights,
        ) = memory

        standardized_base = _standardize(base_logits)
        preference_scores = _standardize(raw_preference_scores)
        evidence_scores = _standardize(raw_evidence_scores)
        if not self.use_hard:
            preference_scores = torch.zeros_like(preference_scores)
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()
        final_logits = standardized_base + alpha * preference_scores
        if self.use_scene:
            final_logits = final_logits + beta * effective_gate[:, None] * evidence_scores

        preference_rep = self.preference_projection(
            torch.cat([positive_preferences, negative_preferences], dim=-1)
        )
        projected_evidence = self.evidence_projection(evidence_rep)
        gated_evidence = effective_gate[:, None] * projected_evidence
        if candidate_item_embeddings is None:
            recommendation_summary = torch.zeros_like(dialogue_rep)
        else:
            recommendation_summary = F.softmax(final_logits, dim=-1) @ candidate_item_embeddings
        generation_condition = self.response_conditioner(
            torch.cat(
                [dialogue_rep, preference_rep, gated_evidence, recommendation_summary],
                dim=-1,
            )
        )
        group_weights = positive_preferences.new_full(
            (positive_preferences.shape[0], self.gate.num_groups),
            1.0 / self.gate.num_groups,
        )
        reliability = (
            scene_weights.max(dim=-1).values
            if scene_weights.shape[-1] > 0
            else torch.zeros_like(gate_probability)
        )
        return SERAOutput(
            final_logits=final_logits,
            standardized_base_scores=standardized_base,
            preference_scores=preference_scores,
            evidence_scores=evidence_scores,
            positive_preference_logits=positive_logits,
            negative_preference_logits=negative_logits,
            positive_preferences=positive_preferences,
            negative_preferences=negative_preferences,
            preference_representation=preference_rep,
            retrieved_evidence_representation=projected_evidence,
            recommendation_summary=recommendation_summary,
            generation_condition=generation_condition,
            gate_probability=gate_probability,
            effective_gate=effective_gate,
            gate_features=gate_features,
            preference_uncertainty=uncertainty,
            preference_contrast=contrast,
            ranking_confidence=confidence,
            preference_coverage=coverage,
            retrieval_logits=retrieval_logits,
            retrieval_pool_mask=retrieval_pool,
            positive_evidence_mask=positive_evidence,
            retrieved_scene_ids=scene_ids,
            retrieved_scene_weights=scene_weights,
            current_positive_items=current_positive,
            current_negative_items=current_negative,
            hard_logits=preference_scores,
            scene_logits=evidence_scores,
            tag_logits=positive_logits,
            tag_probabilities=positive_preferences,
            negative_tag_probabilities=negative_preferences,
            tag_group_weights=group_weights,
            scene_gate=effective_gate,
            scene_reliability=reliability,
            entity_preference_logits=None,
            predicted_positive_mask=positive_entity_mask,
        )

    def training_losses(
        self,
        output: SERAOutput,
        target_ids: torch.Tensor,
        preference_weight: float = 1.0,
        negative_preference_weight: float = 1.0,
        evidence_weight: float = 1.0,
        gate_weight: float = 1.0,
        gate_margin: float = 0.0,
    ) -> Dict[str, torch.Tensor]:

        rec_loss = F.cross_entropy(output.final_logits, target_ids)
        positive_observed = (output.current_positive_items @ self.item_tags).gt(0)
        negative_observed = (output.current_negative_items @ self.item_tags).gt(0)
        observed = positive_observed.logical_xor(negative_observed)

        def observed_preference_loss(logits, targets):
            if observed.any():
                elementwise = F.binary_cross_entropy_with_logits(
                    logits, targets.to(logits.dtype), reduction="none"
                )
                return elementwise.masked_select(observed).mean()
            return logits.new_zeros(())

        positive_supervision_loss = observed_preference_loss(
            output.positive_preference_logits, positive_observed
        )
        negative_supervision_loss = observed_preference_loss(
            output.negative_preference_logits, negative_observed
        )
        preference_loss = (
            positive_supervision_loss
            + float(negative_preference_weight) * negative_supervision_loss
        )

        positive_rows = output.positive_evidence_mask.any(dim=-1)
        if positive_rows.any():
            scaled = output.retrieval_logits / max(
                self.scene_memory.temperature if self.scene_memory is not None else 1.0,
                EPS,
            )
            denominator = torch.logsumexp(
                scaled.masked_fill(~output.retrieval_pool_mask, -1e4), dim=-1
            )
            numerator = torch.logsumexp(
                scaled.masked_fill(~output.positive_evidence_mask, -1e4), dim=-1
            )
            evidence_loss = (denominator - numerator)[positive_rows].mean()
        else:
            evidence_loss = rec_loss.new_zeros(())

        alpha = self.log_alpha.exp().detach()
        beta = self.log_beta.exp().detach()
        off_scores = (
            output.standardized_base_scores.detach()
            + alpha * output.preference_scores.detach()
        )
        on_scores = off_scores + beta * output.evidence_scores.detach()
        loss_off = F.cross_entropy(off_scores, target_ids, reduction="none")
        loss_on = F.cross_entropy(on_scores, target_ids, reduction="none")
        gate_target = (loss_off - loss_on).gt(gate_margin).to(
            output.gate_probability.dtype
        )
        gate_loss = F.binary_cross_entropy(
            output.gate_probability.clamp(EPS, 1.0 - EPS), gate_target
        )
        total = (
            rec_loss
            + float(preference_weight) * preference_loss
            + float(evidence_weight) * evidence_loss
            + float(gate_weight) * gate_loss
        )
        return {
            "total": total,
            "recommendation": rec_loss,
            "preference": preference_loss,
            "evidence": evidence_loss,
            "gate": gate_loss,
            "gate_target_rate": gate_target.mean(),
        }


SERARecommender = EnhancedRecommender
