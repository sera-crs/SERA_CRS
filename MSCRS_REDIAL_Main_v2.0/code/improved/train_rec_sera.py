import argparse
import inspect
import math
import os
import sys
import time
import json
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia, Co_occurrence, text_sim, image_sim
from dataset_rec_copy import CRSRecDataset, CRSRecDataCollator
from evaluate_rec import RecEvaluator
from fixed_negative_eval import FixedCandidateEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import MMPrompt
from sera_recommender import EnhancedRecommender


def _distributed_mean(accelerator, values, device):

    stats = torch.tensor(
        [float(sum(values)), float(len(values))],
        dtype=torch.float64,
        device=device,
    )
    stats = accelerator.gather(stats[None]).sum(dim=0)
    return float(stats[0] / stats[1].clamp_min(1.0))


def _project_root():
    return os.environ.get(
        'MSCRS_ROOT',
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
    )


class HyperGraph:


    def __init__(
        self,
        dataset: str,
        split: str = "train",
        debug: bool = False,
        n_entity: Optional[int] = None,
        pad_entity_id: Optional[int] = None,
        entity_max_length: int = 32,
        max_edges: int = 80000,
        min_edge_size: int = 2,
        cache_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.split = split
        self.debug = debug
        if n_entity is None or pad_entity_id is None:
            raise ValueError("n_entity and pad_entity_id must be provided")
        self.n_entity = int(n_entity)
        self.pad_entity_id = int(pad_entity_id)
        self.entity_max_length = int(entity_max_length)
        self.max_edges = int(max_edges)
        self.min_edge_size = int(min_edge_size)
        self.hyper_lambda = 0.5


        if cache_dir is None:
            cache_dir = os.path.join(_project_root(), "data", dataset)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"hyper_H_{split}.pt")

    def get_entity_hyper_info(self) -> Dict[str, torch.Tensor]:
        if os.path.exists(self.cache_path):
            H = torch.load(self.cache_path, map_location="cpu")
            if not isinstance(H, torch.Tensor) or not H.is_sparse:
                raise ValueError(f"{self.cache_path} is not a sparse tensor")
            return {"hyper_H": H.coalesce()}

        H = self._build_hypergraph_incidence()
        torch.save(H, self.cache_path)
        return {"hyper_H": H}

    def _load_jsonl(self) -> List[dict]:

        dataset_dir = os.path.join(_project_root(), "rec_data", self.dataset)
        data_file = os.path.join(dataset_dir, f"{self.split}_data.jsonl")
        if not os.path.exists(data_file):
            data_file = os.path.join(dataset_dir, f"{self.split}_data_train.jsonl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Cannot find {data_file}. Please check your rec_data path or pass cache_dir to HyperGraph."
            )
        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if self.debug:
            lines = lines[: min(2000, len(lines))]
        return [json.loads(line) for line in lines]

    @staticmethod
    def _downsample_indices(n: int, max_keep: int) -> List[int]:
        if n <= max_keep:
            return list(range(n))
        step = max(1, n // max_keep)
        idx = list(range(0, n, step))
        return idx[:max_keep]

    def _build_hypergraph_incidence(self) -> torch.Tensor:
        dialogs = self._load_jsonl()
        keep = self._downsample_indices(len(dialogs), self.max_edges)

        indices: List[List[int]] = []
        values: List[float] = []

        edge_id = 0
        for i in tqdm(keep, desc=f"Building hypergraph H ({self.dataset}/{self.split})", disable=False):
            dialog = dialogs[i]
            ent_list = dialog.get("context_entities", dialog.get("entity", []))
            if not isinstance(ent_list, list) or len(ent_list) == 0:
                continue
            ent_list = ent_list[-self.entity_max_length:]

            ents = [int(e) for e in set(ent_list) if int(e) != self.pad_entity_id and 0 <= int(e) < self.n_entity]
            if len(ents) < self.min_edge_size:
                continue

            w = 1.0 / float(len(ents))
            for e in ents:
                indices.append([e, edge_id])
                values.append(w)
            edge_id += 1

        if edge_id == 0:
            raise RuntimeError("No hyperedges were constructed. Check your data format and entity ids.")

        idx = torch.tensor(indices, dtype=torch.long).t().contiguous()
        val = torch.tensor(values, dtype=torch.float)
        H = torch.sparse_coo_tensor(idx, val, size=(self.n_entity, edge_id)).coalesce()
        return H


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = _project_root()
    parser.add_argument("--seed", type=int, default=22, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='./prompt-for-rec', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--dataset", type=str, default='redial', help="Dataset name.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")

    parser.add_argument("--context_max_length", type=int, default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=200)
    parser.add_argument("--entity_max_length", type=int, default=32, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument("--tokenizer", type=str, default=os.path.join(project_root, 'model', 'DialoGPT-small'))
    parser.add_argument("--text_tokenizer", type=str, default=os.path.join(project_root, 'model', 'roberta-base'))
    parser.add_argument("--model", type=str, default=os.path.join(project_root, 'model', 'DialoGPT-small'),
                        help="Path to pretrained model.")
    parser.add_argument("--text_encoder", type=str, default=os.path.join(project_root, 'model', 'roberta-base'))

    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int, default=10)
    parser.add_argument("--n_prefix_conv", type=int, default=10)
    parser.add_argument("--response_max_length", type=int, default=128)
    parser.add_argument(
        "--paper_multimodal_fusion", action="store_true",
        help="Use the MSCRS Eq. (8, 12-16) three-layer multimodal fusion path.",
    )
    parser.add_argument(
        "--multimodal_lambda", type=float, default=0.5,
        help="Text/image fusion ratio from MSCRS Eq. (14); ReDial uses 0.5.",
    )
    parser.add_argument(
        "--inspired_legacy_fusion", action="store_true",
        help="Reproduce the released MMPrompt_inspired fusion order exactly.",
    )

    parser.add_argument("--prompt_encoder", type=str, default=None)

    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps. Overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=40)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=40)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=530)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true")


    parser.add_argument("--use_hypergraph", action="store_true",
                        help="Enable session-level hypergraph view for entity embedding.")
    parser.add_argument("--hyper_layers", type=int, default=2, help="Number of HGNN layers.")
    parser.add_argument("--hyper_lambda", type=float, default=0.5, help="Fusion weight alpha for hypergraph view.")
    parser.add_argument("--hyper_dropout", type=float, default=0.0, help="Dropout in HGNN layers.")
    parser.add_argument("--hyper_max_edges", type=int, default=80000, help="Max hyperedges to build.")
    parser.add_argument("--hyper_entity_max_length", type=int, default=32, help="Max entities per hyperedge.")
    parser.add_argument("--hyper_min_edge_size", type=int, default=2, help="Min unique entities in a hyperedge.")
    parser.add_argument("--hyper_cache_dir", type=str, default=None, help="Cache directory for hypergraph H.")


    parser.add_argument("--cl_weight", type=float, default=1e-4, help="Weight for contrastive loss.")


    parser.add_argument("--enhanced", action="store_true", help="Enable the paper-faithful SERA module.")
    parser.add_argument("--item_tag_file", type=str, default=None, help="item_tags_r*.pt from build_assets.py")
    parser.add_argument("--scene_memory_file", type=str, default=None, help="scene_memory_r*.pt from build_assets.py")
    parser.add_argument("--scene_top_k", type=int, default=3)
    parser.add_argument(
        "--scene_profile_weight", type=float, default=0.35,
        help="Blend retrieved hyperedge membership with scene-profile/item matching.",
    )
    parser.add_argument("--hard_alpha", type=float, default=1.0)
    parser.add_argument("--scene_beta", type=float, default=1.0)
    parser.add_argument("--negative_preference_penalty", type=float, default=1.0)
    parser.add_argument("--scene_tag_threshold", type=float, default=0.5)
    parser.add_argument("--scene_semantic_weight", type=float, default=1.0)
    parser.add_argument("--scene_positive_weight", type=float, default=1.0)
    parser.add_argument("--scene_negative_weight", type=float, default=1.0)
    parser.add_argument("--scene_temperature", type=float, default=0.1)
    parser.add_argument("--scene_item_degree_power", type=float, default=1.0)
    parser.add_argument("--scene_edge_degree_power", type=float, default=1.0)
    parser.add_argument("--coverage_top_k", type=int, default=10)
    parser.add_argument("--preference_threshold", type=float, default=0.0)
    parser.add_argument("--coverage_smoothness", type=float, default=0.1)
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--preference_loss_weight", type=float, default=1.0)
    parser.add_argument("--negative_preference_loss_weight", type=float, default=1.0)
    parser.add_argument("--evidence_loss_weight", type=float, default=1.0)
    parser.add_argument("--gate_loss_weight", type=float, default=1.0)
    parser.add_argument("--gate_margin", type=float, default=0.0)
    parser.add_argument("--generation_loss_weight", type=float, default=1.0)
    parser.add_argument("--tag_loss_weight", type=float, default=0.0,
                        help="Deprecated v1 option; retained only for CLI compatibility.")
    parser.add_argument("--gate_positive_weight", type=float, default=3.0)
    parser.add_argument("--hard_threshold", type=float, default=0.8)
    parser.add_argument("--positive_anchor_threshold", type=float, default=0.0,
                        help="Minimum predicted P(positive) for scene anchoring.")
    parser.add_argument("--disable_score_normalization", action="store_true",
                        help="Disable per-dialogue scale alignment for hard/scene logits.")
    parser.add_argument("--hard_gate_inference", action="store_true")
    parser.add_argument("--always_on_gate", action="store_true")
    parser.add_argument("--evaluate_test_each_epoch", action="store_true",
                        help="Diagnostic only. Default evaluates test once after model selection.")
    parser.add_argument("--skip_test", action="store_true",
                        help="Do not evaluate test (use for validation-only R selection).")
    parser.add_argument("--disable_hard_branch", action="store_true")
    parser.add_argument("--disable_scene_branch", action="store_true")
    parser.add_argument("--freeze_prompt_encoder", action="store_true",
                        help="Train only the enhancement head during R pre-screening.")
    parser.add_argument("--freeze_enhanced_except_gate", action="store_true",
                        help="Second-stage calibration: update only SceneNeedGate.")
    parser.add_argument("--freeze_enhanced_except_scene_fusion", action="store_true",
                        help="Second stage: update SceneNeedGate and beta only.")
    parser.add_argument("--allow_partial_enhanced_checkpoint", action="store_true",
                        help="Allow adding scene-memory buffers to a hard-only checkpoint.")
    parser.add_argument(
        "--selection_metric", default="ndcg@50",
        choices=["loss", "generation_loss", "recall@1", "recall@10", "recall@50", "mrr@50", "ndcg@50"],
        help="Validation-only metric used for checkpoint and tag-count selection.",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=0,
        help="Stop after this many non-improving validation epochs; 0 disables it.",
    )
    parser.add_argument("--enhanced_checkpoint", default=None,
                        help="Load enhanced_model.pt for evaluation/calibration.")
    parser.add_argument("--eval_calibration", action="store_true",
                        help="Evaluate an alpha/beta scale grid without training.")
    parser.add_argument("--calibration_split", choices=["valid", "test"], default="valid")
    parser.add_argument("--calibration_alpha_scales", default="0,0.5,0.75,1,1.25,1.5")
    parser.add_argument("--calibration_beta_scales", default="0,0.25,0.5,0.75,1,1.25")
    parser.add_argument("--calibration_output", default=None)
    parser.add_argument(
        "--fixed_negative_file", default=None,
        help="Frozen one-positive-plus-99-negatives artifact for sampled test evaluation.",
    )

    args = parser.parse_args()
    nonnegative = (
        'negative_preference_penalty',
        'preference_loss_weight',
        'negative_preference_loss_weight',
        'evidence_loss_weight',
        'gate_loss_weight',
        'generation_loss_weight',
        'cl_weight',
        'scene_semantic_weight',
        'scene_positive_weight',
        'scene_negative_weight',
        'scene_item_degree_power',
        'scene_edge_degree_power',
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f'--{name} must be non-negative')
    if args.hard_alpha <= 0 or args.scene_beta <= 0:
        parser.error('--hard_alpha and --scene_beta must be positive; use disable flags for ablations')
    if args.scene_temperature <= 0 or args.coverage_smoothness <= 0:
        parser.error('--scene_temperature and --coverage_smoothness must be positive')
    if args.scene_top_k <= 0 or args.coverage_top_k <= 0:
        parser.error('--scene_top_k and --coverage_top_k must be positive')
    if not 0.0 < args.gate_threshold < 1.0:
        parser.error('--gate_threshold must lie strictly between zero and one')
    if not 0.0 < args.hard_threshold <= 1.0:
        parser.error('--hard_threshold must lie in (0, 1]')
    if args.hard_gate_inference and args.always_on_gate:
        parser.error('--hard_gate_inference and --always_on_gate are mutually exclusive')
    return args


def evaluate_score_grid(
    args, accelerator, device, model, text_encoder, prompt_encoder, enhanced_model,
    dataloader, item_ids_tensor, global_to_local, item_ids,
):

    alpha_scales = [float(value) for value in args.calibration_alpha_scales.split(',')]
    beta_scales = [float(value) for value in args.calibration_beta_scales.split(',')]
    combinations = [(alpha, beta) for alpha in alpha_scales for beta in beta_scales]
    if args.calibration_split == 'test' and len(combinations) != 1:
        raise ValueError("test evaluation accepts exactly one preselected alpha/beta pair")
    evaluators = {pair: RecEvaluator(device=device) for pair in combinations}
    sampled_evaluators = None
    if args.fixed_negative_file:
        if args.calibration_split != 'test':
            raise ValueError("fixed-negative evaluation is permitted only on the test split")
        sampled_evaluators = {
            pair: FixedCandidateEvaluator(
                args.fixed_negative_file, item_ids, device=device
            )
            for pair in combinations
        }
    loss_sums = {pair: 0.0 for pair in combinations}
    count = 0
    prompt_encoder.eval()
    unwrapped = None
    learned_alpha = 0.0
    learned_beta = 0.0
    if enhanced_model is not None:
        enhanced_model.eval()
        unwrapped = accelerator.unwrap_model(enhanced_model)
        learned_alpha = float(unwrapped.log_alpha.exp().detach())
        learned_beta = float(unwrapped.log_beta.exp().detach())

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds, _, entity_table = prompt_encoder(
                entity_ids=batch['entity'], token_embeds=token_embeds,
                output_entity=True, use_rec_prefix=True,
                entity_mask=None if args.inspired_legacy_fusion else batch['entity_mask'],
                return_entity_table=True,
            )
            context_inputs = dict(batch['context'])
            context_inputs['prompt_embeds'] = prompt_embeds
            context_inputs['entity_embeds'] = entity_table
            outputs = model(**context_inputs, rec=True)
            base_logits = outputs.rec_logits[:, item_ids_tensor]
            fusion_base = base_logits
            labels_local = global_to_local[batch['context']['rec_labels']]
            hard_component = torch.zeros_like(base_logits)
            scene_component = torch.zeros_like(base_logits)
            if enhanced_model is not None:
                safe_entities = batch['entity'].clamp(
                    min=0, max=global_to_local.shape[0] - 1
                )
                local_entities = global_to_local[safe_entities]
                enhanced = enhanced_model(
                    base_logits=base_logits, dialogue_rep=outputs.rec_rep,
                    conversation_ids=batch['conversation_ids'],
                    hard_inference=args.hard_gate_inference,
                    always_on_gate=args.always_on_gate,
                    entity_vectors=entity_table[batch['entity']],
                    entity_mask=batch['entity_mask'] & local_entities.ge(0),
                    entity_candidate_ids=local_entities,
                    positive_entity_mask=batch['positive_entity_mask'],
                    negative_entity_mask=batch['negative_entity_mask'],
                    candidate_item_embeddings=entity_table[item_ids_tensor],
                )
                fusion_base = enhanced.standardized_base_scores
                if unwrapped.use_hard:
                    hard_component = learned_alpha * enhanced.hard_logits
                if unwrapped.use_scene:
                    scene_component = (
                        learned_beta * enhanced.scene_gate[:, None]
                        * enhanced.scene_logits
                    )
            labels_global = batch['context']['rec_labels']
            for pair in combinations:
                alpha_scale, beta_scale = pair
                logits = fusion_base + alpha_scale * hard_component + beta_scale * scene_component
                loss_sums[pair] += float(
                    torch.nn.functional.cross_entropy(logits, labels_local, reduction='sum')
                )
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[item_ids[index] for index in batch_rank] for batch_rank in ranks]
                evaluators[pair].evaluate(ranks, labels_global)
                if sampled_evaluators is not None:
                    sampled_evaluators[pair].evaluate(
                        logits, labels_global, batch['sample_ids']
                    )
            count += base_logits.shape[0]

    rows = []
    for alpha_scale, beta_scale in combinations:
        loss_stats = torch.tensor(
            [loss_sums[(alpha_scale, beta_scale)], float(count)],
            dtype=torch.float64,
            device=device,
        )
        loss_stats = accelerator.gather(loss_stats[None]).sum(dim=0)
        report = evaluators[(alpha_scale, beta_scale)].report()
        report = {key: value.sum().item() for key, value in accelerator.gather(report).items()}
        row = {
            'alpha_scale': alpha_scale,
            'beta_scale': beta_scale,
            'effective_alpha': learned_alpha * alpha_scale,
            'effective_beta': learned_beta * beta_scale,
            'loss': float(loss_stats[0] / loss_stats[1].clamp_min(1.0)),
        }
        row.update({key: value / report['count'] for key, value in report.items() if key != 'count'})
        if sampled_evaluators is not None:
            sampled_report = sampled_evaluators[(alpha_scale, beta_scale)].report()
            sampled_report = {
                key: value.sum().item()
                for key, value in accelerator.gather(sampled_report).items()
            }
            sampled_count = sampled_report['count']
            row['sampled_100'] = {
                key: value / sampled_count
                for key, value in sampled_report.items()
                if key != 'count'
            }
            row['sampled_100']['count'] = int(sampled_count)
        rows.append(row)
    metric = args.selection_metric
    best = min(rows, key=lambda row: row[metric]) if metric == 'loss' else max(
        rows, key=lambda row: row[metric]
    )
    payload = {
        'split': args.calibration_split,
        'selection_metric': metric,
        'learned_alpha': learned_alpha,
        'learned_beta': learned_beta,
        'rows': rows,
        'best': best,
    }
    if sampled_evaluators is not None:
        payload['fixed_candidate_artifact'] = next(
            iter(sampled_evaluators.values())
        ).metadata
    if args.calibration_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.calibration_output)), exist_ok=True)
        with open(args.calibration_output, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2)
    logger.info(f'calibration best: {best}')
    return payload


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    accelerator = Accelerator(
        device_placement=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device

    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    run_log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(run_log_dir, exist_ok=True)
    logger.add(
        os.path.join(run_log_dir, f'{local_time}.log'),
        level='DEBUG' if accelerator.is_local_main_process else 'ERROR',
    )
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()


    if args.use_wandb:
        try:
            import wandb
        except ImportError as error:
            raise ImportError("--use_wandb requires the optional 'wandb' package") from error
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)
        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            run = wandb.init(entity=args.entity, project=args.project, config=config, name=name) \
                if accelerator.is_local_main_process else None
    else:
        run = None

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, 'run_config.json'), 'w', encoding='utf-8') as stream:
                json.dump(config, stream, indent=2, sort_keys=True)
    if args.prompt_encoder is not None and str(args.prompt_encoder).lower() not in {'none', 'null', ''}:
        checkpoint_file = os.path.join(args.prompt_encoder, 'model.pt')
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(
                f"Prompt checkpoint not found: {checkpoint_file}. "
                "Pass --prompt_encoder none to train without initialization."
            )


    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)


    train_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
        response_max_length=args.response_max_length,
    )

    co = Co_occurrence(
        dataset=args.dataset, split='train', debug=args.debug,
        all_items=kg['item_ids'], entity_max_length=args.entity_max_length, n_entity=kg['num_entities']
    ).get_entity_co_info()

    text_simi = text_sim(
        pad_entity_id=kg['pad_entity_id'], dataset=args.dataset
    ).get_entity_ts_info()
    image_simi = image_sim(
        pad_entity_id=kg['pad_entity_id'], dataset=args.dataset
    ).get_entity_is_info()


    hyper_H = None
    if args.use_hypergraph:
        hg = HyperGraph(
            dataset=args.dataset,
            split="train",
            debug=args.debug,
            n_entity=kg["num_entities"],
            pad_entity_id=kg["pad_entity_id"],
            entity_max_length=args.hyper_entity_max_length,
            max_edges=args.hyper_max_edges,
            min_edge_size=args.hyper_min_edge_size,
            cache_dir=args.hyper_cache_dir,
        )
        hyper_H = hg.get_entity_hyper_info()["hyper_H"]
        logger.info(f"Hypergraph H loaded: shape={tuple(hyper_H.shape)}, nnz={hyper_H._nnz()}")

    shot_len = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    assert len(train_dataset) == shot_len

    valid_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
        response_max_length=args.response_max_length,
    )


    test_dataset = None
    if not args.skip_test:
        test_dataset = CRSRecDataset(
            dataset=args.dataset, split='test', debug=args.debug,
            tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
            prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
            entity_max_length=args.entity_max_length,
            response_max_length=args.response_max_length,
        )

    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        include_generation=args.generation_loss_weight > 0,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
    )
    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=args.num_workers,
        )


    prompt_encoder = MMPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        edge_index_c=co['edge_index_c'],
        edge_index_t_s=text_simi['edge_index_t_s'],
        edge_index_i_s=image_simi['edge_index_i_s'],
        idx_to_id=text_simi['idx_to_id'],
        n_prefix_rec=args.n_prefix_rec,
        n_prefix_conv=args.n_prefix_conv,
        hyper_H=hyper_H,
        hyper_layers=args.hyper_layers,
        hyper_lambda=args.hyper_lambda,
        hyper_dropout=args.hyper_dropout,
        paper_multimodal_fusion=args.paper_multimodal_fusion,
        multimodal_lambda=args.multimodal_lambda,
        inspired_legacy_fusion=args.inspired_legacy_fusion,
    )

    if args.prompt_encoder is not None and str(args.prompt_encoder).lower() not in {'none', 'null', ''}:
        prompt_encoder.load(args.prompt_encoder)

    prompt_encoder = prompt_encoder.to(device)
    if args.freeze_prompt_encoder:
        prompt_encoder.requires_grad_(False)


    text_encoder.requires_grad_(False)
    model.requires_grad_(args.generation_loss_weight > 0)

    enhanced_model = None
    tag_group_names = []
    item_ids_tensor = torch.as_tensor(kg['item_ids'], device=device)
    global_to_local = torch.full((kg['num_entities'],), -1, dtype=torch.long, device=device)
    global_to_local[item_ids_tensor] = torch.arange(len(kg['item_ids']), device=device)
    if args.enhanced:
        if not args.item_tag_file or not os.path.isfile(args.item_tag_file):
            raise FileNotFoundError("--enhanced requires a valid --item_tag_file")
        tag_asset = torch.load(args.item_tag_file, map_location='cpu')
        tag_group_names = list(tag_asset.get('group_names', ['all']))
        if args.dataset == 'redial':
            tag_metrics = tag_asset.get('metrics', {})
            if tag_metrics.get('backend') != 'predefined_metadata':
                raise ValueError('ReDial requires predefined metadata attributes')
            if len(tag_group_names) < 2:
                raise ValueError('ReDial attributes must retain semantic groups')
            if not tag_metrics.get('category_source_sha256'):
                raise ValueError('ReDial item tags are missing source provenance')
        if tag_asset['item_ids'].tolist() != list(kg['item_ids']):
            raise ValueError("item tag candidate order does not match DBpedia item_ids")
        scene_assets = None
        if args.scene_memory_file:
            scene_asset = torch.load(args.scene_memory_file, map_location='cpu')
            if scene_asset.get('item_ids') is None:
                raise ValueError("scene memory is missing candidate item_ids")
            if scene_asset['item_ids'].tolist() != list(kg['item_ids']):
                raise ValueError("scene-memory candidate order does not match DBpedia item_ids")
            if scene_asset['scene_tag_profiles'].shape[1] != tag_asset['item_tags'].shape[1]:
                raise ValueError("scene and item hard-tag dimensions do not match")
            if scene_asset['scene_movie_incidence'].shape[1] != len(kg['item_ids']):
                raise ValueError("scene incidence candidate dimension is invalid")
            scene_assets = {
                key: scene_asset[key]
                for key in ['scene_embeddings', 'scene_tag_profiles', 'scene_movie_incidence', 'scene_conversation_ids']
            }
        enhanced_kwargs = dict(
            hidden_size=model.config.n_embd,
            item_tags=tag_asset['item_tags'],
            tag_group_ids=tag_asset.get('tag_group_ids'),
            group_is_exclusive=tag_asset.get('group_is_exclusive'),
            scene_assets=scene_assets,
            alpha=args.hard_alpha,
            beta=args.scene_beta,
            negative_penalty=args.negative_preference_penalty,
            scene_top_k=args.scene_top_k,
            scene_tag_threshold=args.scene_tag_threshold,
            scene_semantic_weight=args.scene_semantic_weight,
            scene_positive_weight=args.scene_positive_weight,
            scene_negative_weight=args.scene_negative_weight,
            scene_temperature=args.scene_temperature,
            scene_item_degree_power=args.scene_item_degree_power,
            scene_edge_degree_power=args.scene_edge_degree_power,
            coverage_top_k=args.coverage_top_k,
            preference_threshold=args.preference_threshold,
            coverage_smoothness=args.coverage_smoothness,
            gate_threshold=args.gate_threshold,
            hard_threshold=args.hard_threshold,
            use_hard=not args.disable_hard_branch,
            use_scene=not args.disable_scene_branch,
        )


        enhanced_parameters = inspect.signature(EnhancedRecommender.__init__).parameters
        enhanced_model = EnhancedRecommender(**enhanced_kwargs).to(device)
        if not hasattr(enhanced_model.tag_head, 'num_groups'):
            tag_group_names = ['all']
        if args.enhanced_checkpoint:
            checkpoint_path = args.enhanced_checkpoint
            if os.path.isdir(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_path, 'enhanced_model.pt')
            load_result = enhanced_model.load_state_dict(
                torch.load(checkpoint_path, map_location='cpu'),
                strict=not args.allow_partial_enhanced_checkpoint,
            )
            if args.allow_partial_enhanced_checkpoint:
                logger.info(
                    f"partial enhanced load: missing={load_result.missing_keys}, "
                    f"unexpected={load_result.unexpected_keys}"
                )
        if args.freeze_enhanced_except_gate:
            enhanced_model.requires_grad_(False)
            enhanced_model.gate.requires_grad_(True)
        if args.freeze_enhanced_except_scene_fusion:
            enhanced_model.requires_grad_(False)
            enhanced_model.gate.requires_grad_(True)
            enhanced_model.log_beta.requires_grad_(True)

    modules = [prompt_encoder]
    if args.generation_loss_weight > 0:
        modules.append(model)
    if enhanced_model is not None:
        modules.append(enhanced_model)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for m in modules for n, p in m.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for m in modules for n, p in m.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    evaluator = RecEvaluator(device=device)

    if enhanced_model is None:
        if test_dataloader is None:
            model, prompt_encoder, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, prompt_encoder, optimizer, train_dataloader, valid_dataloader
            )
        else:
            model, prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
                model, prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
            )
    else:
        if test_dataloader is None:
            model, prompt_encoder, enhanced_model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, prompt_encoder, enhanced_model, optimizer, train_dataloader, valid_dataloader
            )
        else:
            model, prompt_encoder, enhanced_model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
                model, prompt_encoder, enhanced_model, optimizer, train_dataloader, valid_dataloader, test_dataloader
            )

    if args.eval_calibration:
        calibration_dataloader = (
            valid_dataloader if args.calibration_split == 'valid' else test_dataloader
        )
        evaluate_score_grid(
            args, accelerator, device, model, text_encoder, prompt_encoder,
            enhanced_model, calibration_dataloader, item_ids_tensor,
            global_to_local, kg['item_ids'],
        )
        raise SystemExit(0)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num valid examples = {len(valid_dataset)}")
    logger.info(
        f"  Num test examples  = {len(test_dataset) if test_dataset is not None else 'SEALED'}"
    )
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    metric = args.selection_metric
    mode = -1 if metric in {'loss', 'generation_loss'} else 1
    best_metric = float('inf') if mode == -1 else float('-inf')
    epochs_without_improvement = 0
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    for epoch in range(args.num_train_epochs):
        train_losses = []
        model.train() if args.generation_loss_weight > 0 else model.eval()
        prompt_encoder.eval() if args.freeze_prompt_encoder else prompt_encoder.train()
        if enhanced_model is not None:
            if args.freeze_enhanced_except_gate or args.freeze_enhanced_except_scene_fusion:
                enhanced_model.eval()
                accelerator.unwrap_model(enhanced_model).gate.train()
            else:
                enhanced_model.train()

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state

            prompt_result = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True,
                entity_mask=None if args.inspired_legacy_fusion else batch['entity_mask'],
                return_entity_table=True,
            )
            prompt_embeds, loss_cl, entity_table = prompt_result

            context_inputs = dict(batch['context'])
            context_inputs['prompt_embeds'] = prompt_embeds
            context_inputs['entity_embeds'] = entity_table
            outputs = model(**context_inputs, rec=True)
            if enhanced_model is None:
                loss = outputs.rec_loss
            else:
                candidate_logits = outputs.rec_logits[:, item_ids_tensor]
                labels_local = global_to_local[batch['context']['rec_labels']]
                if labels_local.lt(0).any():
                    raise ValueError("a recommendation label is not in item_ids")
                safe_entities = batch['entity'].clamp(min=0, max=global_to_local.shape[0] - 1)
                local_entities = global_to_local[safe_entities]
                enhanced_output = enhanced_model(
                    base_logits=candidate_logits,
                    dialogue_rep=outputs.rec_rep,
                    conversation_ids=batch['conversation_ids'],
                    entity_vectors=entity_table[batch['entity']],
                    entity_mask=batch['entity_mask'] & local_entities.ge(0),
                    entity_candidate_ids=local_entities,
                    positive_entity_mask=batch['positive_entity_mask'],
                    negative_entity_mask=batch['negative_entity_mask'],
                    candidate_item_embeddings=entity_table[item_ids_tensor],
                    target_ids=labels_local,
                )
                sera_losses = accelerator.unwrap_model(enhanced_model).training_losses(
                    enhanced_output,
                    labels_local,
                    preference_weight=args.preference_loss_weight,
                    negative_preference_weight=args.negative_preference_loss_weight,
                    evidence_weight=args.evidence_loss_weight,
                    gate_weight=args.gate_loss_weight,
                    gate_margin=args.gate_margin,
                )
                loss = sera_losses['total']
                if args.generation_loss_weight > 0:


                    generation_token_embeds = torch.cat(
                        [token_embeds, enhanced_output.generation_condition.unsqueeze(1)],
                        dim=1,
                    )
                    generation_prompt, generation_aux = prompt_encoder(
                        entity_ids=batch['entity'],
                        token_embeds=generation_token_embeds,
                        output_entity=True,
                        use_conv_prefix=True,
                        entity_mask=None if args.inspired_legacy_fusion else batch['entity_mask'],
                    )
                    generation_inputs = dict(batch['generation'])
                    generation_inputs['prompt_embeds'] = generation_prompt
                    generation_output = model(
                        **generation_inputs,
                        conv=True,
                        conv_labels=batch['generation_labels'],
                    )
                    loss = loss + args.generation_loss_weight * generation_output.conv_loss
                    if generation_aux is not None:
                        loss = loss + args.cl_weight * generation_aux
            if loss_cl is not None:
                loss = loss + args.cl_weight * loss_cl
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            train_losses.append(float(loss))

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    trainable = [p for module in modules for p in module.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        train_loss = np.mean(train_losses) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')
        del train_losses, batch


        valid_loss = []
        valid_generation_loss = []
        gate_sum = scene_active_sum = enhanced_count = 0.0
        scene_target_sum = 0.0
        if enhanced_model is not None:
            validation_sera = accelerator.unwrap_model(enhanced_model)
            num_tag_groups = int(validation_sera.gate.num_groups)
            tag_stats = torch.zeros(num_tag_groups, 3, device=device)
            group_weight_sum = torch.zeros(num_tag_groups, device=device)
        evaluator.reset_metric()
        model.eval()
        prompt_encoder.eval()
        if enhanced_model is not None:
            enhanced_model.eval()
        for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds, _, entity_table = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True,
                    entity_mask=None if args.inspired_legacy_fusion else batch['entity_mask'],
                    return_entity_table=True,
                )
                context_inputs = dict(batch['context'])
                context_inputs['prompt_embeds'] = prompt_embeds
                context_inputs['entity_embeds'] = entity_table
                outputs = model(**context_inputs, rec=True)
                logits = outputs.rec_logits[:, item_ids_tensor]
                if enhanced_model is not None:
                    labels_local = global_to_local[batch['context']['rec_labels']]
                    safe_entities = batch['entity'].clamp(min=0, max=global_to_local.shape[0] - 1)
                    local_entities = global_to_local[safe_entities]
                    enhanced_output = enhanced_model(
                        base_logits=logits,
                        dialogue_rep=outputs.rec_rep,
                        conversation_ids=batch['conversation_ids'],
                        hard_inference=args.hard_gate_inference,
                        always_on_gate=args.always_on_gate,
                        entity_vectors=entity_table[batch['entity']],
                        entity_mask=batch['entity_mask'] & local_entities.ge(0),
                        entity_candidate_ids=local_entities,
                        positive_entity_mask=batch['positive_entity_mask'],
                        negative_entity_mask=batch['negative_entity_mask'],
                        candidate_item_embeddings=entity_table[item_ids_tensor],
                    )
                    logits = enhanced_output.final_logits
                    valid_loss.append(float(torch.nn.functional.cross_entropy(logits, labels_local)))
                    if args.generation_loss_weight > 0:
                        generation_token_embeds = torch.cat(
                            [token_embeds, enhanced_output.generation_condition.unsqueeze(1)],
                            dim=1,
                        )
                        generation_prompt, _ = prompt_encoder(
                            entity_ids=batch['entity'],
                            token_embeds=generation_token_embeds,
                            output_entity=True,
                            use_conv_prefix=True,
                            entity_mask=None if args.inspired_legacy_fusion else batch['entity_mask'],
                        )
                        generation_inputs = dict(batch['generation'])
                        generation_inputs['prompt_embeds'] = generation_prompt
                        generation_output = model(
                            **generation_inputs,
                            conv=True,
                            conv_labels=batch['generation_labels'],
                        )
                        valid_generation_loss.append(float(generation_output.conv_loss))
                    unwrapped_enhanced = accelerator.unwrap_model(enhanced_model)
                    target_tags = unwrapped_enhanced.item_tags[labels_local].bool()
                    predicted_tags = enhanced_output.tag_probabilities.ge(0.5)
                    validation_group_ids = unwrapped_enhanced.tag_group_ids
                    for group in range(num_tag_groups):
                        group_mask = validation_group_ids.eq(group)
                        target_group = target_tags[:, group_mask]
                        predicted_group = predicted_tags[:, group_mask]
                        tag_stats[group, 0] += (target_group & predicted_group).sum()
                        tag_stats[group, 1] += (~target_group & predicted_group).sum()
                        tag_stats[group, 2] += (target_group & ~predicted_group).sum()
                    output_group_weights = getattr(
                        enhanced_output, 'tag_group_weights',
                        torch.ones(
                            logits.shape[0], num_tag_groups,
                            dtype=logits.dtype, device=device,
                        ),
                    )
                    group_weight_sum += output_group_weights.sum(dim=0)
                    gate_sum += float(enhanced_output.scene_gate.sum())
                    scene_active_sum += float(
                        enhanced_output.retrieved_scene_weights.sum(dim=-1).gt(0).sum()
                    )
                    scene_target_sum += float(
                        enhanced_output.scene_logits[
                            torch.arange(labels_local.shape[0], device=device), labels_local
                        ].gt(0).sum()
                    )
                    enhanced_count += logits.shape[0]
                else:
                    valid_loss.append(float(outputs.rec_loss))
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = _distributed_mean(
            accelerator, valid_loss, device
        )
        valid_report['valid/generation_loss'] = _distributed_mean(
            accelerator, valid_generation_loss, device
        )
        if enhanced_model is not None:
            gathered_tag_stats = accelerator.gather(tag_stats[None]).sum(dim=0)
            tag_f1 = (
                2 * gathered_tag_stats[:, 0]
                / (
                    2 * gathered_tag_stats[:, 0]
                    + gathered_tag_stats[:, 1]
                    + gathered_tag_stats[:, 2]
                ).clamp_min(1)
            )
            valid_report['valid/positive_preference_group_macro_f1'] = float(tag_f1.mean())
            gathered_group_weights = accelerator.gather(group_weight_sum[None]).sum(dim=0)
            gathered_gate_stats = accelerator.gather(
                torch.tensor(
                    [[gate_sum, scene_active_sum, scene_target_sum, enhanced_count]],
                    dtype=torch.float64,
                    device=device,
                )
            ).sum(dim=0)
            global_enhanced_count = max(1.0, float(gathered_gate_stats[3]))
            for group, name in enumerate(tag_group_names):
                if group < gathered_group_weights.numel():
                    valid_report[f'valid/group_weight_{name}'] = float(
                        gathered_group_weights[group] / global_enhanced_count
                    )
            valid_report['valid/scene_gate_mean'] = (
                float(gathered_gate_stats[0]) / global_enhanced_count
            )
            valid_report['valid/scene_activation_rate'] = (
                float(gathered_gate_stats[1]) / global_enhanced_count
            )
            valid_report['valid/scene_target_coverage'] = (
                float(gathered_gate_stats[2]) / global_enhanced_count
            )
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if accelerator.is_main_process:
            with open(
                os.path.join(args.output_dir, 'validation_metrics.jsonl'), 'a', encoding='utf-8'
            ) as stream:
                stream.write(json.dumps(valid_report, sort_keys=True) + '\n')

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.unwrap_model(prompt_encoder).save(best_metric_dir)
                if args.generation_loss_weight > 0:
                    accelerator.unwrap_model(model).save_pretrained(
                        os.path.join(best_metric_dir, 'generator')
                    )
                if enhanced_model is not None:
                    enhanced_state = accelerator.unwrap_model(enhanced_model).state_dict()
                    torch.save(enhanced_state, os.path.join(best_metric_dir, 'enhanced_model.pt'))
            accelerator.wait_for_everyone()
            best_metric = valid_report[f'valid/{metric}']
            epochs_without_improvement = 0
            logger.info(f'new best model with {metric}')
        else:
            epochs_without_improvement += 1

        should_stop_early = (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        )


        if not args.evaluate_test_each_epoch or args.skip_test:
            if should_stop_early:
                logger.info(
                    f'early stopping after {epochs_without_improvement} '
                    f'non-improving validation epochs'
                )
                break
            continue
        test_loss = []
        evaluator.reset_metric()
        model.eval()
        prompt_encoder.eval()
        if enhanced_model is not None:
            enhanced_model.eval()
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds, _, entity_table = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True,
                    entity_mask=None if args.inspired_legacy_fusion else batch['entity_mask'],
                    return_entity_table=True,
                )
                context_inputs = dict(batch['context'])
                context_inputs['prompt_embeds'] = prompt_embeds
                context_inputs['entity_embeds'] = entity_table
                outputs = model(**context_inputs, rec=True)
                logits = outputs.rec_logits[:, item_ids_tensor]
                if enhanced_model is not None:
                    labels_local = global_to_local[batch['context']['rec_labels']]
                    safe_entities = batch['entity'].clamp(min=0, max=global_to_local.shape[0] - 1)
                    local_entities = global_to_local[safe_entities]
                    enhanced_output = enhanced_model(
                        base_logits=logits,
                        dialogue_rep=outputs.rec_rep,
                        conversation_ids=batch['conversation_ids'],
                        hard_inference=args.hard_gate_inference,
                        always_on_gate=args.always_on_gate,
                        entity_vectors=entity_table[batch['entity']],
                        entity_mask=batch['entity_mask'] & local_entities.ge(0),
                        entity_candidate_ids=local_entities,
                        positive_entity_mask=batch['positive_entity_mask'],
                        negative_entity_mask=batch['negative_entity_mask'],
                        candidate_item_embeddings=entity_table[item_ids_tensor],
                    )
                    logits = enhanced_output.final_logits
                    test_loss.append(float(torch.nn.functional.cross_entropy(logits, labels_local)))
                else:
                    test_loss.append(float(outputs.rec_loss))
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / report['count']
        test_report['test/loss'] = _distributed_mean(
            accelerator, test_loss, device
        )
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if should_stop_early:
            logger.info(
                f'early stopping after {epochs_without_improvement} '
                f'non-improving validation epochs'
            )
            break

    final_dir = os.path.join(args.output_dir, 'final')
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(prompt_encoder).save(final_dir)
        if args.generation_loss_weight > 0:
            accelerator.unwrap_model(model).save_pretrained(
                os.path.join(final_dir, 'generator')
            )
        if enhanced_model is not None:
            enhanced_state = accelerator.unwrap_model(enhanced_model).state_dict()
            torch.save(enhanced_state, os.path.join(final_dir, 'enhanced_model.pt'))
    accelerator.wait_for_everyone()
    logger.info('save final model')

    if not args.evaluate_test_each_epoch and not args.skip_test:

        accelerator.unwrap_model(prompt_encoder).load(best_metric_dir)
        if args.generation_loss_weight > 0:
            selected_generator = PromptGPT2forCRS.from_pretrained(
                os.path.join(best_metric_dir, 'generator')
            )
            accelerator.unwrap_model(model).load_state_dict(
                selected_generator.state_dict()
            )
        if enhanced_model is not None:
            enhanced_state = torch.load(
                os.path.join(best_metric_dir, 'enhanced_model.pt'), map_location='cpu'
            )
            accelerator.unwrap_model(enhanced_model).load_state_dict(enhanced_state)
        evaluator.reset_metric()
        test_loss = []
        model.eval()
        prompt_encoder.eval()
        if enhanced_model is not None:
            enhanced_model.eval()
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds, _, entity_table = prompt_encoder(
                    entity_ids=batch['entity'], token_embeds=token_embeds,
                    output_entity=True, use_rec_prefix=True,
                    entity_mask=None if args.inspired_legacy_fusion else batch['entity_mask'],
                    return_entity_table=True,
                )
                context_inputs = dict(batch['context'])
                context_inputs['prompt_embeds'] = prompt_embeds
                context_inputs['entity_embeds'] = entity_table
                outputs = model(**context_inputs, rec=True)
                logits = outputs.rec_logits[:, item_ids_tensor]
                if enhanced_model is not None:
                    labels_local = global_to_local[batch['context']['rec_labels']]
                    safe_entities = batch['entity'].clamp(min=0, max=global_to_local.shape[0] - 1)
                    local_entities = global_to_local[safe_entities]
                    enhanced_output = enhanced_model(
                        base_logits=logits, dialogue_rep=outputs.rec_rep,
                        conversation_ids=batch['conversation_ids'],
                        hard_inference=args.hard_gate_inference,
                        always_on_gate=args.always_on_gate,
                        entity_vectors=entity_table[batch['entity']],
                        entity_mask=batch['entity_mask'] & local_entities.ge(0),
                        entity_candidate_ids=local_entities,
                        positive_entity_mask=batch['positive_entity_mask'],
                        negative_entity_mask=batch['negative_entity_mask'],
                        candidate_item_embeddings=entity_table[item_ids_tensor],
                    )
                    logits = enhanced_output.final_logits
                    test_loss.append(float(torch.nn.functional.cross_entropy(logits, labels_local)))
                else:
                    test_loss.append(float(outputs.rec_loss))
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                evaluator.evaluate(ranks, batch['context']['rec_labels'])
        report = accelerator.gather(evaluator.report())
        report = {key: value.sum().item() for key, value in report.items()}
        test_report = {
            f'test/{key}': value / report['count']
            for key, value in report.items() if key != 'count'
        }
        test_report['test/loss'] = _distributed_mean(
            accelerator, test_loss, device
        )
        logger.info(f'{test_report}')
