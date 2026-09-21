

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_conv_sera import SERAConvDataCollator, SERAConvDataset
from dataset_dbpedia import DBpedia, Co_occurrence, image_sim, text_sim
from model_gpt2 import PromptGPT2forCRS
from model_prompt import MMPrompt
from sera_recommender import EnhancedRecommender


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["redial", "inspired"], required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text_tokenizer", required=True)
    parser.add_argument("--model", required=True, help="Validation-selected generator directory")
    parser.add_argument("--text_encoder", required=True)
    parser.add_argument("--prompt_encoder", required=True)
    parser.add_argument("--enhanced_checkpoint", required=True)
    parser.add_argument("--item_tag_file", required=True)
    parser.add_argument("--scene_memory_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--context_max_length", type=int, default=200)
    parser.add_argument("--response_max_length", type=int, default=128)
    parser.add_argument("--entity_max_length", type=int, default=32)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--n_prefix_rec", type=int, default=10)
    parser.add_argument("--n_prefix_conv", type=int, default=10)
    parser.add_argument("--paper_multimodal_fusion", action="store_true")
    parser.add_argument("--multimodal_lambda", type=float, default=0.5)
    parser.add_argument("--inspired_legacy_fusion", action="store_true")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--negative_preference_penalty", type=float, default=1.0)
    parser.add_argument("--scene_top_k", type=int, default=3)
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
    parser.add_argument("--hard_threshold", type=float, default=0.8)
    parser.add_argument("--hard_gate_inference", action="store_true")
    parser.add_argument("--always_on_gate", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hard_gate_inference and args.always_on_gate:
        raise ValueError("gate ablation modes are mutually exclusive")
    if not 0.0 < args.hard_threshold <= 1.0:
        raise ValueError("hard_threshold must lie in (0, 1]")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model).to(device).eval()
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    text_encoder = AutoModel.from_pretrained(args.text_encoder).to(device).eval()
    text_encoder.resize_token_embeddings(len(text_tokenizer))

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    co = Co_occurrence(
        dataset=args.dataset,
        split="train",
        debug=args.debug,
        all_items=kg["item_ids"],
        entity_max_length=args.entity_max_length,
        n_entity=kg["num_entities"],
    ).get_entity_co_info()
    text_graph = text_sim(
        pad_entity_id=kg["pad_entity_id"], dataset=args.dataset
    ).get_entity_ts_info()
    image_graph = image_sim(
        pad_entity_id=kg["pad_entity_id"], dataset=args.dataset
    ).get_entity_is_info()
    prompt_encoder = MMPrompt(
        model.config.n_embd,
        text_encoder.config.hidden_size,
        model.config.n_head,
        model.config.n_layer,
        2,
        n_entity=kg["num_entities"],
        num_relations=kg["num_relations"],
        num_bases=args.num_bases,
        edge_index=kg["edge_index"],
        edge_type=kg["edge_type"],
        edge_index_c=co["edge_index_c"],
        edge_index_t_s=text_graph["edge_index_t_s"],
        edge_index_i_s=image_graph["edge_index_i_s"],
        idx_to_id=text_graph["idx_to_id"],
        n_prefix_rec=args.n_prefix_rec,
        n_prefix_conv=args.n_prefix_conv,
        paper_multimodal_fusion=args.paper_multimodal_fusion,
        multimodal_lambda=args.multimodal_lambda,
        inspired_legacy_fusion=args.inspired_legacy_fusion,
    ).to(device)
    prompt_encoder.load(args.prompt_encoder)
    prompt_encoder.eval()

    tag_asset = torch.load(args.item_tag_file, map_location="cpu")
    scene_asset = torch.load(args.scene_memory_file, map_location="cpu")
    if args.dataset == "redial":
        tag_metrics = tag_asset.get("metrics", {})
        if tag_metrics.get("backend") != "predefined_metadata":
            raise ValueError("ReDial requires predefined metadata attributes")
        if len(tag_asset.get("group_names", [])) < 2:
            raise ValueError("ReDial attributes must retain semantic groups")
        if not tag_metrics.get("category_source_sha256"):
            raise ValueError("ReDial item tags are missing source provenance")
    if tag_asset["item_ids"].tolist() != list(kg["item_ids"]):
        raise ValueError("item-tag candidate order mismatch")
    if scene_asset["item_ids"].tolist() != list(kg["item_ids"]):
        raise ValueError("scene-memory candidate order mismatch")
    scene_assets = {
        key: scene_asset[key]
        for key in [
            "scene_embeddings",
            "scene_tag_profiles",
            "scene_movie_incidence",
            "scene_conversation_ids",
        ]
    }
    sera = EnhancedRecommender(
        hidden_size=model.config.n_embd,
        item_tags=tag_asset["item_tags"],
        tag_group_ids=tag_asset.get("tag_group_ids"),
        group_is_exclusive=tag_asset.get("group_is_exclusive"),
        scene_assets=scene_assets,
        alpha=args.alpha,
        beta=args.beta,
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
    ).to(device)
    checkpoint = args.enhanced_checkpoint
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "enhanced_model.pt")
    sera.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    sera.eval()

    dataset = SERAConvDataset(
        args.dataset,
        "test",
        tokenizer,
        text_tokenizer,
        context_max_length=args.context_max_length,
        response_max_length=args.response_max_length,
        prompt_max_length=args.context_max_length,
        entity_max_length=args.entity_max_length,
        debug=args.debug,
    )
    collator = SERAConvDataCollator(
        tokenizer,
        text_tokenizer,
        kg["pad_entity_id"],
        device,
        generation=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    item_ids = torch.tensor(kg["item_ids"], device=device)
    global_to_local = torch.full(
        (kg["num_entities"],), -1, dtype=torch.long, device=device
    )
    global_to_local[item_ids] = torch.arange(item_ids.numel(), device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream, torch.no_grad():
        for batch in tqdm(loader):
            token_embeds = text_encoder(**batch["prompt"]).last_hidden_state
            base_prompt, _, entity_table = prompt_encoder(
                entity_ids=batch["entity"],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True,
                entity_mask=None if args.inspired_legacy_fusion else batch["entity_mask"],
                return_entity_table=True,
            )
            rec_inputs = dict(batch["context"])
            rec_inputs["prompt_embeds"] = base_prompt
            rec_inputs["entity_embeds"] = entity_table
            rec_output = model(**rec_inputs, rec=True)
            base_logits = rec_output.rec_logits[:, item_ids]
            safe_entities = batch["entity"].clamp(0, global_to_local.shape[0] - 1)
            local_entities = global_to_local[safe_entities]
            sera_output = sera(
                base_logits=base_logits,
                dialogue_rep=rec_output.rec_rep,
                entity_vectors=entity_table[batch["entity"]],
                entity_mask=batch["entity_mask"] & local_entities.ge(0),
                entity_candidate_ids=local_entities,
                positive_entity_mask=batch["positive_entity_mask"],
                negative_entity_mask=batch["negative_entity_mask"],
                conversation_ids=batch["conversation_ids"],
                candidate_item_embeddings=entity_table[item_ids],
                hard_inference=args.hard_gate_inference,
                always_on_gate=args.always_on_gate,
            )
            conditioned_tokens = torch.cat(
                [token_embeds, sera_output.generation_condition.unsqueeze(1)], dim=1
            )
            generation_prompt, _ = prompt_encoder(
                entity_ids=batch["entity"],
                token_embeds=conditioned_tokens,
                output_entity=True,
                use_conv_prefix=True,
                entity_mask=None if args.inspired_legacy_fusion else batch["entity_mask"],
            )
            generated = model.generate(
                **batch["lm"],
                prompt_embeds=generation_prompt,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_width = batch["lm"]["input_ids"].shape[1]
            for identity, reference, sequence, gate in zip(
                batch["identities"],
                batch["references"],
                generated,
                sera_output.effective_gate,
            ):
                prediction = tokenizer.decode(
                    sequence[input_width:], skip_special_tokens=True
                ).strip()
                stream.write(
                    json.dumps(
                        {
                            "identity": identity,
                            "prediction": prediction,
                            "reference": reference,
                            "evidence_gate": float(gate),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
