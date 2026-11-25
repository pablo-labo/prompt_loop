import json
import math
import random
from typing import Any

from storage import LoopStorage, utc_now_iso
from tag_classifier import (
    CLUSTER_DEFS,
    CURATED_EXPANSION_PACKS,
    DEFAULT_CLUSTER_FOR_SLOT,
    PROMPT_OUTPUT_ORDER,
    SLOT_META,
    SLOT_ORDER,
    SLOT_SOFT_CAPS,
    build_seed_lookup,
    classify_tag,
)
from tagger import LocalWDTagger

SEED_CLUSTERS = CLUSTER_DEFS

SEED_TAGS = [
    # camera (17)
    {"tag": "close-up", "slot": "camera", "cluster": "cam_framing"},
    {"tag": "upper body", "slot": "camera", "cluster": "cam_framing"},
    {"tag": "cowboy shot", "slot": "camera", "cluster": "cam_framing"},
    {"tag": "full body", "slot": "camera", "cluster": "cam_framing"},
    {"tag": "wide shot", "slot": "camera", "cluster": "cam_framing"},
    {"tag": "low angle", "slot": "camera", "cluster": "cam_low"},
    {"tag": "from below", "slot": "camera", "cluster": "cam_low"},
    {"tag": "worm's-eye view", "slot": "camera", "cluster": "cam_low"},
    {"tag": "foreshortening", "slot": "camera", "cluster": "cam_low"},
    {"tag": "high angle", "slot": "camera", "cluster": "cam_high"},
    {"tag": "from above", "slot": "camera", "cluster": "cam_high"},
    {"tag": "overhead shot", "slot": "camera", "cluster": "cam_high"},
    {"tag": "dutch angle", "slot": "camera", "cluster": "cam_high"},
    {"tag": "from side", "slot": "camera", "cluster": "cam_orient"},
    {"tag": "from behind", "slot": "camera", "cluster": "cam_orient"},
    {"tag": "looking over shoulder", "slot": "camera", "cluster": "cam_orient"},
    {"tag": "over-the-shoulder shot", "slot": "camera", "cluster": "cam_orient"},
    # pose (17)
    {"tag": "standing", "slot": "pose", "cluster": "pose_stand"},
    {"tag": "contrapposto", "slot": "pose", "cluster": "pose_stand"},
    {"tag": "leaning forward", "slot": "pose", "cluster": "pose_stand"},
    {"tag": "leaning against wall", "slot": "pose", "cluster": "pose_stand"},
    {"tag": "hands behind back", "slot": "pose", "cluster": "pose_stand"},
    {"tag": "hands on hips", "slot": "pose", "cluster": "pose_stand"},
    {"tag": "arms crossed", "slot": "pose", "cluster": "pose_stand"},
    {"tag": "sitting", "slot": "pose", "cluster": "pose_sit_lie"},
    {"tag": "sitting on ledge", "slot": "pose", "cluster": "pose_sit_lie"},
    {"tag": "hugging own legs", "slot": "pose", "cluster": "pose_sit_lie"},
    {"tag": "crossed legs", "slot": "pose", "cluster": "pose_sit_lie"},
    {"tag": "lying on side", "slot": "pose", "cluster": "pose_sit_lie"},
    {"tag": "lying on back", "slot": "pose", "cluster": "pose_sit_lie"},
    {"tag": "dynamic pose", "slot": "pose", "cluster": "pose_dynamic"},
    {"tag": "crouching", "slot": "pose", "cluster": "pose_dynamic"},
    {"tag": "wind-blown hair", "slot": "pose", "cluster": "pose_dynamic"},
    {"tag": "flowing clothes", "slot": "pose", "cluster": "pose_dynamic"},
    {"tag": "hand on cheek", "slot": "pose", "cluster": "pose_hands"},
    {"tag": "finger to mouth", "slot": "pose", "cluster": "pose_hands"},
    {"tag": "adjusting hair", "slot": "pose", "cluster": "pose_hands"},
    {"tag": "reaching out", "slot": "pose", "cluster": "pose_hands"},
    # expression (10)
    {"tag": "looking at viewer", "slot": "expression", "cluster": "expr_gaze"},
    {"tag": "looking away", "slot": "expression", "cluster": "expr_gaze"},
    {"tag": "looking back", "slot": "expression", "cluster": "expr_gaze"},
    {"tag": "glancing sideways", "slot": "expression", "cluster": "expr_gaze"},
    {"tag": "gentle smile", "slot": "expression", "cluster": "expr_mood"},
    {"tag": "parted lips", "slot": "expression", "cluster": "expr_mood"},
    {"tag": "half-closed eyes", "slot": "expression", "cluster": "expr_mood"},
    {"tag": "slight blush", "slot": "expression", "cluster": "expr_mood"},
    {"tag": "serene expression", "slot": "expression", "cluster": "expr_mood"},
    {"tag": "playful smile", "slot": "expression", "cluster": "expr_mood"},
    # lighting (10)
    {"tag": "cinematic lighting", "slot": "lighting", "cluster": "light_dramatic"},
    {"tag": "rim light", "slot": "lighting", "cluster": "light_dramatic"},
    {"tag": "backlighting", "slot": "lighting", "cluster": "light_dramatic"},
    {"tag": "chiaroscuro", "slot": "lighting", "cluster": "light_dramatic"},
    {"tag": "tyndall effect", "slot": "lighting", "cluster": "light_dramatic"},
    {"tag": "golden hour", "slot": "lighting", "cluster": "light_soft"},
    {"tag": "soft lighting", "slot": "lighting", "cluster": "light_soft"},
    {"tag": "sunlight through blinds", "slot": "lighting", "cluster": "light_soft"},
    {"tag": "bokeh", "slot": "lighting", "cluster": "light_soft"},
    {"tag": "depth of field", "slot": "lighting", "cluster": "light_soft"},
]

STOPLIST_FOR_MINING = {
    "1girl",
    "solo",
    "1boy",
    "2girls",
    "multiple girls",
    "multiple boys",
    "masterpiece",
    "best quality",
    "good quality",
    "normal quality",
    "low quality",
    "worst quality",
    "absurdres",
    "highres",
    "very aesthetic",
    "general",
    "sensitive",
    "questionable",
    "explicit",
    "safe",
    "nsfw",
    "score_7",
    "score_8",
    "score_9",
    "score_6",
    "score_5",
    "score_1",
    "score_2",
    "score_3",
    "artist name",
    "watermark",
    "text",
    "signature",
    "english text",
    "username",
    "patreon username",
    "blurry",
    "jpeg artifacts",
}


def _softmax_with_cap(logits: list[float], tau: float, cap: float) -> list[float]:
    n = len(logits)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    scaled = [x / max(tau, 1e-4) for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    probs = [e / total for e in exps]

    if n <= 2:
        return probs
    effective_cap = max(cap, 1.0 / n + 1e-6)
    if effective_cap >= 1.0:
        return probs

    # Water-filling redistribution so max(p) <= effective_cap and sum(p) == 1
    fixed = [False] * n
    for _ in range(n):
        over = [i for i in range(n) if not fixed[i] and probs[i] > effective_cap]
        if not over:
            break
        for i in over:
            probs[i] = effective_cap
            fixed[i] = True
        remaining_mass = 1.0 - sum(probs[i] for i in range(n) if fixed[i])
        free_idx = [i for i in range(n) if not fixed[i]]
        if not free_idx:
            break
        free_sum = sum(probs[i] for i in free_idx)
        if free_sum <= 1e-12:
            share = remaining_mass / len(free_idx)
            for i in free_idx:
                probs[i] = share
        else:
            scale = remaining_mass / free_sum
            for i in free_idx:
                probs[i] *= scale

    s = sum(probs)
    if s > 0:
        probs = [p / s for p in probs]
    return probs


def _weighted_sample_without_replacement(
    items: list[dict[str, Any]],
    probs: list[float],
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    pool_items = list(items)
    pool_probs = list(probs)
    for _ in range(min(k, len(pool_items))):
        total = sum(pool_probs)
        if total <= 1e-12:
            idx = rng.randrange(len(pool_items))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = len(pool_items) - 1
            for j, p in enumerate(pool_probs):
                acc += p
                if r <= acc:
                    idx = j
                    break
        chosen.append(pool_items.pop(idx))
        pool_probs.pop(idx)
    return chosen


def _pair_key(tag_a: str, tag_b: str) -> tuple[str, str]:
    return (tag_a, tag_b) if tag_a <= tag_b else (tag_b, tag_a)


def _compute_pair_metrics(
    eff_count: float,
    rating_sum: float,
    synergy_sum: float,
    pair_prior_m: float = 4.0,
) -> tuple[float, float, float]:
    mean_r = rating_sum / eff_count if eff_count > 1e-6 else 0.0
    synergy = synergy_sum / (eff_count + max(pair_prior_m, 0.5))
    quality = rating_sum / (eff_count + 3.0)
    affinity = max(-1.5, min(1.5, 0.55 * synergy + 0.45 * quality))
    return round(mean_r, 6), round(synergy, 6), round(affinity, 6)


class RLEngine:
    """Slot-normalized RL / Bandit prompt evolution and WD14 difference mining."""

    def __init__(self, storage: LoopStorage, tagger: LocalWDTagger) -> None:
        self.storage = storage
        self.tagger = tagger
        self._seed_lookup = build_seed_lookup(SEED_TAGS)
        self.storage.seed_initial_tags_if_empty(CLUSTER_DEFS, SEED_TAGS)
        self.storage.ensure_clusters(CLUSTER_DEFS)
        if self.storage.get_pair_count() == 0:
            if self.storage.get_rated_generations_chronological():
                self.rebuild_pair_stats()

    @staticmethod
    def _split_prompt_tokens(text: str) -> list[str]:
        return [
            part.strip()
            for part in str(text or "").replace("\n", ",").split(",")
            if part.strip()
        ]

    @staticmethod
    def _parse_prefix_tags(fixed_prefix: str) -> set[str]:
        return {
            part.strip().lower()
            for part in fixed_prefix.replace("\n", ",").split(",")
            if part.strip()
        }

    def _build_auto_negative_prompt(self, config: dict[str, Any]) -> str:
        rl = config["rl"]
        negative_base = str(config.get("negative_base", "")).strip().rstrip(",").strip()
        neg_tags = sorted(
            self.storage.get_all_tags(stage="negative"),
            key=lambda x: (x["weight"], x["tag"]),
        )
        max_auto_neg = int(rl.get("max_auto_negatives", 10))
        auto_neg_list = [t["tag"] for t in neg_tags[:max_auto_neg]]
        neg_parts = [p.strip() for p in [negative_base, *auto_neg_list] if p.strip()]
        return ", ".join(neg_parts)

    def _load_pair_affinity_map(self, min_co_count: int = 2) -> dict[tuple[str, str], float]:
        rows = self.storage.get_all_pairs(min_co_count=min_co_count)
        return {(str(r["tag_a"]), str(r["tag_b"])): float(r["affinity"]) for r in rows}

    def _load_pair_context(
        self, min_co_count: int = 2, min_cluster_co: int = 5
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int], dict[tuple[str, str], float]]:
        rows = self.storage.get_all_pairs(min_co_count=1)
        pair_aff_map: dict[tuple[str, str], float] = {}
        pair_co_map: dict[tuple[str, str], int] = {}
        tag_cluster = {
            str(t["tag"]): str(t.get("cluster") or "discovered_misc")
            for t in self.storage.get_all_tags()
        }
        cl_agg: dict[tuple[str, str], list[float]] = {}

        for r in rows:
            a, b = str(r["tag_a"]), str(r["tag_b"])
            co = int(r["co_count"])
            pk = (a, b)
            pair_co_map[pk] = co
            if co >= min_co_count:
                pair_aff_map[pk] = float(r["affinity"])
            ca = tag_cluster.get(a)
            cb = tag_cluster.get(b)
            if ca and cb:
                cpk = _pair_key(ca, cb)
                bucket = cl_agg.get(cpk)
                if bucket is None:
                    bucket = [0.0, 0.0, 0.0, 0.0]
                    cl_agg[cpk] = bucket
                bucket[0] += float(co)
                bucket[1] += float(r["eff_count"])
                bucket[2] += float(r["rating_sum"])
                bucket[3] += float(r["synergy_sum"])

        cluster_pair_map: dict[tuple[str, str], float] = {}
        for cpk, (co_sum, eff_sum, r_sum, s_sum) in cl_agg.items():
            if co_sum >= min_cluster_co and eff_sum > 1e-6:
                _, _, c_aff = _compute_pair_metrics(eff_sum, r_sum, s_sum, pair_prior_m=6.0)
                cluster_pair_map[cpk] = c_aff
        return pair_aff_map, pair_co_map, cluster_pair_map

    @staticmethod
    def _compute_synergy_bonus(
        tag: str,
        chosen_tags: list[str],
        pair_map: dict[tuple[str, str], float],
    ) -> float:
        if not chosen_tags or not pair_map:
            return 0.0
        vals: list[float] = []
        for other in chosen_tags:
            if other == tag:
                continue
            aff = pair_map.get(_pair_key(tag, other))
            if aff is not None and abs(aff) > 1e-6:
                vals.append(aff)
        if not vals:
            return 0.0
        raw = sum(vals) / math.sqrt(len(vals))
        return max(-1.2, min(1.2, raw))

    @staticmethod
    def _compute_synergy_terms(
        tag: str,
        cluster: str,
        stage: str,
        z_score: float,
        chosen_items: list[dict[str, Any]],
        pair_aff_map: dict[tuple[str, str], float],
        pair_co_map: dict[tuple[str, str], int] | None = None,
        cluster_pair_map: dict[tuple[str, str], float] | None = None,
    ) -> tuple[float, float]:
        if not chosen_items:
            return 0.0, 0.0
        syn_vals: list[float] = []
        explore_hits = 0
        tag_strong = (stage == "core") or (z_score >= 0.2)

        for other in chosen_items:
            otag = str(other.get("tag", ""))
            if not otag or otag == tag:
                continue
            pk = _pair_key(tag, otag)
            co = (
                pair_co_map.get(pk, 0)
                if pair_co_map is not None
                else (2 if pk in pair_aff_map else 0)
            )
            if pk in pair_aff_map:
                syn_vals.append(pair_aff_map[pk])
            elif cluster_pair_map and cluster and other.get("cluster"):
                cpk = _pair_key(cluster, str(other["cluster"]))
                c_aff = cluster_pair_map.get(cpk)
                if c_aff is not None and abs(c_aff) > 1e-6:
                    syn_vals.append(0.40 * c_aff)

            if (
                co == 0
                and tag_strong
                and (other.get("stage") == "core" or float(other.get("z_score", 0.0)) >= 0.2)
            ):
                explore_hits += 1

        syn_bonus = (
            max(-1.2, min(1.2, sum(syn_vals) / math.sqrt(len(syn_vals))))
            if syn_vals
            else 0.0
        )
        pair_ucb = (
            min(1.0, explore_hits / math.sqrt(max(1, len(chosen_items))))
            if explore_hits > 0
            else 0.0
        )
        return syn_bonus, pair_ucb

    def _sample_chained_tags(
        self,
        candidates: list[dict[str, Any]],
        target_k: int,
        current_round: int,
        rl: dict[str, Any],
        tau: float,
        batch_seen_tags: set[str],
        rng: random.Random,
        initial_chosen: list[dict[str, Any]] | None = None,
        slot_caps: dict[str, int] | None = None,
        allowed_slots: set[str] | None = None,
        explore_quota: int = 0,
        prefer_core_only: bool = False,
        pair_aff_map: dict[tuple[str, str], float] | None = None,
        pair_co_map: dict[tuple[str, str], int] | None = None,
        cluster_pair_map: dict[tuple[str, str], float] | None = None,
        synergy_weight: float = 0.0,
        pair_explore_beta: float = 0.0,
        batch_div_penalty: float = 2.0,
        source_tag: str = "auto",
        explore_source_tag: str | None = None,
        prefer_uncovered_slots: bool = False,
    ) -> list[dict[str, Any]]:
        if not candidates or target_k <= 0:
            return []
        ucb_beta = float(rl.get("ucb_beta", 0.45))
        cd_rounds = int(rl.get("cooldown_rounds", 3))
        cd_penalty = float(rl.get("cooldown_penalty", 0.45))
        max_prob_cap = float(rl.get("max_prob_cap", 0.45))

        caps = dict(slot_caps or SLOT_SOFT_CAPS)
        chosen_items: list[dict[str, Any]] = [dict(x) for x in (initial_chosen or [])]
        chosen_lower = {str(x.get("tag", "")).strip().lower() for x in chosen_items}
        slot_used: dict[str, int] = {}
        for x in chosen_items:
            sl = str(x.get("slot", "discovered"))
            slot_used[sl] = slot_used.get(sl, 0) + 1

        pool = [
            dict(c)
            for c in candidates
            if str(c["tag"]).strip().lower() not in chosen_lower
            and (allowed_slots is None or c["slot"] in allowed_slots)
        ]
        if prefer_core_only:
            core_pool = [c for c in pool if c["stage"] == "core"]
            if len(core_pool) >= target_k:
                pool = core_pool

        newly_picked: list[dict[str, Any]] = []
        explore_rem = max(0, int(explore_quota))
        pair_aff = pair_aff_map or {}

        for step in range(target_k):
            eligible = [
                c
                for c in pool
                if slot_used.get(c["slot"], 0) < caps.get(c["slot"], 2)
            ]
            if not eligible:
                break

            rem_steps = target_k - step
            explore_eligible = [
                c for c in eligible if c["stage"] == "explore" or int(c["active_count"]) < 4
            ]
            is_explore_step = False
            if explore_rem > 0 and explore_eligible:
                if rem_steps <= explore_rem or rng.random() < (explore_rem / rem_steps):
                    is_explore_step = True

            if is_explore_step:
                step_pool = explore_eligible
            elif prefer_uncovered_slots:
                uncovered = [c for c in eligible if slot_used.get(c["slot"], 0) == 0]
                step_pool = uncovered if uncovered else eligible
            else:
                step_pool = eligible

            logits: list[float] = []
            syn_list: list[float] = []
            for c in step_pool:
                ucb = ucb_beta * math.sqrt(
                    math.log(max(current_round, 1) + 1.0) / (int(c["active_count"]) + 1.0)
                )
                cd = 0.0
                if int(c["last_used_round"]) > 0 and (current_round - int(c["last_used_round"])) < cd_rounds:
                    cd = cd_penalty * max(1, int(c["consecutive_uses"]))
                explore_nudge = (
                    0.45 if (c["stage"] == "explore" and int(c["active_count"]) < 4) else 0.0
                )
                batch_div = batch_div_penalty if c["tag"] in batch_seen_tags else 0.0

                syn_bonus = 0.0
                pair_ucb = 0.0
                if chosen_items and (synergy_weight > 0.0 or pair_explore_beta > 0.0):
                    syn_bonus, pair_ucb = self._compute_synergy_terms(
                        tag=str(c["tag"]),
                        cluster=str(c.get("cluster", "")),
                        stage=str(c.get("stage", "explore")),
                        z_score=float(c.get("z_score", 0.0)),
                        chosen_items=chosen_items,
                        pair_aff_map=pair_aff,
                        pair_co_map=pair_co_map,
                        cluster_pair_map=cluster_pair_map,
                    )
                syn_list.append(syn_bonus)
                logits.append(
                    float(c["z_score"])
                    + ucb
                    + explore_nudge
                    - cd
                    - batch_div
                    + synergy_weight * syn_bonus
                    + pair_explore_beta * pair_ucb
                )

            probs = _softmax_with_cap(logits, tau=tau, cap=max_prob_cap)
            one = _weighted_sample_without_replacement(step_pool, probs, 1, rng)[0]
            chosen_idx = next(i for i, item in enumerate(step_pool) if item["tag"] == one["tag"])
            pool = [item for item in pool if item["tag"] != one["tag"]]
            chosen_item = dict(step_pool[chosen_idx])
            chosen_item["_syn_bonus"] = round(float(syn_list[chosen_idx]), 3)
            chosen_item["_picked_source"] = (
                explore_source_tag if (is_explore_step and explore_source_tag) else source_tag
            )
            slot_used[chosen_item["slot"]] = slot_used.get(chosen_item["slot"], 0) + 1
            chosen_items.append(chosen_item)
            newly_picked.append(chosen_item)
            if is_explore_step:
                explore_rem -= 1

        return newly_picked

    def _sample_slot_tags(
        self,
        candidates: list[dict[str, Any]],
        k: int,
        current_round: int,
        rl: dict[str, Any],
        tau: float,
        batch_seen_tags: set[str],
        rng: random.Random,
        chosen_tags: list[str] | None = None,
        pair_map: dict[tuple[str, str], float] | None = None,
        synergy_weight: float = 0.0,
    ) -> list[dict[str, Any]]:
        init = [{"tag": t, "slot": "_ext", "cluster": ""} for t in (chosen_tags or [])]
        picked = self._sample_chained_tags(
            candidates=candidates,
            target_k=k,
            current_round=current_round,
            rl=rl,
            tau=tau,
            batch_seen_tags=batch_seen_tags,
            rng=rng,
            initial_chosen=init,
            slot_caps={s: max(k, 8) for s in SLOT_ORDER},
            pair_aff_map=pair_map,
            synergy_weight=synergy_weight,
        )
        picked.sort(key=lambda x: (float(x["z_score"]), float(x["weight"])), reverse=True)
        return picked

    def _prepare_manual_prompt_parts(
        self,
        config: dict[str, Any],
        positive_prompt: str,
        negative_prompt: str,
        include_prefix: bool,
    ) -> tuple[list[str], list[str], str, list[dict[str, Any]]]:
        user_tokens = self._split_prompt_tokens(positive_prompt)
        prefix_tokens = (
            self._split_prompt_tokens(str(config.get("fixed_prefix", "")))
            if include_prefix
            else []
        )
        prefix_lower = {t.lower() for t in prefix_tokens}

        seen_body_lower: set[str] = set()
        body_tokens: list[str] = []
        for tok in user_tokens:
            low = tok.lower()
            if low in prefix_lower or low in seen_body_lower:
                continue
            seen_body_lower.add(low)
            body_tokens.append(tok)

        neg_tokens = self._split_prompt_tokens(negative_prompt)
        final_negative = (
            ", ".join(neg_tokens)
            if neg_tokens
            else self._build_auto_negative_prompt(config)
        )

        by_lower = {
            str(t["tag"]).strip().lower(): t for t in self.storage.get_all_tags()
        }
        pinned_meta: list[dict[str, Any]] = []
        for tok in body_tokens:
            t_row = by_lower.get(tok.lower())
            if t_row is not None and t_row["stage"] in ("core", "explore", "staging"):
                pinned_meta.append(
                    {
                        "tag": t_row["tag"],
                        "slot": t_row["slot"],
                        "cluster": t_row["cluster"],
                        "stage": t_row["stage"],
                        "weight": round(float(t_row["weight"]), 4),
                        "z_score": round(float(t_row["z_score"]), 4),
                        "position": len(pinned_meta) + 1,
                        "source": "manual",
                    }
                )

        return prefix_tokens, body_tokens, final_negative, pinned_meta

    @staticmethod
    def _prompt_sort_key(item: dict[str, Any]) -> tuple[int, float, float]:
        slot_rank = PROMPT_OUTPUT_ORDER.index(item["slot"]) if item.get("slot") in PROMPT_OUTPUT_ORDER else 99
        return (slot_rank, -float(item.get("z_score", 0.0)), -float(item.get("weight", 0.0)))

    def _finalize_sampled_meta(
        self,
        items: list[dict[str, Any]],
        default_source: str = "auto",
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            out.append(
                {
                    "tag": item["tag"],
                    "slot": item["slot"],
                    "cluster": item["cluster"],
                    "stage": item["stage"],
                    "weight": round(float(item["weight"]), 4),
                    "z_score": round(float(item["z_score"]), 4),
                    "synergy_bonus": round(float(item.get("_syn_bonus", 0.0)), 3),
                    "position": idx,
                    "source": str(item.get("_picked_source") or item.get("source") or default_source),
                }
            )
        return out

    def sample_batch_prompts(
        self,
        config: dict[str, Any],
        batch_size_override: int | None = None,
        force_contrast: bool = False,
        same_seed_override: bool | None = None,
        variant_count: int | None = None,
        allowed_variant_slots: list[str] | None = None,
        custom_prefix_tokens: list[str] | None = None,
        custom_negative: str | None = None,
        include_prefix: bool = True,
    ) -> list[dict[str, Any]]:
        raw_bs = (
            batch_size_override
            if batch_size_override is not None
            else config.get("batch_size", 2)
        )
        batch_size = max(1, min(4, int(raw_bs)))
        rl = config["rl"]
        prefix_tokens = (
            list(custom_prefix_tokens)
            if custom_prefix_tokens is not None
            else (self._split_prompt_tokens(str(config.get("fixed_prefix", ""))) if include_prefix else [])
        )
        prefix_lower = {t.lower() for t in prefix_tokens}

        rng = random.Random()
        current_round = int(self.storage.get_meta("round_num", "0")) + 1
        all_active = [
            t
            for t in self.storage.get_all_tags()
            if t["stage"] in ("core", "explore")
            and t["tag"].strip().lower() not in prefix_lower
        ]
        if not all_active:
            raise ValueError("当前词池没有可用的活跃词（core/explore）")

        negative_prompt = (
            custom_negative
            if custom_negative is not None and custom_negative.strip()
            else self._build_auto_negative_prompt(config)
        )

        progress = min(1.0, current_round / max(1, int(rl.get("temp_decay_rounds", 40))))
        tau = float(rl.get("temp_start", 1.2)) - progress * (
            float(rl.get("temp_start", 1.2)) - float(rl.get("temp_min", 0.55))
        )
        dropout_prob = float(rl.get("dropout_prob", 0.25))
        contrast_prob = float(rl.get("contrast_prob", 0.50))
        contrast_same_seed = bool(rl.get("contrast_same_seed", True))
        target_min = max(2, min(12, int(rl.get("target_tag_min", 5))))
        target_max = max(target_min, min(14, int(rl.get("target_tag_max", 8))))
        explore_wip_limit = max(1, int(rl.get("explore_wip_limit", 25)))
        explore_slots_cfg = max(0, min(3, int(rl.get("explore_slots_per_img", 1))))
        explore_pool_cnt = sum(1 for t in all_active if t["stage"] == "explore")
        if explore_pool_cnt > explore_wip_limit and explore_slots_cfg > 0:
            explore_slots_cfg = max(explore_slots_cfg, 2)

        synergy_weight = float(rl.get("synergy_weight", 0.55))
        pair_explore_beta = float(rl.get("pair_explore_beta", 0.25))
        pair_aff_map, pair_co_map, cluster_pair_map = self._load_pair_context(
            min_co_count=2, min_cluster_co=5
        )

        allowed_var_set = (
            {str(s).strip() for s in allowed_variant_slots if str(s).strip() in SLOT_ORDER}
            if allowed_variant_slots is not None
            else None
        )
        if allowed_var_set is not None and len(allowed_var_set) == 0:
            allowed_var_set = None

        use_contrast = force_contrast or (
            batch_size >= 2 and rng.random() < contrast_prob
        )
        results: list[dict[str, Any]] = []

        if use_contrast:
            total_k = rng.randint(target_min, target_max)
            var_k = (
                max(1, min(3, int(variant_count)))
                if variant_count is not None
                else (1 if total_k <= 6 else rng.choice([1, 2]))
            )
            skel_k = max(2, min(total_k - var_k, len(all_active)))
            skeleton_items = self._sample_chained_tags(
                candidates=all_active,
                target_k=skel_k,
                current_round=current_round,
                rl=rl,
                tau=tau,
                batch_seen_tags=set(),
                rng=rng,
                slot_caps=SLOT_SOFT_CAPS,
                explore_quota=0,
                prefer_core_only=True,
                pair_aff_map=pair_aff_map,
                pair_co_map=pair_co_map,
                cluster_pair_map=cluster_pair_map,
                synergy_weight=synergy_weight,
                pair_explore_beta=pair_explore_beta,
                batch_div_penalty=0.0,
                source_tag="skeleton",
            )
            use_same_seed = (
                contrast_same_seed
                if same_seed_override is None
                else bool(same_seed_override)
            )
            shared_seed = rng.randint(1, 2**31 - 1) if use_same_seed else None
            batch_seen_variants: set[str] = set()

            for slot_index in range(batch_size):
                var_explore_q = min(var_k, explore_slots_cfg, explore_pool_cnt)
                variants = self._sample_chained_tags(
                    candidates=all_active,
                    target_k=var_k,
                    current_round=current_round,
                    rl=rl,
                    tau=tau,
                    batch_seen_tags=batch_seen_variants,
                    rng=rng,
                    initial_chosen=skeleton_items,
                    slot_caps=SLOT_SOFT_CAPS,
                    allowed_slots=allowed_var_set,
                    explore_quota=var_explore_q,
                    prefer_core_only=False,
                    pair_aff_map=pair_aff_map,
                    pair_co_map=pair_co_map,
                    cluster_pair_map=cluster_pair_map,
                    synergy_weight=synergy_weight,
                    pair_explore_beta=max(pair_explore_beta, 0.30),
                    batch_div_penalty=6.0,
                    source_tag="variant",
                    explore_source_tag="variant",
                )
                if not variants and allowed_var_set is not None:
                    # Retry without soft caps if skeleton filled the chosen variant slot
                    variants = self._sample_chained_tags(
                        candidates=all_active,
                        target_k=var_k,
                        current_round=current_round,
                        rl=rl,
                        tau=tau,
                        batch_seen_tags=batch_seen_variants,
                        rng=rng,
                        initial_chosen=skeleton_items,
                        slot_caps={s: 8 for s in SLOT_ORDER},
                        allowed_slots=allowed_var_set,
                        explore_quota=var_explore_q,
                        prefer_core_only=False,
                        pair_aff_map=pair_aff_map,
                        pair_co_map=pair_co_map,
                        cluster_pair_map=cluster_pair_map,
                        synergy_weight=synergy_weight,
                        pair_explore_beta=max(pair_explore_beta, 0.30),
                        batch_div_penalty=6.0,
                        source_tag="variant",
                        explore_source_tag="variant",
                    )
                for v in variants:
                    batch_seen_variants.add(v["tag"])

                combined = [dict(x) for x in skeleton_items] + [dict(v) for v in variants]
                combined.sort(key=self._prompt_sort_key)
                prompt_chunks = [*prefix_tokens, *[item["tag"] for item in combined]]
                sampled_meta = self._finalize_sampled_meta(combined, default_source="skeleton")
                seed = shared_seed if shared_seed is not None else rng.randint(1, 2**31 - 1)
                results.append(
                    {
                        "slot_index": slot_index,
                        "seed": seed,
                        "is_dropout_round": False,
                        "positive_prompt": ", ".join(prompt_chunks),
                        "negative_prompt": negative_prompt,
                        "sampled_tags": sampled_meta,
                    }
                )
            return results

        # Scheme A free full-pool chained sampling
        batch_seen_tags: set[str] = set()
        for slot_index in range(batch_size):
            is_dropout = rng.random() < dropout_prob
            if is_dropout:
                target_k = min(rng.choice([2, 3]), len(all_active))
                explore_q = 1 if (explore_pool_cnt > 0 and rng.random() < 0.5) else 0
            else:
                target_k = min(rng.randint(target_min, target_max), len(all_active))
                explore_q = min(explore_slots_cfg, max(0, target_k - 1), explore_pool_cnt)

            picked = self._sample_chained_tags(
                candidates=all_active,
                target_k=target_k,
                current_round=current_round,
                rl=rl,
                tau=tau,
                batch_seen_tags=batch_seen_tags,
                rng=rng,
                slot_caps=SLOT_SOFT_CAPS,
                explore_quota=explore_q,
                prefer_core_only=False,
                pair_aff_map=pair_aff_map,
                pair_co_map=pair_co_map,
                cluster_pair_map=cluster_pair_map,
                synergy_weight=synergy_weight,
                pair_explore_beta=pair_explore_beta,
                batch_div_penalty=2.0,
                source_tag="auto",
                explore_source_tag="explore_slot",
            )
            for item in picked:
                batch_seen_tags.add(item["tag"])
            picked.sort(key=self._prompt_sort_key)
            prompt_chunks = [*prefix_tokens, *[item["tag"] for item in picked]]
            sampled_meta = self._finalize_sampled_meta(picked, default_source="auto")
            results.append(
                {
                    "slot_index": slot_index,
                    "seed": rng.randint(1, 2**31 - 1),
                    "is_dropout_round": is_dropout,
                    "positive_prompt": ", ".join(prompt_chunks),
                    "negative_prompt": negative_prompt,
                    "sampled_tags": sampled_meta,
                }
            )
        return results

    def build_contrast_batch_prompts(
        self,
        config: dict[str, Any],
        positive_prompt: str = "",
        negative_prompt: str = "",
        include_prefix: bool = True,
        batch_size: int | None = None,
        roll_slots: list[str] | None = None,
        same_seed: bool = True,
        variant_count: int | None = None,
    ) -> list[dict[str, Any]]:
        prefix_tokens, body_tokens, final_negative, pinned_meta = (
            self._prepare_manual_prompt_parts(
                config=config,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                include_prefix=include_prefix,
            )
        )
        if not body_tokens:
            return self.sample_batch_prompts(
                config,
                batch_size_override=batch_size,
                force_contrast=True,
                same_seed_override=same_seed,
                variant_count=variant_count,
                allowed_variant_slots=roll_slots,
                custom_prefix_tokens=prefix_tokens if include_prefix else [],
                custom_negative=final_negative,
                include_prefix=include_prefix,
            )

        rl = config["rl"]
        rng = random.Random()
        count = max(
            1,
            min(4, int(batch_size if batch_size is not None else config.get("batch_size", 2))),
        )
        var_k = max(1, min(3, int(variant_count if variant_count is not None else 1)))
        current_round = int(self.storage.get_meta("round_num", "0")) + 1
        progress = min(1.0, current_round / max(1, int(rl.get("temp_decay_rounds", 40))))
        tau = float(rl.get("temp_start", 1.2)) - progress * (
            float(rl.get("temp_start", 1.2)) - float(rl.get("temp_min", 0.55))
        )
        synergy_weight = float(rl.get("synergy_weight", 0.55))
        pair_explore_beta = float(rl.get("pair_explore_beta", 0.25))
        pair_aff_map, pair_co_map, cluster_pair_map = self._load_pair_context(
            min_co_count=2, min_cluster_co=5
        )

        pinned_lower = {t.lower() for t in (*prefix_tokens, *body_tokens)}
        allowed_set = (
            {str(s).strip() for s in roll_slots if str(s).strip() in SLOT_ORDER}
            if roll_slots is not None
            else None
        )
        if allowed_set is not None and len(allowed_set) == 0:
            allowed_set = None

        all_active = [
            t
            for t in self.storage.get_all_tags()
            if t["stage"] in ("core", "explore")
            and t["tag"].strip().lower() not in pinned_lower
        ]
        skeleton_meta = [{**m, "source": "skeleton"} for m in pinned_meta]
        shared_seed = rng.randint(1, 2**31 - 1) if same_seed else None
        batch_seen_variants: set[str] = set()
        results: list[dict[str, Any]] = []

        for slot_index in range(count):
            variants = self._sample_chained_tags(
                candidates=all_active,
                target_k=var_k,
                current_round=current_round,
                rl=rl,
                tau=tau,
                batch_seen_tags=batch_seen_variants,
                rng=rng,
                initial_chosen=skeleton_meta,
                slot_caps=SLOT_SOFT_CAPS,
                allowed_slots=allowed_set,
                explore_quota=1 if any(c["stage"] == "explore" for c in all_active) else 0,
                pair_aff_map=pair_aff_map,
                pair_co_map=pair_co_map,
                cluster_pair_map=cluster_pair_map,
                synergy_weight=synergy_weight,
                pair_explore_beta=max(pair_explore_beta, 0.30),
                batch_div_penalty=6.0,
                source_tag="variant",
                explore_source_tag="variant",
            )
            if not variants and all_active:
                variants = self._sample_chained_tags(
                    candidates=all_active,
                    target_k=var_k,
                    current_round=current_round,
                    rl=rl,
                    tau=tau,
                    batch_seen_tags=batch_seen_variants,
                    rng=rng,
                    initial_chosen=skeleton_meta,
                    slot_caps={s: 8 for s in SLOT_ORDER},
                    allowed_slots=allowed_set,
                    pair_aff_map=pair_aff_map,
                    pair_co_map=pair_co_map,
                    cluster_pair_map=cluster_pair_map,
                    synergy_weight=synergy_weight,
                    pair_explore_beta=max(pair_explore_beta, 0.30),
                    batch_div_penalty=6.0,
                    source_tag="variant",
                    explore_source_tag="variant",
                )
            for v in variants:
                batch_seen_variants.add(v["tag"])

            variants.sort(key=self._prompt_sort_key)
            prompt_chunks = [*prefix_tokens, *body_tokens, *[v["tag"] for v in variants]]
            sampled_meta = [dict(m) for m in skeleton_meta]
            for idx, v in enumerate(variants, start=len(skeleton_meta) + 1):
                sampled_meta.append(
                    {
                        "tag": v["tag"],
                        "slot": v["slot"],
                        "cluster": v["cluster"],
                        "stage": v["stage"],
                        "weight": round(float(v["weight"]), 4),
                        "z_score": round(float(v["z_score"]), 4),
                        "synergy_bonus": round(float(v.get("_syn_bonus", 0.0)), 3),
                        "position": idx,
                        "source": "variant",
                    }
                )
            results.append(
                {
                    "slot_index": slot_index,
                    "seed": shared_seed if shared_seed is not None else rng.randint(1, 2**31 - 1),
                    "is_dropout_round": False,
                    "positive_prompt": ", ".join(prompt_chunks),
                    "negative_prompt": final_negative,
                    "sampled_tags": sampled_meta,
                }
            )
        return results

    def build_manual_batch_prompts(
        self,
        config: dict[str, Any],
        positive_prompt: str,
        negative_prompt: str = "",
        include_prefix: bool = True,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self._split_prompt_tokens(positive_prompt):
            raise ValueError("不 Roll 模式下请至少输入或从词池点选一个正向 Tag")

        prefix_tokens, body_tokens, final_negative, pinned_meta = (
            self._prepare_manual_prompt_parts(
                config=config,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                include_prefix=include_prefix,
            )
        )
        chunks = [*prefix_tokens, *body_tokens] if include_prefix else list(body_tokens)
        final_positive = ", ".join(chunks)
        if not final_positive:
            raise ValueError("正向 Prompt 不能为空")

        count = max(
            1,
            min(4, int(batch_size if batch_size is not None else config.get("batch_size", 2))),
        )
        rng = random.Random()
        results: list[dict[str, Any]] = []
        for slot_index in range(count):
            results.append(
                {
                    "slot_index": slot_index,
                    "seed": rng.randint(1, 2**31 - 1),
                    "is_dropout_round": False,
                    "positive_prompt": final_positive,
                    "negative_prompt": final_negative,
                    "sampled_tags": [dict(m) for m in pinned_meta],
                }
            )
        return results

    def build_hybrid_batch_prompts(
        self,
        config: dict[str, Any],
        positive_prompt: str,
        negative_prompt: str = "",
        include_prefix: bool = True,
        batch_size: int | None = None,
        roll_slots: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        prefix_tokens, body_tokens, final_negative, pinned_meta = (
            self._prepare_manual_prompt_parts(
                config=config,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                include_prefix=include_prefix,
            )
        )
        covered_slots = {m["slot"] for m in pinned_meta}
        pinned_lower = {t.lower() for t in (*prefix_tokens, *body_tokens)}

        if roll_slots is None:
            target_roll_slots = [s for s in SLOT_ORDER if s not in covered_slots]
        else:
            req_set = {str(s).strip() for s in roll_slots}
            target_roll_slots = [
                s for s in SLOT_ORDER if s in req_set and s not in covered_slots
            ]

        if not body_tokens and not target_roll_slots:
            raise ValueError("请至少输入/点选标签，或勾选要补全 Roll 的槽位")

        rl = config["rl"]
        rng = random.Random()
        current_round = int(self.storage.get_meta("round_num", "0")) + 1
        progress = min(1.0, current_round / max(1, int(rl.get("temp_decay_rounds", 40))))
        tau = float(rl.get("temp_start", 1.2)) - progress * (
            float(rl.get("temp_start", 1.2)) - float(rl.get("temp_min", 0.55))
        )

        candidates = [
            t
            for t in self.storage.get_all_tags()
            if t["stage"] in ("core", "explore")
            and t["slot"] in set(target_roll_slots)
            and t["tag"].strip().lower() not in pinned_lower
        ]
        available_slots = {t["slot"] for t in candidates}
        active_roll_slots = [s for s in target_roll_slots if s in available_slots]

        count = max(
            1,
            min(4, int(batch_size if batch_size is not None else config.get("batch_size", 2))),
        )
        synergy_weight = float(rl.get("synergy_weight", 0.55))
        pair_explore_beta = float(rl.get("pair_explore_beta", 0.25))
        pair_aff_map, pair_co_map, cluster_pair_map = self._load_pair_context(
            min_co_count=2, min_cluster_co=5
        )
        target_min = max(2, min(12, int(rl.get("target_tag_min", 5))))
        target_max = max(target_min, min(14, int(rl.get("target_tag_max", 8))))

        batch_seen_tags: set[str] = set()
        results: list[dict[str, Any]] = []

        for slot_index in range(count):
            prompt_chunks = [*prefix_tokens, *body_tokens]
            sampled_meta = [dict(m) for m in pinned_meta]
            if active_roll_slots:
                desired_total = rng.randint(target_min, target_max)
                if len(active_roll_slots) <= 3:
                    max_cap_sum = sum(SLOT_SOFT_CAPS.get(s, 2) for s in active_roll_slots)
                    rem_k = min(
                        max_cap_sum,
                        max(len(active_roll_slots), desired_total - len(body_tokens)),
                    )
                else:
                    rem_k = max(2, desired_total - len(body_tokens))
                explore_avail = any(t["stage"] == "explore" for t in candidates)
                picked = self._sample_chained_tags(
                    candidates=candidates,
                    target_k=rem_k,
                    current_round=current_round,
                    rl=rl,
                    tau=tau,
                    batch_seen_tags=batch_seen_tags,
                    rng=rng,
                    initial_chosen=pinned_meta,
                    slot_caps=SLOT_SOFT_CAPS,
                    allowed_slots=available_slots,
                    explore_quota=1 if (explore_avail and rem_k >= 2) else 0,
                    pair_aff_map=pair_aff_map,
                    pair_co_map=pair_co_map,
                    cluster_pair_map=cluster_pair_map,
                    synergy_weight=synergy_weight,
                    pair_explore_beta=pair_explore_beta,
                    batch_div_penalty=2.0,
                    source_tag="rolled",
                    explore_source_tag="rolled",
                    prefer_uncovered_slots=True,
                )
                picked.sort(key=self._prompt_sort_key)
                for item in picked:
                    batch_seen_tags.add(item["tag"])
                    prompt_chunks.append(item["tag"])
                    sampled_meta.append(
                        {
                            "tag": item["tag"],
                            "slot": item["slot"],
                            "cluster": item["cluster"],
                            "stage": item["stage"],
                            "weight": round(float(item["weight"]), 4),
                            "z_score": round(float(item["z_score"]), 4),
                            "synergy_bonus": round(float(item.get("_syn_bonus", 0.0)), 3),
                            "position": len(sampled_meta) + 1,
                            "source": "rolled",
                        }
                    )

            final_positive = ", ".join(prompt_chunks)
            if not final_positive:
                raise ValueError("当前所选槽位没有可用的活跃词，请补充自选 Tag 或调整词池")
            results.append(
                {
                    "slot_index": slot_index,
                    "seed": rng.randint(1, 2**31 - 1),
                    "is_dropout_round": False,
                    "positive_prompt": final_positive,
                    "negative_prompt": final_negative,
                    "sampled_tags": sampled_meta,
                }
            )

        return results

    def compute_emergent_tags(
        self,
        positive_prompt: str,
        sampled_tags: list[dict[str, Any]],
        detected_tags: list[dict[str, Any]],
        fixed_prefix: str,
    ) -> list[dict[str, Any]]:
        input_set = self._parse_prefix_tags(fixed_prefix)
        for item in sampled_tags:
            input_set.add(item["tag"].strip().lower())
        for part in positive_prompt.replace("\n", ",").split(","):
            clean = part.strip().lower()
            if clean:
                input_set.add(clean)

        emergent: list[dict[str, Any]] = []
        for dt in detected_tags:
            name_lower = dt["tag"].strip().lower()
            if (
                dt["score"] >= 0.35
                and dt.get("category") == "general"
                and name_lower not in input_set
                and name_lower not in STOPLIST_FOR_MINING
            ):
                emergent.append(
                    {
                        "tag": dt["tag"],
                        "raw_tag": dt.get("raw_tag", dt["tag"]),
                        "score": dt["score"],
                    }
                )
        return emergent

    def _recompute_weights_and_zscores(
        self,
        conn: Any,
        rl: dict[str, Any],
    ) -> None:
        prior_m = float(rl.get("prior_m", 3.0))
        cluster_alpha = float(rl.get("cluster_prior_alpha", 0.25))

        clusters = self.storage.get_all_clusters(conn=conn)
        cluster_weights: dict[str, float] = {}
        for cl in clusters:
            w = float(cl["reward_sum"]) / (float(cl["count"]) + prior_m)
            cluster_weights[cl["cluster"]] = w
            conn.execute(
                "UPDATE cluster_stats SET weight = ? WHERE cluster = ?",
                (round(w, 6), cl["cluster"]),
            )

        tags = self.storage.get_all_tags(conn=conn)
        slot_groups: dict[str, list[dict[str, Any]]] = {s: [] for s in SLOT_ORDER}
        now = utc_now_iso()

        for t in tags:
            cl_w = cluster_weights.get(t["cluster"], 0.0)
            w = (float(t["active_reward_sum"]) + prior_m * cluster_alpha * cl_w) / (
                float(t["active_count"]) + prior_m
            )
            t["weight"] = w

            if t["stage"] in ("core", "explore") and t["slot"] in slot_groups:
                slot_groups[t["slot"]].append(t)
            else:
                conn.execute(
                    "UPDATE tag_stats SET weight = ?, z_score = 0.0, updated_at = ? WHERE tag = ?",
                    (round(w, 6), now, t["tag"]),
                )

        def _assign_zscores(items: list[dict[str, Any]]) -> None:
            if not items:
                return
            weights = [float(x["weight"]) for x in items]
            mean_w = sum(weights) / len(weights)
            var = sum((w - mean_w) ** 2 for w in weights) / len(weights)
            std_w = math.sqrt(var)
            for item in items:
                if std_w < 1e-6:
                    item["z_score"] = 0.0
                else:
                    z = (float(item["weight"]) - mean_w) / std_w
                    item["z_score"] = max(-3.0, min(3.0, z))

        for slot, items in slot_groups.items():
            if not items:
                continue
            _assign_zscores(items)

            kept_items: list[dict[str, Any]] = []
            demoted = False
            for item in items:
                stage = item["stage"]
                w = float(item["weight"])
                z = float(item["z_score"])
                if stage == "explore" and int(item["active_count"]) >= 4:
                    if z >= 0.50 or w >= 0.08:
                        stage = "core"
                    elif (
                        (z <= -1.50 and w <= -0.05)
                        or w <= -0.15
                        or (int(item["active_count"]) >= 8 and z < 0.15 and w < 0.02)
                    ):
                        stage = "staging"
                item["stage"] = stage

                if stage == "staging":
                    demoted = True
                    conn.execute(
                        "UPDATE tag_stats SET weight = ?, z_score = 0.0, stage = ?, updated_at = ? WHERE tag = ?",
                        (round(w, 6), stage, now, item["tag"]),
                    )
                else:
                    kept_items.append(item)

            if demoted:
                _assign_zscores(kept_items)

            for item in kept_items:
                conn.execute(
                    "UPDATE tag_stats SET weight = ?, z_score = ?, stage = ?, updated_at = ? WHERE tag = ?",
                    (
                        round(float(item["weight"]), 6),
                        round(float(item["z_score"]), 6),
                        item["stage"],
                        now,
                        item["tag"],
                    ),
                )

    def recompute_all(self, config: dict[str, Any]) -> None:
        rl = config["rl"]
        with self.storage.transaction() as conn:
            self._recompute_weights_and_zscores(conn, rl)

    @staticmethod
    def _upsert_pair_observations(
        conn: Any,
        pair_observations: list[tuple[str, str, float, float, float]],
        pair_prior_m: float,
        now: str,
    ) -> None:
        for tag_a, tag_b, eff_inc, r_inc, syn_inc in pair_observations:
            row = conn.execute(
                "SELECT co_count, eff_count, rating_sum, synergy_sum "
                "FROM tag_pair_stats WHERE tag_a = ? AND tag_b = ?",
                (tag_a, tag_b),
            ).fetchone()
            new_co = (int(row["co_count"]) if row else 0) + 1
            new_eff = (float(row["eff_count"]) if row else 0.0) + eff_inc
            new_rsum = (float(row["rating_sum"]) if row else 0.0) + r_inc
            new_ssum = (float(row["synergy_sum"]) if row else 0.0) + syn_inc
            mean_r, syn, aff = _compute_pair_metrics(
                new_eff, new_rsum, new_ssum, pair_prior_m=pair_prior_m
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO tag_pair_stats(
                    tag_a, tag_b, co_count, eff_count, rating_sum, synergy_sum,
                    mean_rating, synergy, affinity, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tag_a,
                    tag_b,
                    new_co,
                    round(new_eff, 6),
                    round(new_rsum, 6),
                    round(new_ssum, 6),
                    mean_r,
                    syn,
                    aff,
                    now,
                ),
            )

    @staticmethod
    def _refresh_all_pair_metrics(conn: Any, pair_prior_m: float) -> None:
        rows = conn.execute(
            "SELECT tag_a, tag_b, eff_count, rating_sum, synergy_sum FROM tag_pair_stats"
        ).fetchall()
        for r in rows:
            mean_r, syn, aff = _compute_pair_metrics(
                float(r["eff_count"]),
                float(r["rating_sum"]),
                float(r["synergy_sum"]),
                pair_prior_m=pair_prior_m,
            )
            conn.execute(
                "UPDATE tag_pair_stats SET mean_rating = ?, synergy = ?, affinity = ? "
                "WHERE tag_a = ? AND tag_b = ?",
                (mean_r, syn, aff, r["tag_a"], r["tag_b"]),
            )

    def _promote_staging_candidates(
        self,
        conn: Any,
        rl: dict[str, Any],
        now: str,
    ) -> list[str]:
        if not bool(rl.get("auto_promote", True)):
            return []
        prior_m = float(rl.get("prior_m", 3.0))
        promote_min_obs = int(rl.get("promote_min_obs", 2))
        promote_min_gain = float(rl.get("promote_min_gain", 0.25))
        wip_limit = max(1, int(rl.get("explore_wip_limit", 25)))

        explore_cnt = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM tag_stats WHERE stage = 'explore'"
            ).fetchone()["c"]
        )
        capacity = wip_limit - explore_cnt
        if capacity <= 0:
            return []

        rows = conn.execute(
            "SELECT tag, observed_count, observed_eff_count, observed_gain_sum "
            "FROM tag_stats WHERE stage = 'staging' AND observed_count >= ?",
            (promote_min_obs,),
        ).fetchall()
        candidates: list[tuple[float, int, str]] = []
        for r in rows:
            obs_eff = float(r["observed_eff_count"])
            denom_cnt = obs_eff if obs_eff > 1e-6 else float(r["observed_count"])
            gain = float(r["observed_gain_sum"]) / (denom_cnt + prior_m)
            if gain >= promote_min_gain:
                candidates.append((gain, int(r["observed_count"]), str(r["tag"])))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        promoted: list[str] = []
        for _, _, t_name in candidates[:capacity]:
            conn.execute(
                "UPDATE tag_stats SET stage = 'explore', updated_at = ? WHERE tag = ?",
                (now, t_name),
            )
            promoted.append(t_name)
        return promoted

    def rebuild_observed_eff_counts(
        self, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Rebuilds decayed observed_eff_count and observed_gain_sum from history to fix legacy denominator skew."""
        cfg = config or {}
        rl = cfg.get("rl", {})
        weight_decay = float(rl.get("weight_decay", 0.985))
        prior_m = float(rl.get("prior_m", 3.0))
        promote_min_obs = int(rl.get("promote_min_obs", 2))
        promote_min_gain = float(rl.get("promote_min_gain", 0.25))
        now = utc_now_iso()

        with self.storage.transaction() as conn:
            all_tags = self.storage.get_all_tags(conn=conn)
            tag_lower_map = {str(t["tag"]).strip().lower(): str(t["tag"]) for t in all_tags}
            stage_map = {str(t["tag"]): str(t["stage"]) for t in all_tags}

            rated_gens = self.storage.get_rated_generations_chronological(conn=conn)
            batches_by_id: dict[int, list[dict[str, Any]]] = {}
            for g in rated_gens:
                batches_by_id.setdefault(int(g["batch_id"]), []).append(g)

            obs_cnt: dict[str, int] = {}
            obs_eff: dict[str, float] = {}
            obs_gsum: dict[str, float] = {}
            running_ema = 0.0

            for batch_idx, batch_gens in enumerate(batches_by_id.values()):
                if batch_idx > 0:
                    for t in list(obs_eff.keys()):
                        obs_eff[t] *= weight_decay
                        obs_gsum[t] *= weight_decay

                batch_avg = sum(float(g["user_rating"]) for g in batch_gens) / len(batch_gens)
                running_ema = 0.85 * running_ema + 0.15 * batch_avg

                for gen in batch_gens:
                    r_val = float(gen["user_rating"])
                    for et in gen.get("emergent_tags") or []:
                        conf = float(et.get("score", 0.0))
                        et_name = str(et.get("tag", "")).strip()
                        if conf < 0.35 or et_name.lower() in STOPLIST_FOR_MINING:
                            continue
                        canon = tag_lower_map.get(et_name.lower())
                        if canon is None:
                            continue
                        gain = (r_val - running_ema) * conf
                        obs_cnt[canon] = obs_cnt.get(canon, 0) + 1
                        obs_eff[canon] = obs_eff.get(canon, 0.0) + 1.0
                        obs_gsum[canon] = obs_gsum.get(canon, 0.0) + gain

            ext_rows = self.storage.get_external_imports_chronological(conn=conn)
            for ext in ext_rows:
                r_ext = float(ext["rating"])
                for item in ext["applied_tags"]:
                    raw_name = str(item.get("tag", "")).strip()
                    canon = tag_lower_map.get(raw_name.lower())
                    if canon is None:
                        continue
                    conf = max(0.10, min(1.0, float(item.get("score", 0.5))))
                    if stage_map.get(canon) == "staging":
                        obs_cnt[canon] = obs_cnt.get(canon, 0) + 1
                        obs_eff[canon] = obs_eff.get(canon, 0.0) + 1.0
                        obs_gsum[canon] = obs_gsum.get(canon, 0.0) + r_ext * conf

            updated_rows = 0
            qualifying_staging = 0
            for t_name, eff_val in obs_eff.items():
                cnt_val = obs_cnt.get(t_name, 0)
                gsum_val = obs_gsum.get(t_name, 0.0)
                conn.execute(
                    """
                    UPDATE tag_stats
                    SET observed_count = MAX(observed_count, ?),
                        observed_eff_count = ?,
                        observed_gain_sum = ?,
                        updated_at = ?
                    WHERE tag = ?
                    """,
                    (cnt_val, round(eff_val, 6), round(gsum_val, 6), now, t_name),
                )
                updated_rows += 1
                if stage_map.get(t_name) == "staging" and cnt_val >= promote_min_obs:
                    if (gsum_val / (eff_val + prior_m)) >= promote_min_gain:
                        qualifying_staging += 1

        return {
            "updated_tags": updated_rows,
            "qualifying_staging": qualifying_staging,
        }

    def rebuild_pair_stats(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = config or {}
        rl = cfg.get("rl", {})
        enable_wd14 = bool(cfg.get("enable_wd14", True))
        pair_decay = float(rl.get("pair_decay", 0.997))
        pair_prior_m = float(rl.get("pair_prior_m", 4.0))
        prior_m = float(rl.get("prior_m", 3.0))
        now = utc_now_iso()

        with self.storage.transaction() as conn:
            all_tags = self.storage.get_all_tags(conn=conn)
            tag_map = {str(t["tag"]): t for t in all_tags}
            tag_lower_map = {str(t["tag"]).strip().lower(): str(t["tag"]) for t in all_tags}

            conn.execute(
                "UPDATE tag_stats SET rating_count = 0, rating_eff_count = 0.0, rating_sum = 0.0, mean_rating = 0.0"
            )
            conn.execute("DELETE FROM tag_pair_stats")

            rated_gens = self.storage.get_rated_generations_chronological(conn=conn)
            batches_by_id: dict[int, list[dict[str, Any]]] = {}
            for g in rated_gens:
                batches_by_id.setdefault(int(g["batch_id"]), []).append(g)

            t_cnt: dict[str, int] = {t: 0 for t in tag_map}
            t_eff: dict[str, float] = {t: 0.0 for t in tag_map}
            t_rsum: dict[str, float] = {t: 0.0 for t in tag_map}

            p_cnt: dict[tuple[str, str], int] = {}
            p_eff: dict[tuple[str, str], float] = {}
            p_rsum: dict[tuple[str, str], float] = {}
            p_ssum: dict[tuple[str, str], float] = {}

            running_ema = 0.0
            for batch_idx, batch_gens in enumerate(batches_by_id.values()):
                if batch_idx > 0:
                    for t in t_eff:
                        t_eff[t] *= pair_decay
                        t_rsum[t] *= pair_decay
                    for pk in p_eff:
                        p_eff[pk] *= pair_decay
                        p_rsum[pk] *= pair_decay
                        p_ssum[pk] *= pair_decay

                ema_for_mu = running_ema
                gen_tag_lists: list[tuple[dict[str, Any], list[str]]] = []
                for gen in batch_gens:
                    unique_tags: list[str] = []
                    seen_gen: set[str] = set()
                    for st in gen["sampled_tags"]:
                        canon = tag_lower_map.get(str(st.get("tag", "")).strip().lower())
                        if canon is None or canon in seen_gen:
                            continue
                        seen_gen.add(canon)
                        unique_tags.append(canon)
                    gen_tag_lists.append((gen, unique_tags))

                shared_skeleton: set[str] = set()
                is_contrast_batch = False
                if len(gen_tag_lists) >= 2 and all(len(ts) >= 2 for _, ts in gen_tag_lists):
                    shared_skeleton = set.intersection(*(set(ts) for _, ts in gen_tag_lists))
                    if len(shared_skeleton) >= 2:
                        var_sizes = [len(set(ts) - shared_skeleton) for _, ts in gen_tag_lists]
                        if all(1 <= vs <= 3 for vs in var_sizes):
                            is_contrast_batch = True

                for gen, unique_tags in gen_tag_lists:
                    r_val = float(gen["user_rating"])
                    det_map = {
                        str(d.get("tag", "")).strip().lower(): float(d.get("score", 0.0))
                        for d in gen["detected_tags"]
                    }
                    variants_g = set(unique_tags) - shared_skeleton if is_contrast_batch else set()

                    mu: dict[str, float] = {}
                    w_t: dict[str, float] = {}
                    for t in unique_tags:
                        mu[t] = (t_rsum[t] + prior_m * ema_for_mu) / (t_eff[t] + prior_m)
                        if (
                            enable_wd14
                            and det_map
                            and self.tagger.is_known_tag(t)
                            and r_val > 0.0
                            and det_map.get(t.lower(), 0.0) < 0.12
                        ):
                            w_t[t] = 0.5
                        else:
                            w_t[t] = 1.0

                    mu_all = (
                        sum(mu[t] for t in unique_tags) / len(unique_tags)
                        if unique_tags
                        else ema_for_mu
                    )

                    for i in range(len(unique_tags)):
                        ti = unique_tags[i]
                        for j in range(i + 1, len(unique_tags)):
                            tj = unique_tags[j]
                            pk = _pair_key(ti, tj)
                            w_p = min(w_t[ti], w_t[tj])
                            if is_contrast_batch:
                                if (ti in variants_g) or (tj in variants_g):
                                    w_p *= 1.35
                                else:
                                    w_p *= 0.85
                            s_ab = r_val - mu_all
                            p_cnt[pk] = p_cnt.get(pk, 0) + 1
                            p_eff[pk] = p_eff.get(pk, 0.0) + w_p
                            p_rsum[pk] = p_rsum.get(pk, 0.0) + w_p * r_val
                            p_ssum[pk] = p_ssum.get(pk, 0.0) + w_p * s_ab

                    for t in unique_tags:
                        t_cnt[t] += 1
                        t_eff[t] += w_t[t]
                        t_rsum[t] += w_t[t] * r_val

                batch_avg = sum(float(g["user_rating"]) for g in batch_gens) / len(batch_gens)
                running_ema = 0.85 * running_ema + 0.15 * batch_avg

            ext_rows = self.storage.get_external_imports_chronological(conn=conn)
            for ext in ext_rows:
                r_ext = float(ext["rating"])
                ext_items: list[tuple[str, float]] = []
                seen_ext: set[str] = set()
                for item in ext["applied_tags"]:
                    canon = tag_lower_map.get(str(item.get("tag", "")).strip().lower())
                    if canon is None or canon in seen_ext:
                        continue
                    if tag_map[canon]["stage"] not in ("core", "explore"):
                        continue
                    seen_ext.add(canon)
                    conf = max(0.10, min(1.0, float(item.get("score", 0.5))))
                    ext_items.append((canon, conf))
                ext_items.sort(key=lambda x: x[1], reverse=True)
                ext_items = ext_items[:12]
                if len(ext_items) >= 2:
                    mu_ext = {
                        t: (t_rsum[t] + prior_m * running_ema) / (t_eff[t] + prior_m)
                        for t, _ in ext_items
                    }
                    mu_ext_all = sum(mu_ext.values()) / len(mu_ext)
                    for i in range(len(ext_items)):
                        ti, ci = ext_items[i]
                        for j in range(i + 1, len(ext_items)):
                            tj, cj = ext_items[j]
                            pk = _pair_key(ti, tj)
                            w_p = 0.5 * min(ci, cj)
                            s_ab = r_ext - mu_ext_all
                            p_cnt[pk] = p_cnt.get(pk, 0) + 1
                            p_eff[pk] = p_eff.get(pk, 0.0) + w_p
                            p_rsum[pk] = p_rsum.get(pk, 0.0) + w_p * r_ext
                            p_ssum[pk] = p_ssum.get(pk, 0.0) + w_p * s_ab

            for t in tag_map:
                if t_cnt[t] > 0:
                    mean_t = t_rsum[t] / t_eff[t] if t_eff[t] > 1e-6 else 0.0
                    conn.execute(
                        "UPDATE tag_stats SET rating_count = ?, rating_eff_count = ?, rating_sum = ?, mean_rating = ? WHERE tag = ?",
                        (t_cnt[t], round(t_eff[t], 6), round(t_rsum[t], 6), round(mean_t, 6), t),
                    )

            strong_positive = 0
            for (tag_a, tag_b), cnt in p_cnt.items():
                eff_c = p_eff[(tag_a, tag_b)]
                r_s = p_rsum[(tag_a, tag_b)]
                s_s = p_ssum[(tag_a, tag_b)]
                mean_r, syn, aff = _compute_pair_metrics(
                    eff_c, r_s, s_s, pair_prior_m=pair_prior_m
                )
                if cnt >= 2 and aff >= 0.25:
                    strong_positive += 1
                conn.execute(
                    """
                    INSERT INTO tag_pair_stats(
                        tag_a, tag_b, co_count, eff_count, rating_sum, synergy_sum,
                        mean_rating, synergy, affinity, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tag_a,
                        tag_b,
                        cnt,
                        round(eff_c, 6),
                        round(r_s, 6),
                        round(s_s, 6),
                        mean_r,
                        syn,
                        aff,
                        now,
                    ),
                )

        return {
            "generations_processed": len(rated_gens),
            "pairs_count": len(p_cnt),
            "strong_positive_pairs": strong_positive,
        }

    def apply_batch_ratings(
        self,
        batch_id: int,
        ratings_input: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        rl = config["rl"]
        enable_wd14 = bool(config.get("enable_wd14", True))
        weight_decay = float(rl.get("weight_decay", 0.985))
        pair_decay = float(rl.get("pair_decay", 0.997))
        pair_prior_m = float(rl.get("pair_prior_m", 4.0))
        prior_m = float(rl.get("prior_m", 3.0))

        rating_by_gen_id: dict[int, dict[str, Any]] = {
            int(item["generation_id"]): item for item in ratings_input
        }

        with self.storage.transaction() as conn:
            gen_rows = conn.execute(
                "SELECT * FROM generations WHERE batch_id = ? ORDER BY slot_index ASC",
                (batch_id,),
            ).fetchall()
            if not gen_rows:
                raise ValueError(f"批次 {batch_id} 不存在")

            valid_items: list[tuple[Any, int]] = []
            corrupted_ids: list[int] = []

            for row in gen_rows:
                gid = int(row["id"])
                if gid not in rating_by_gen_id:
                    continue
                if int(row["weight_applied"]) == 1:
                    continue
                inp = rating_by_gen_id[gid]
                if bool(inp.get("is_corrupted", False)):
                    corrupted_ids.append(gid)
                else:
                    raw_r = inp.get("rating")
                    if raw_r is None:
                        raise ValueError(f"图片 #{gid} 未提供评分且未标记为崩图")
                    r_val = int(raw_r)
                    if r_val not in (-2, -1, 0, 1, 2):
                        raise ValueError(f"评分必须在 -2..2 之间，收到: {r_val}")
                    valid_items.append((row, r_val))

            now = utc_now_iso()
            for gid in corrupted_ids:
                conn.execute(
                    "UPDATE generations SET user_rating = NULL, is_corrupted = 1, weight_applied = 1 WHERE id = ?",
                    (gid,),
                )

            promoted_tags: list[str] = []
            round_num = int(self.storage.get_meta("round_num", "0"))
            reward_ema = float(self.storage.get_meta("reward_ema", "0.0"))

            if valid_items:
                round_num += 1
                self.storage.set_meta("round_num", str(round_num), conn=conn)

                # Decay accumulated sums synchronously (including observed_eff_count)
                conn.execute(
                    """
                    UPDATE tag_stats
                    SET active_reward_sum = active_reward_sum * ?,
                        observed_gain_sum = observed_gain_sum * ?,
                        observed_eff_count = observed_eff_count * ?,
                        rating_eff_count = rating_eff_count * ?,
                        rating_sum = rating_sum * ?
                    """,
                    (weight_decay, weight_decay, weight_decay, pair_decay, pair_decay),
                )
                conn.execute(
                    "UPDATE cluster_stats SET reward_sum = reward_sum * ?",
                    (weight_decay,),
                )
                conn.execute(
                    "UPDATE tag_pair_stats SET eff_count = eff_count * ?, rating_sum = rating_sum * ?, synergy_sum = synergy_sum * ?",
                    (pair_decay, pair_decay, pair_decay),
                )

                ema_for_mu = reward_ema
                r_bar_eff = sum(r for _, r in valid_items) / len(valid_items)
                reward_ema = 0.85 * reward_ema + 0.15 * r_bar_eff
                self.storage.set_meta("reward_ema", f"{reward_ema:.6f}", conn=conn)

                parsed_valid: list[tuple[Any, int, list[dict[str, Any]], set[str]]] = []
                for row, r_val in valid_items:
                    s_tags = json.loads(row["sampled_tags_json"] or "[]")
                    s_set = {
                        str(st.get("tag", "")).strip()
                        for st in s_tags
                        if str(st.get("tag", "")).strip()
                    }
                    parsed_valid.append((row, r_val, s_tags, s_set))

                shared_skeleton: set[str] = set()
                is_contrast_batch = False
                if len(parsed_valid) >= 2 and all(len(s) >= 2 for _, _, _, s in parsed_valid):
                    shared_skeleton = set.intersection(*(s for _, _, _, s in parsed_valid))
                    if len(shared_skeleton) >= 2:
                        var_sizes = [len(s - shared_skeleton) for _, _, _, s in parsed_valid]
                        if all(1 <= vs <= 3 for vs in var_sizes):
                            is_contrast_batch = True

                self.storage.ensure_clusters(CLUSTER_DEFS, conn=conn)

                for row, r_val, sampled_tags, s_set in parsed_valid:
                    gid = int(row["id"])
                    conn.execute(
                        "UPDATE generations SET user_rating = ?, is_corrupted = 0, weight_applied = 1 WHERE id = ?",
                        (r_val, gid),
                    )

                    detected_tags = json.loads(row["detected_tags_json"] or "[]")
                    emergent_tags = json.loads(row["emergent_tags_json"] or "[]")
                    variants_g = (s_set - shared_skeleton) if is_contrast_batch else set()

                    detected_map = {
                        d["tag"].strip().lower(): float(d["score"])
                        for d in detected_tags
                    }

                    # Collect unique sampled tags present in DB before mutating rating_* columns
                    unique_tags: list[str] = []
                    t_rows: dict[str, dict[str, Any]] = {}
                    seen_gen_tags: set[str] = set()
                    for st in sampled_tags:
                        t_name = str(st.get("tag", "")).strip()
                        if not t_name or t_name in seen_gen_tags:
                            continue
                        t_row = self.storage.get_tag(t_name, conn=conn)
                        if t_row is not None:
                            seen_gen_tags.add(t_name)
                            unique_tags.append(t_name)
                            t_rows[t_name] = t_row

                    # Pairwise synergy update using full-prompt excess residual
                    if unique_tags:
                        mu: dict[str, float] = {}
                        w_t: dict[str, float] = {}
                        for t in unique_tags:
                            tr = t_rows[t]
                            mu[t] = (float(tr["rating_sum"]) + prior_m * ema_for_mu) / (
                                float(tr["rating_eff_count"]) + prior_m
                            )
                            if (
                                enable_wd14
                                and detected_tags
                                and self.tagger.is_known_tag(t)
                                and r_val > 0
                                and detected_map.get(t.lower(), 0.0) < 0.12
                            ):
                                w_t[t] = 0.5
                            else:
                                w_t[t] = 1.0

                        mu_all = sum(mu[t] for t in unique_tags) / len(unique_tags)

                        pair_obs: list[tuple[str, str, float, float, float]] = []
                        for i in range(len(unique_tags)):
                            ti = unique_tags[i]
                            for j in range(i + 1, len(unique_tags)):
                                tj = unique_tags[j]
                                tag_a, tag_b = _pair_key(ti, tj)
                                w_p = min(w_t[ti], w_t[tj])
                                if is_contrast_batch:
                                    if (ti in variants_g) or (tj in variants_g):
                                        w_p *= 1.35
                                    else:
                                        w_p *= 0.85
                                s_ab = float(r_val) - mu_all
                                pair_obs.append(
                                    (tag_a, tag_b, w_p, w_p * float(r_val), w_p * s_ab)
                                )
                        self._upsert_pair_observations(
                            conn, pair_obs, pair_prior_m=pair_prior_m, now=now
                        )

                        for t in unique_tags:
                            tr = t_rows[t]
                            new_reff = float(tr["rating_eff_count"]) + w_t[t]
                            new_rsum = float(tr["rating_sum"]) + w_t[t] * float(r_val)
                            new_mean = new_rsum / new_reff if new_reff > 1e-6 else 0.0
                            conn.execute(
                                """
                                UPDATE tag_stats
                                SET rating_count = rating_count + 1,
                                    rating_eff_count = ?,
                                    rating_sum = ?,
                                    mean_rating = ?
                                WHERE tag = ?
                                """,
                                (round(new_reff, 6), round(new_rsum, 6), round(new_mean, 6), t),
                            )

                    # Expected reward from current smoothed weights
                    current_weights: list[float] = []
                    for st in sampled_tags:
                        t_row = self.storage.get_tag(st["tag"], conn=conn)
                        if t_row is not None:
                            current_weights.append(float(t_row["weight"]))
                    r_hat = (
                        sum(current_weights) / len(current_weights)
                        if current_weights
                        else 0.0
                    )

                    delta = 0.65 * (r_val - r_bar_eff) + 0.35 * (r_val - r_hat)
                    if r_val == 0:
                        delta *= 0.35

                    n_tags = max(1, len(sampled_tags))
                    base_attr = 1.0 / math.sqrt(n_tags)

                    for st in sampled_tags:
                        tag_name = st["tag"]
                        t_row = self.storage.get_tag(tag_name, conn=conn)
                        if t_row is None:
                            continue

                        attr = base_attr
                        if is_contrast_batch:
                            if tag_name in variants_g:
                                attr *= 1.35
                            elif tag_name in shared_skeleton:
                                attr *= 0.75

                        exec_hits_inc = 0
                        exec_trials_inc = 0
                        if enable_wd14 and detected_tags and self.tagger.is_known_tag(tag_name):
                            conf = detected_map.get(tag_name.strip().lower(), 0.0)
                            exec_trials_inc = 1
                            if conf >= 0.20:
                                exec_hits_inc = 1
                            if r_val > 0 and conf < 0.12:
                                attr *= 0.30

                        delta_t = delta * attr
                        prev_last = int(t_row["last_used_round"])
                        prev_consec = int(t_row["consecutive_uses"])
                        new_consec = (prev_consec + 1) if prev_last == (round_num - 1) else 1

                        conn.execute(
                            """
                            UPDATE tag_stats
                            SET active_count = active_count + 1,
                                active_reward_sum = active_reward_sum + ?,
                                exec_hits = exec_hits + ?,
                                exec_trials = exec_trials + ?,
                                last_used_round = ?,
                                consecutive_uses = ?,
                                updated_at = ?
                            WHERE tag = ?
                            """,
                            (
                                delta_t,
                                exec_hits_inc,
                                exec_trials_inc,
                                round_num,
                                new_consec,
                                now,
                                tag_name,
                            ),
                        )

                        conn.execute(
                            """
                            UPDATE cluster_stats
                            SET count = count + 1,
                                reward_sum = reward_sum + ?
                            WHERE cluster = ?
                            """,
                            (0.25 * delta_t, t_row["cluster"]),
                        )

                    # WD14 difference mining into staging pool with automatic classification
                    if enable_wd14 and emergent_tags:
                        for et in emergent_tags:
                            conf = float(et["score"])
                            et_name = et["tag"].strip()
                            if conf < 0.35 or et_name.lower() in STOPLIST_FOR_MINING:
                                continue
                            gain = (r_val - reward_ema) * conf
                            existing = self.storage.get_tag(et_name, conn=conn)
                            if existing is None:
                                new_slot, new_cluster = classify_tag(et_name, self._seed_lookup)
                                conn.execute(
                                    """
                                    INSERT INTO tag_stats(
                                        tag, slot, cluster, stage,
                                        active_count, active_reward_sum,
                                        observed_count, observed_eff_count, observed_gain_sum,
                                        exec_hits, exec_trials,
                                        weight, z_score, last_used_round, consecutive_uses, updated_at
                                    ) VALUES (?, ?, ?, 'staging', 0, 0.0, 1, 1.0, ?, 0, 0, 0.0, 0.0, 0, 0, ?)
                                    """,
                                    (et_name, new_slot, new_cluster, gain, now),
                                )
                            elif existing["stage"] == "staging":
                                new_obs = int(existing["observed_count"]) + 1
                                new_obs_eff = float(existing.get("observed_eff_count", 0.0)) + 1.0
                                new_gain_sum = float(existing["observed_gain_sum"]) + gain
                                conn.execute(
                                    """
                                    UPDATE tag_stats
                                    SET observed_count = ?,
                                        observed_eff_count = ?,
                                        observed_gain_sum = ?,
                                        updated_at = ?
                                    WHERE tag = ?
                                    """,
                                    (new_obs, round(new_obs_eff, 6), new_gain_sum, now, et_name),
                                )

                self._refresh_all_pair_metrics(conn, pair_prior_m=pair_prior_m)
                self._recompute_weights_and_zscores(conn, rl)
                promoted_tags = self._promote_staging_candidates(conn, rl, now)
                if promoted_tags:
                    self._recompute_weights_and_zscores(conn, rl)

            conn.execute(
                "UPDATE batches SET status = 'rated', updated_at = ? WHERE id = ?",
                (now, batch_id),
            )

        return {
            "batch_id": batch_id,
            "round_num": round_num,
            "reward_ema": round(reward_ema, 4),
            "valid_rated_count": len(valid_items),
            "corrupted_count": len(corrupted_ids),
            "promoted_tags": promoted_tags,
        }

    def perform_tag_action(
        self,
        action: str,
        payload: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        rl = config["rl"]
        now = utc_now_iso()

        with self.storage.transaction() as conn:
            self.storage.ensure_clusters(CLUSTER_DEFS, conn=conn)

            if action == "reset_weights":
                conn.execute(
                    """
                    UPDATE tag_stats
                    SET active_count = 0, active_reward_sum = 0.0,
                        observed_count = 0, observed_eff_count = 0.0, observed_gain_sum = 0.0,
                        exec_hits = 0, exec_trials = 0,
                        rating_count = 0, rating_eff_count = 0.0,
                        rating_sum = 0.0, mean_rating = 0.0,
                        weight = 0.0, z_score = 0.0,
                        last_used_round = 0, consecutive_uses = 0,
                        updated_at = ?
                    """,
                    (now,),
                )
                conn.execute("UPDATE cluster_stats SET count = 0, reward_sum = 0.0, weight = 0.0")
                conn.execute("DELETE FROM tag_pair_stats")
                self.storage.set_meta("round_num", "0", conn=conn)
                self.storage.set_meta("reward_ema", "0.0", conn=conn)
                return {"ok": True, "action": action}

            if action == "reclassify":
                only_discovered = bool(payload.get("only_discovered", False))
                rows = self.storage.get_all_tags(conn=conn)
                moved_count = 0
                slot_changes: dict[str, int] = {}

                for r in rows:
                    cur_slot = str(r["slot"])
                    cur_cluster = str(r["cluster"])
                    if only_discovered and cur_slot != "discovered" and cur_cluster != "discovered_misc":
                        continue
                    new_slot, new_cluster = classify_tag(str(r["tag"]), self._seed_lookup)
                    if new_slot != cur_slot or new_cluster != cur_cluster:
                        conn.execute(
                            "UPDATE tag_stats SET slot = ?, cluster = ?, updated_at = ? WHERE tag = ?",
                            (new_slot, new_cluster, now, r["tag"]),
                        )
                        moved_count += 1
                        slot_changes[new_slot] = slot_changes.get(new_slot, 0) + 1

                # Rebuild cluster_stats aggregations cleanly from reclassified tag_stats
                conn.execute("UPDATE cluster_stats SET count = 0, reward_sum = 0.0, weight = 0.0")
                agg_rows = conn.execute(
                    "SELECT cluster, SUM(active_count) AS c_cnt, SUM(0.25 * active_reward_sum) AS c_rsum "
                    "FROM tag_stats GROUP BY cluster"
                ).fetchall()
                for ar in agg_rows:
                    conn.execute(
                        "UPDATE cluster_stats SET count = ?, reward_sum = ? WHERE cluster = ?",
                        (int(ar["c_cnt"] or 0), float(ar["c_rsum"] or 0.0), ar["cluster"]),
                    )

                self._recompute_weights_and_zscores(conn, rl)
                slot_summary = self.storage.get_stage_slot_summary(conn=conn)
                return {
                    "ok": True,
                    "action": action,
                    "total_tags": len(rows),
                    "moved_count": moved_count,
                    "slot_changes": slot_changes,
                    "slot_summary": slot_summary,
                }

            if action == "import_pack":
                raw_packs = payload.get("packs")
                if not isinstance(raw_packs, list) or not raw_packs:
                    raise ValueError("请选择至少一个要导入的扩展词包")
                target_stage = str(payload.get("target_stage", "explore")).strip().lower()
                if target_stage not in ("explore", "staging"):
                    target_stage = "explore"
                promote_existing = bool(payload.get("promote_existing", True))

                all_rows = self.storage.get_all_tags(conn=conn)
                by_lower = {str(r["tag"]).strip().lower(): r for r in all_rows}
                added: list[str] = []
                promoted: list[str] = []
                skipped: list[str] = []

                for pack_key in raw_packs:
                    pack = CURATED_EXPANSION_PACKS.get(str(pack_key).strip())
                    if pack is None:
                        continue
                    for item in pack.get("tags", []):
                        tag_str = str(item["tag"]).strip()
                        lower = tag_str.lower()
                        t_slot = str(item["slot"])
                        t_cluster = str(item["cluster"])
                        existing = by_lower.get(lower)
                        if existing is None:
                            conn.execute(
                                """
                                INSERT INTO tag_stats(
                                    tag, slot, cluster, stage,
                                    active_count, active_reward_sum,
                                    observed_count, observed_eff_count, observed_gain_sum,
                                    exec_hits, exec_trials,
                                    weight, z_score, last_used_round, consecutive_uses, updated_at
                                ) VALUES (?, ?, ?, ?, 0, 0.0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0, ?)
                                """,
                                (tag_str, t_slot, t_cluster, target_stage, now),
                            )
                            added.append(tag_str)
                            by_lower[lower] = {"tag": tag_str, "stage": target_stage}
                        elif (
                            promote_existing
                            and target_stage == "explore"
                            and existing["stage"] == "staging"
                        ):
                            conn.execute(
                                "UPDATE tag_stats SET stage = 'explore', slot = ?, cluster = ?, updated_at = ? WHERE tag = ?",
                                (t_slot, t_cluster, now, existing["tag"]),
                            )
                            promoted.append(existing["tag"])
                            by_lower[lower] = {**existing, "stage": "explore"}
                        else:
                            skipped.append(tag_str)

                if added or promoted:
                    self._recompute_weights_and_zscores(conn, rl)
                return {
                    "ok": True,
                    "action": action,
                    "target_stage": target_stage,
                    "added": added,
                    "promoted": promoted,
                    "skipped_count": len(skipped),
                }

            tag_name = str(payload.get("tag", "")).strip()
            if not tag_name:
                raise ValueError("缺少 tag 名称")

            if action == "add":
                raw_slot = str(payload.get("slot", "auto")).strip().lower()
                if not raw_slot or raw_slot == "auto":
                    slot, auto_cluster = classify_tag(tag_name, self._seed_lookup)
                elif raw_slot in SLOT_ORDER:
                    slot = raw_slot
                    auto_cluster = DEFAULT_CLUSTER_FOR_SLOT[slot]
                else:
                    slot, auto_cluster = classify_tag(tag_name, self._seed_lookup)
                stage = str(payload.get("stage", "explore"))
                if stage not in ("core", "explore", "staging", "negative", "ignored"):
                    stage = "explore"
                cluster = str(payload.get("cluster") or auto_cluster)
                conn.execute(
                    "INSERT OR IGNORE INTO cluster_stats(cluster, slot, label, count, reward_sum, weight) VALUES (?, ?, ?, 0, 0.0, 0.0)",
                    (cluster, slot, cluster),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tag_stats(
                        tag, slot, cluster, stage,
                        active_count, active_reward_sum,
                        observed_count, observed_eff_count, observed_gain_sum,
                        exec_hits, exec_trials,
                        weight, z_score, last_used_round, consecutive_uses, updated_at
                    ) VALUES (?, ?, ?, ?, 0, 0.0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0, ?)
                    """,
                    (tag_name, slot, cluster, stage, now),
                )
                self._recompute_weights_and_zscores(conn, rl)
                return {"ok": True, "action": action, "tag": tag_name, "slot": slot, "cluster": cluster}

            if action == "set_slot":
                target_slot = str(payload.get("slot", "")).strip()
                if target_slot not in SLOT_ORDER:
                    raise ValueError(f"无效槽位: {target_slot}")
                target_cluster = str(payload.get("cluster") or DEFAULT_CLUSTER_FOR_SLOT[target_slot])
                conn.execute(
                    "INSERT OR IGNORE INTO cluster_stats(cluster, slot, label, count, reward_sum, weight) VALUES (?, ?, ?, 0, 0.0, 0.0)",
                    (target_cluster, target_slot, target_cluster),
                )
                conn.execute(
                    "UPDATE tag_stats SET slot = ?, cluster = ?, updated_at = ? WHERE tag = ?",
                    (target_slot, target_cluster, now, tag_name),
                )
                self._recompute_weights_and_zscores(conn, rl)
                return {"ok": True, "action": action, "tag": tag_name, "slot": target_slot, "cluster": target_cluster}

            if action in ("promote", "set_stage"):
                target_stage = (
                    "explore"
                    if action == "promote"
                    else str(payload.get("stage", "explore"))
                )
                if target_stage not in ("core", "explore", "staging", "negative", "ignored"):
                    raise ValueError(f"无效阶段: {target_stage}")
                target_slot = payload.get("slot")
                if target_slot and target_slot in SLOT_ORDER:
                    target_cluster = DEFAULT_CLUSTER_FOR_SLOT[target_slot]
                    conn.execute(
                        "UPDATE tag_stats SET stage = ?, slot = ?, cluster = ?, updated_at = ? WHERE tag = ?",
                        (target_stage, target_slot, target_cluster, now, tag_name),
                    )
                else:
                    conn.execute(
                        "UPDATE tag_stats SET stage = ?, updated_at = ? WHERE tag = ?",
                        (target_stage, now, tag_name),
                    )
                self._recompute_weights_and_zscores(conn, rl)
                return {"ok": True, "action": action, "tag": tag_name, "stage": target_stage}

            raise ValueError(f"未知操作: {action}")

    def preview_external_tags(
        self,
        detected_tags: list[dict[str, Any]],
        threshold: float = 0.35,
    ) -> list[dict[str, Any]]:
        all_tags = self.storage.get_all_tags()
        by_lower = {str(t["tag"]).strip().lower(): t for t in all_tags}

        preview: list[dict[str, Any]] = []
        for dt in detected_tags:
            cat = str(dt.get("category", "general"))
            if cat != "general":
                continue
            tag_name = str(dt.get("tag", "")).strip()
            if not tag_name:
                continue
            score = float(dt.get("score", 0.0))
            lower = tag_name.lower()
            existing = by_lower.get(lower)
            is_stop = lower in STOPLIST_FOR_MINING
            blocked_stage = existing is not None and existing["stage"] in ("ignored", "negative")
            pred_slot, pred_cluster = classify_tag(tag_name, self._seed_lookup)
            preview.append(
                {
                    "tag": existing["tag"] if existing else tag_name,
                    "raw_tag": dt.get("raw_tag", tag_name),
                    "score": round(score, 4),
                    "category": cat,
                    "current_stage": existing["stage"] if existing else "new",
                    "current_slot": existing["slot"] if existing else pred_slot,
                    "predicted_slot": pred_slot,
                    "predicted_cluster": pred_cluster,
                    "current_weight": round(float(existing["weight"]), 4) if existing else 0.0,
                    "is_stoplist": is_stop,
                    "default_selected": (
                        score >= threshold and not is_stop and not blocked_stage
                    ),
                }
            )
        return preview

    def apply_external_image(
        self,
        selected_tags: list[dict[str, Any]],
        rating: int,
        direct_explore: bool,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        r_ext = int(rating)
        if r_ext not in (-2, -1, 1, 2):
            raise ValueError(f"外部图评分必须为 -2, -1, +1 或 +2，收到: {rating}")
        if not selected_tags:
            raise ValueError("请至少保留一个标签再导入")

        rl = config["rl"]
        prior_m = float(rl.get("prior_m", 3.0))
        now = utc_now_iso()

        with self.storage.transaction() as conn:
            self.storage.ensure_clusters(CLUSTER_DEFS, conn=conn)
            all_rows = self.storage.get_all_tags(conn=conn)
            by_lower = {str(r["tag"]).strip().lower(): r for r in all_rows}

            seen_lower: set[str] = set()
            active_items: list[tuple[dict[str, Any], float]] = []
            novel_items: list[tuple[dict[str, Any] | None, str, float]] = []

            for item in selected_tags:
                raw_name = str(item.get("tag", "")).strip()
                if not raw_name:
                    continue
                lower = raw_name.lower()
                if lower in seen_lower or lower in STOPLIST_FOR_MINING:
                    continue
                seen_lower.add(lower)

                conf = max(0.10, min(1.0, float(item.get("score", 0.5))))
                existing = by_lower.get(lower)
                if existing is not None:
                    if existing["stage"] in ("ignored", "negative"):
                        continue
                    if existing["stage"] in ("core", "explore"):
                        active_items.append((existing, conf))
                    else:
                        novel_items.append((existing, existing["tag"], conf))
                else:
                    novel_items.append((None, raw_name, conf))

            if not active_items and not novel_items:
                raise ValueError("没有可应用权重的有效标签（已跳过基础词、负面词和忽略词）")

            before_state: dict[str, tuple[float, float]] = {}
            for existing, _ in active_items:
                before_state[existing["tag"]] = (
                    float(existing["weight"]),
                    float(existing["z_score"]),
                )

            if active_items:
                denom = math.sqrt(max(len(active_items), 4))
                for existing, conf in active_items:
                    delta_t = (r_ext * conf) / denom
                    conn.execute(
                        """
                        UPDATE tag_stats
                        SET active_count = active_count + 1,
                            active_reward_sum = active_reward_sum + ?,
                            updated_at = ?
                        WHERE tag = ?
                        """,
                        (delta_t, now, existing["tag"]),
                    )
                    conn.execute(
                        """
                        UPDATE cluster_stats
                        SET count = count + 1,
                            reward_sum = reward_sum + ?
                        WHERE cluster = ?
                        """,
                        (0.25 * delta_t, existing["cluster"]),
                    )

            promoted_tags: list[str] = []
            direct_explored_tags: list[str] = []
            staged_tags: list[str] = []

            if novel_items:
                explore_denom = math.sqrt(max(len(active_items) + len(novel_items), 6))
                for existing, tag_name, conf in novel_items:
                    obs_gain_inc = r_ext * conf
                    if direct_explore and r_ext > 0:
                        init_delta = (r_ext * conf) / explore_denom
                        before_state[tag_name] = (
                            float(existing["weight"]) if existing else 0.0,
                            float(existing["z_score"]) if existing else 0.0,
                        )
                        if existing is None:
                            new_slot, new_cluster = classify_tag(tag_name, self._seed_lookup)
                            conn.execute(
                                """
                                INSERT INTO tag_stats(
                                    tag, slot, cluster, stage,
                                    active_count, active_reward_sum,
                                    observed_count, observed_eff_count, observed_gain_sum,
                                    exec_hits, exec_trials,
                                    weight, z_score, last_used_round, consecutive_uses, updated_at
                                ) VALUES (?, ?, ?, 'explore', 1, ?, 1, 1.0, ?, 0, 0, 0.0, 0.0, 0, 0, ?)
                                """,
                                (tag_name, new_slot, new_cluster, init_delta, obs_gain_inc, now),
                            )
                            target_cluster = new_cluster
                        else:
                            conn.execute(
                                """
                                UPDATE tag_stats
                                SET stage = 'explore',
                                    active_count = active_count + 1,
                                    active_reward_sum = active_reward_sum + ?,
                                    observed_count = observed_count + 1,
                                    observed_eff_count = observed_eff_count + 1.0,
                                    observed_gain_sum = observed_gain_sum + ?,
                                    updated_at = ?
                                WHERE tag = ?
                                """,
                                (init_delta, obs_gain_inc, now, tag_name),
                            )
                            target_cluster = str(existing["cluster"])
                        conn.execute(
                            """
                            UPDATE cluster_stats
                            SET count = count + 1,
                                reward_sum = reward_sum + ?
                            WHERE cluster = ?
                            """,
                            (0.25 * init_delta, target_cluster),
                        )
                        direct_explored_tags.append(tag_name)
                    else:
                        if existing is None:
                            new_slot, new_cluster = classify_tag(tag_name, self._seed_lookup)
                            staged_tags.append(tag_name)
                            conn.execute(
                                """
                                INSERT INTO tag_stats(
                                    tag, slot, cluster, stage,
                                    active_count, active_reward_sum,
                                    observed_count, observed_eff_count, observed_gain_sum,
                                    exec_hits, exec_trials,
                                    weight, z_score, last_used_round, consecutive_uses, updated_at
                                ) VALUES (?, ?, ?, 'staging', 0, 0.0, 1, 1.0, ?, 0, 0, 0.0, 0.0, 0, 0, ?)
                                """,
                                (tag_name, new_slot, new_cluster, obs_gain_inc, now),
                            )
                        else:
                            new_obs = int(existing["observed_count"]) + 1
                            new_obs_eff = float(existing.get("observed_eff_count", 0.0)) + 1.0
                            new_gain_sum = float(existing["observed_gain_sum"]) + obs_gain_inc
                            staged_tags.append(tag_name)
                            conn.execute(
                                """
                                UPDATE tag_stats
                                SET observed_count = ?,
                                    observed_eff_count = ?,
                                    observed_gain_sum = ?,
                                    updated_at = ?
                                WHERE tag = ?
                                """,
                                (new_obs, round(new_obs_eff, 6), new_gain_sum, now, tag_name),
                            )

            if active_items and len(active_items) >= 2:
                pair_prior_m = float(rl.get("pair_prior_m", 4.0))
                ext_ema = float(self.storage.get_meta("reward_ema", "0.0"))
                top_ext = sorted(active_items, key=lambda x: x[1], reverse=True)[:12]
                mu_ext = {
                    row["tag"]: (
                        float(row.get("rating_sum", 0.0)) + prior_m * ext_ema
                    )
                    / (float(row.get("rating_eff_count", 0.0)) + prior_m)
                    for row, _ in top_ext
                }
                mu_ext_all = sum(mu_ext.values()) / len(mu_ext)
                ext_pair_obs: list[tuple[str, str, float, float, float]] = []
                for i in range(len(top_ext)):
                    ti, ci = top_ext[i][0]["tag"], top_ext[i][1]
                    for j in range(i + 1, len(top_ext)):
                        tj, cj = top_ext[j][0]["tag"], top_ext[j][1]
                        if ti == tj:
                            continue
                        tag_a, tag_b = _pair_key(ti, tj)
                        w_p = 0.5 * min(ci, cj)
                        s_ab = float(r_ext) - mu_ext_all
                        ext_pair_obs.append(
                            (tag_a, tag_b, w_p, w_p * float(r_ext), w_p * s_ab)
                        )
                self._upsert_pair_observations(
                    conn, ext_pair_obs, pair_prior_m=pair_prior_m, now=now
                )

            self._recompute_weights_and_zscores(conn, rl)
            promoted_tags = self._promote_staging_candidates(conn, rl, now)
            if promoted_tags:
                staged_tags = [t for t in staged_tags if t not in set(promoted_tags)]
                self._recompute_weights_and_zscores(conn, rl)

            updated_active: list[dict[str, Any]] = []
            for tag_name, (old_w, old_z) in before_state.items():
                after_row = self.storage.get_tag(tag_name, conn=conn)
                if after_row is None:
                    continue
                updated_active.append(
                    {
                        "tag": tag_name,
                        "slot": after_row["slot"],
                        "stage": after_row["stage"],
                        "old_weight": round(old_w, 4),
                        "new_weight": round(float(after_row["weight"]), 4),
                        "old_z": round(old_z, 3),
                        "new_z": round(float(after_row["z_score"]), 3),
                    }
                )

        return {
            "rating": r_ext,
            "direct_explore": bool(direct_explore),
            "updated_active": updated_active,
            "updated_active_count": len(updated_active),
            "staged_tags": staged_tags,
            "staged_count": len(staged_tags),
            "promoted_tags": promoted_tags,
            "direct_explored_tags": direct_explored_tags,
        }

    def roll_manual_prompt(
        self,
        config: dict[str, Any],
        mode: str = "weighted",
        slots: list[str] | None = None,
        include_prefix: bool = True,
    ) -> dict[str, Any]:
        """Read-only prompt roller for manual generation using Scheme A soft-cap budget."""
        rl = config["rl"]
        prefix_tokens = self._split_prompt_tokens(str(config.get("fixed_prefix", "")))
        prefix_lower = {t.lower() for t in prefix_tokens}

        if mode not in ("weighted", "top", "random"):
            mode = "weighted"

        if slots is None:
            target_slots = list(SLOT_ORDER)
        else:
            req_set = {str(s).strip() for s in slots}
            target_slots = [s for s in SLOT_ORDER if s in req_set]

        target_slot_set = set(target_slots)
        rng = random.Random()
        candidates = [
            t
            for t in self.storage.get_all_tags()
            if t["stage"] in ("core", "explore")
            and t["slot"] in target_slot_set
            and t["tag"].strip().lower() not in prefix_lower
        ]

        current_round = int(self.storage.get_meta("round_num", "0"))
        progress = min(1.0, current_round / max(1, int(rl.get("temp_decay_rounds", 40))))
        tau = float(rl.get("temp_start", 1.2)) - progress * (
            float(rl.get("temp_start", 1.2)) - float(rl.get("temp_min", 0.55))
        )
        target_min = max(2, min(12, int(rl.get("target_tag_min", 5))))
        target_max = max(target_min, min(14, int(rl.get("target_tag_max", 8))))
        max_cap_sum = sum(SLOT_SOFT_CAPS.get(s, 2) for s in target_slots) if target_slots else 0

        if not candidates or max_cap_sum == 0:
            picked: list[dict[str, Any]] = []
        else:
            if len(target_slots) <= 3:
                desired_k = min(len(candidates), max_cap_sum, max(len(target_slots), 2))
            else:
                desired_k = min(len(candidates), max_cap_sum, rng.randint(target_min, target_max))

            if mode == "weighted":
                synergy_weight = float(rl.get("synergy_weight", 0.55))
                pair_aff_map, pair_co_map, cluster_pair_map = self._load_pair_context(
                    min_co_count=2, min_cluster_co=5
                )
                picked = self._sample_chained_tags(
                    candidates=candidates,
                    target_k=desired_k,
                    current_round=current_round,
                    rl=rl,
                    tau=tau,
                    batch_seen_tags=set(),
                    rng=rng,
                    slot_caps=SLOT_SOFT_CAPS,
                    allowed_slots=target_slot_set,
                    explore_quota=0,
                    pair_aff_map=pair_aff_map,
                    pair_co_map=pair_co_map,
                    cluster_pair_map=cluster_pair_map,
                    synergy_weight=synergy_weight,
                    pair_explore_beta=0.0,
                    batch_div_penalty=0.0,
                    prefer_uncovered_slots=(len(target_slots) <= 3),
                )
            elif mode == "top":
                by_slot: dict[str, list[dict[str, Any]]] = {}
                for c in candidates:
                    by_slot.setdefault(c["slot"], []).append(c)
                top_candidates: list[dict[str, Any]] = []
                for sl_items in by_slot.values():
                    sl_items.sort(key=lambda x: (float(x["z_score"]), float(x["weight"])), reverse=True)
                    top_candidates.extend(sl_items[:5])
                rng.shuffle(top_candidates)
                picked = []
                used_slots: dict[str, int] = {}
                for item in sorted(top_candidates, key=lambda x: (float(x["z_score"]), float(x["weight"])), reverse=True):
                    if len(picked) >= desired_k:
                        break
                    sl = item["slot"]
                    if used_slots.get(sl, 0) < SLOT_SOFT_CAPS.get(sl, 2):
                        picked.append(item)
                        used_slots[sl] = used_slots.get(sl, 0) + 1
            else:
                shuffled = list(candidates)
                rng.shuffle(shuffled)
                picked = []
                used_slots = {}
                for item in shuffled:
                    if len(picked) >= desired_k:
                        break
                    sl = item["slot"]
                    if used_slots.get(sl, 0) < SLOT_SOFT_CAPS.get(sl, 2):
                        picked.append(item)
                        used_slots[sl] = used_slots.get(sl, 0) + 1

        picked.sort(key=self._prompt_sort_key)
        prompt_chunks: list[str] = []
        if include_prefix and prefix_tokens:
            prompt_chunks.extend(prefix_tokens)

        rolled_tags: list[dict[str, Any]] = []
        for item in picked:
            prompt_chunks.append(item["tag"])
            rolled_tags.append(
                {
                    "tag": item["tag"],
                    "slot": item["slot"],
                    "stage": item["stage"],
                    "weight": round(float(item["weight"]), 4),
                    "z_score": round(float(item["z_score"]), 4),
                }
            )

        return {
            "mode": mode,
            "positive_prompt": ", ".join(prompt_chunks),
            "negative_prompt": self._build_auto_negative_prompt(config),
            "rolled_tags": rolled_tags,
        }

    def get_synergy_graph(
        self,
        config: dict[str, Any],
        min_co_count: int = 2,
        include_staging: bool = False,
        max_edges: int = 260,
        metric: str = "affinity",
    ) -> dict[str, Any]:
        clean_metric = "synergy" if str(metric).strip().lower() == "synergy" else "affinity"
        allowed_stages = ("core", "explore", "staging") if include_staging else ("core", "explore")
        all_tags = [t for t in self.storage.get_all_tags() if t["stage"] in allowed_stages]
        node_map = {str(t["tag"]): t for t in all_tags}

        min_co = max(1, int(min_co_count))
        raw_pairs = self.storage.get_all_pairs(min_co_count=min_co)
        edge_objs: list[dict[str, Any]] = []
        norm_denom = 0.55 if clean_metric == "synergy" else 0.85

        for p in raw_pairs:
            a, b = str(p["tag_a"]), str(p["tag_b"])
            if a not in node_map or b not in node_map:
                continue
            co = int(p["co_count"])
            mean_r = float(p["mean_rating"])
            syn = float(p["synergy"])
            aff = float(p["affinity"])
            conf = co / (co + 3.0)
            score_val = syn if clean_metric == "synergy" else aff

            if score_val >= 0.0:
                if clean_metric == "synergy":
                    strength = round(min(1.0, conf * min(1.0, syn / norm_denom) * 1.18), 4)
                else:
                    aff_norm = max(0.0, min(1.0, aff / norm_denom))
                    rat_norm = max(0.0, min(1.0, mean_r / 1.8))
                    strength = round(min(1.0, conf * (0.62 * aff_norm + 0.38 * rat_norm) * 1.18), 4)
                kind = "synergy"
            else:
                strength = round(min(1.0, conf * min(1.0, abs(score_val) / norm_denom) * 1.18), 4)
                kind = "clash"

            importance = abs(score_val) * math.log1p(co) + (
                0.0 if clean_metric == "synergy" else 0.15 * max(0.0, mean_r)
            )
            edge_objs.append(
                {
                    "source": a,
                    "target": b,
                    "co_count": co,
                    "confidence": round(conf, 3),
                    "mean_rating": round(mean_r, 3),
                    "synergy": round(syn, 3),
                    "affinity": round(aff, 3),
                    "strength": strength,
                    "kind": kind,
                    "_score": score_val,
                    "_importance": importance,
                }
            )

        thresh = 0.04 if clean_metric == "synergy" else 0.05
        top_synergies = [
            {k: v for k, v in e.items() if k not in ("_importance", "_score")}
            for e in sorted(
                (e for e in edge_objs if e["kind"] == "synergy" and e["_score"] > thresh),
                key=lambda x: (x["strength"], x["_score"], x["co_count"]),
                reverse=True,
            )[:15]
        ]
        top_clashes = [
            {k: v for k, v in e.items() if k not in ("_importance", "_score")}
            for e in sorted(
                (e for e in edge_objs if e["kind"] == "clash" and e["_score"] < -thresh),
                key=lambda x: (x["strength"], abs(x["_score"]), x["co_count"]),
                reverse=True,
            )[:10]
        ]

        limit = max(20, min(600, int(max_edges)))
        pos_limit = limit
        clash_limit = min(90, max(20, limit // 2))

        pos_sorted = sorted(
            (e for e in edge_objs if e["kind"] == "synergy"),
            key=lambda x: x["_importance"],
            reverse=True,
        )[:pos_limit]
        clash_sorted = sorted(
            (e for e in edge_objs if e["kind"] == "clash"),
            key=lambda x: x["_importance"],
            reverse=True,
        )[:clash_limit]

        # Mark compact positive backbone (~30-52 edges) for legible panoramic constellations
        pos_by_str = sorted(
            pos_sorted,
            key=lambda x: (x["strength"], x["_score"], x["co_count"]),
            reverse=True,
        )
        backbone_keys: set[tuple[str, str]] = {
            (e["source"], e["target"]) for e in pos_by_str[:28]
        }
        node_pos_adj: dict[str, list[dict[str, Any]]] = {}
        for e in pos_by_str:
            if e["strength"] >= 0.16:
                node_pos_adj.setdefault(e["source"], []).append(e)
                node_pos_adj.setdefault(e["target"], []).append(e)

        for rank_idx in (0, 1):
            if len(backbone_keys) >= 48:
                break
            tier_cands: list[dict[str, Any]] = []
            seen_tier: set[tuple[str, str]] = set()
            for adj_list in node_pos_adj.values():
                if rank_idx < len(adj_list):
                    e = adj_list[rank_idx]
                    key = (e["source"], e["target"])
                    if key not in backbone_keys and key not in seen_tier:
                        seen_tier.add(key)
                        tier_cands.append(e)
            tier_cands.sort(
                key=lambda x: (x["strength"], x["_score"], x["co_count"]),
                reverse=True,
            )
            for e in tier_cands:
                if len(backbone_keys) >= 48 or len(backbone_keys) >= 52:
                    break
                backbone_keys.add((e["source"], e["target"]))

        kept_sorted = pos_sorted + clash_sorted
        kept_edges = [
            {
                **{k: v for k, v in e.items() if k not in ("_importance", "_score")},
                "is_backbone": bool(
                    e["kind"] == "synergy" and (e["source"], e["target"]) in backbone_keys
                ),
            }
            for e in kept_sorted
        ]

        # Discover golden 3-tag cliques (triads)
        pos_map: dict[tuple[str, str], dict[str, Any]] = {}
        nbrs: dict[str, set[str]] = {}
        for e in pos_sorted:
            if e["strength"] >= 0.20 and e["_score"] > thresh:
                u, v = e["source"], e["target"]
                key = (u, v) if u < v else (v, u)
                pos_map[key] = e
                nbrs.setdefault(u, set()).add(v)
                nbrs.setdefault(v, set()).add(u)

        triad_candidates: list[tuple[float, dict[str, Any]]] = []
        for a in sorted(nbrs):
            nbr_a = nbrs[a]
            for b in sorted(x for x in nbr_a if x > a):
                common = nbr_a & nbrs[b]
                for c in sorted(x for x in common if x > b):
                    e1 = pos_map.get((a, b))
                    e2 = pos_map.get((a, c))
                    e3 = pos_map.get((b, c))
                    if not (e1 and e2 and e3):
                        continue
                    edges_3 = (e1, e2, e3)
                    avg_strength = sum(float(x["strength"]) for x in edges_3) / 3.0
                    min_strength = min(float(x["strength"]) for x in edges_3)
                    avg_affinity = sum(float(x["affinity"]) for x in edges_3) / 3.0
                    avg_synergy = sum(float(x["synergy"]) for x in edges_3) / 3.0
                    avg_rating = sum(float(x["mean_rating"]) for x in edges_3) / 3.0
                    min_co = min(int(x["co_count"]) for x in edges_3)
                    distinct_slots = len({str(node_map[x]["slot"]) for x in (a, b, c)})
                    tri_score = (
                        0.55 * avg_strength
                        + 0.35 * min_strength
                        + 0.05 * distinct_slots
                    )
                    triad_candidates.append(
                        (
                            tri_score,
                            {
                                "tags": [a, b, c],
                                "slots": [
                                    str(node_map[a]["slot"]),
                                    str(node_map[b]["slot"]),
                                    str(node_map[c]["slot"]),
                                ],
                                "avg_strength": round(avg_strength, 4),
                                "min_strength": round(min_strength, 4),
                                "avg_affinity": round(avg_affinity, 3),
                                "avg_synergy": round(avg_synergy, 3),
                                "avg_rating": round(avg_rating, 3),
                                "min_co": min_co,
                            },
                        )
                    )

        triad_candidates.sort(key=lambda item: item[0], reverse=True)
        top_triads: list[dict[str, Any]] = []
        tag_use: dict[str, int] = {}
        seen_tri: set[tuple[str, str, str]] = set()
        for _, tri in triad_candidates:
            tri_key = tuple(tri["tags"])
            if all(tag_use.get(t, 0) < 2 for t in tri["tags"]):
                top_triads.append(tri)
                seen_tri.add(tri_key)  # type: ignore[arg-type]
                for t in tri["tags"]:
                    tag_use[t] = tag_use.get(t, 0) + 1
                if len(top_triads) >= 6:
                    break
        if len(top_triads) < 3:
            for _, tri in triad_candidates:
                tri_key = tuple(tri["tags"])
                if tri_key not in seen_tri:
                    top_triads.append(tri)
                    seen_tri.add(tri_key)  # type: ignore[arg-type]
                    if len(top_triads) >= 6:
                        break

        degree_map: dict[str, int] = {t: 0 for t in node_map}
        best_aff_map: dict[str, float] = {t: 0.0 for t in node_map}
        for e in kept_edges:
            u, v = e["source"], e["target"]
            degree_map[u] = degree_map.get(u, 0) + 1
            degree_map[v] = degree_map.get(v, 0) + 1
            if e["kind"] == "synergy":
                val = float(e[clean_metric])
                best_aff_map[u] = max(best_aff_map.get(u, 0.0), val)
                best_aff_map[v] = max(best_aff_map.get(v, 0.0), val)

        nodes: list[dict[str, Any]] = []
        for t in all_tags:
            name = str(t["tag"])
            nodes.append(
                {
                    "tag": name,
                    "slot": str(t["slot"]),
                    "cluster": str(t["cluster"]),
                    "stage": str(t["stage"]),
                    "weight": round(float(t["weight"]), 4),
                    "z_score": round(float(t["z_score"]), 4),
                    "active_count": int(t["active_count"]),
                    "rating_count": int(t.get("rating_count", 0)),
                    "mean_rating": round(float(t.get("mean_rating", 0.0)), 3),
                    "degree": degree_map.get(name, 0),
                    "best_affinity": round(best_aff_map.get(name, 0.0), 3),
                }
            )

        return {
            "nodes": nodes,
            "edges": kept_edges,
            "top_synergies": top_synergies,
            "top_clashes": top_clashes,
            "top_triads": top_triads,
            "stats": {
                "rated_generations": len(self.storage.get_rated_generations_chronological()),
                "total_pairs_db": self.storage.get_pair_count(),
                "filtered_pairs": len(edge_objs),
                "shown_edges": len(kept_edges),
                "positive_edges": len(pos_sorted),
                "clash_edges": len(clash_sorted),
                "backbone_edges": sum(1 for e in kept_edges if e.get("is_backbone")),
                "active_nodes": len(nodes),
                "synergy_weight": float(config.get("rl", {}).get("synergy_weight", 0.55)),
                "metric": clean_metric,
            },
        }
