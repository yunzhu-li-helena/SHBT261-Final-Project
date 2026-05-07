"""End-to-end TextVQA pipeline for SHBT261 final project.

This module is designed for Google Colab Pro with a CUDA GPU. It supports
zero-shot evaluation, prompt engineering, OCR-aware prompting, metrics, plots,
and error analysis.
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "lmms-lab/textvqa"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class RunConfig:
    project_dir: str = "/content/drive/MyDrive/textvqa_final"
    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    seed: int = 42
    use_4bit: bool = True
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
    inference_batch_size: int = 4
    judge_model_name: str = JUDGE_MODEL_NAME
    judge_batch_size: int = 16


def ensure_dirs(project_dir: str) -> Dict[str, Path]:
    root = Path(project_dir)
    dirs = {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_textvqa_split(
    split: str,
    dataset_name: str = DATASET_NAME,
    max_samples: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = False,
):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        if shuffle:
            ds = ds.shuffle(seed=seed)
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def get_image(sample: Dict[str, Any]):
    from io import BytesIO

    from PIL import Image

    image = sample.get("image")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    raise ValueError("Could not read image from sample.")


def get_answers(sample: Dict[str, Any]) -> List[str]:
    answers = sample.get("answers", [])
    if answers is None:
        return []
    if isinstance(answers, np.ndarray):
        answers = answers.tolist()
    return [str(a) for a in answers if a is not None]


def get_ocr_tokens(sample: Dict[str, Any]) -> List[str]:
    tokens = sample.get("ocr_tokens", [])
    if tokens is None:
        return []
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    return [str(t) for t in tokens if t is not None and str(t).strip()]


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "isnt": "isn't",
    "shouldnt": "shouldn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
}
_MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    """Approximate the official VQA/TextVQA answer normalization."""
    if answer is None:
        return ""
    answer = str(answer).lower().replace("\n", " ").replace("\t", " ").strip()
    answer = answer.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)
    punct = set(string.punctuation) - {"'", ":"}
    answer = "".join(" " if ch in punct else ch for ch in answer)
    words = []
    for word in answer.split():
        word = _MANUAL_MAP.get(word, word)
        word = _CONTRACTIONS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    return " ".join(words)


def textvqa_accuracy(prediction: str, ground_truths: Sequence[str]) -> float:
    if not ground_truths:
        return 0.0
    pred = normalize_answer(prediction)
    gts = [normalize_answer(gt) for gt in ground_truths]
    return min(1.0, sum(gt == pred for gt in gts) / 3.0)


def exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gt) for gt in ground_truths))


def token_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens) if gt_tokens else 0.0
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def substring_scores(prediction: str, ground_truths: Sequence[str]) -> Tuple[float, float]:
    pred = normalize_answer(prediction)
    refs = [normalize_answer(gt) for gt in ground_truths]
    if not pred or not refs:
        return 0.0, 0.0
    precision = float(any(pred in ref for ref in refs))
    recall = float(any(ref in pred for ref in refs if ref))
    return precision, recall


def bleu_score(prediction: str, ground_truths: Sequence[str]) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        refs = [normalize_answer(gt).split() for gt in ground_truths if normalize_answer(gt)]
        pred = normalize_answer(prediction).split()
        if not refs or not pred:
            return 0.0
        weights = (1.0, 0, 0, 0) if len(pred) < 2 else (0.5, 0.5, 0, 0)
        return float(sentence_bleu(refs, pred, weights=weights, smoothing_function=SmoothingFunction().method1))
    except Exception:
        return 0.0


def meteor_score_best(prediction: str, ground_truths: Sequence[str]) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score

        pred = normalize_answer(prediction).split()
        refs = [normalize_answer(gt).split() for gt in ground_truths if normalize_answer(gt)]
        if not pred or not refs:
            return 0.0
        return float(meteor_score(refs, pred))
    except Exception:
        return token_f1(prediction, ground_truths)


def rouge_l_best(prediction: str, ground_truths: Sequence[str]) -> float:
    def lcs(a: List[str], b: List[str]) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, ai in enumerate(a, start=1):
            for j, bj in enumerate(b, start=1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if ai == bj else max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    pred = normalize_answer(prediction).split()
    if not pred:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        ref = normalize_answer(gt).split()
        if not ref:
            continue
        overlap = lcs(pred, ref)
        precision = overlap / len(pred)
        recall = overlap / len(ref)
        if precision + recall:
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def categorize_question(question: str) -> str:
    q = question.lower().strip()
    patterns = [
        ("brand", r"\bbrand\b|\blogo\b|\bcompany\b|\bmake\b"),
        ("number", r"\bnumber\b|\bhow many\b|\bprice\b|\bcost\b|\bamount\b|\btotal\b|\bphone\b|\btime\b|\bdate\b|\byear\b"),
        ("color", r"\bcolor\b|\bcolour\b"),
        ("place", r"\bwhere\b|\bplace\b|\blocation\b|\bcity\b|\bcountry\b"),
        ("person", r"\bwho\b|\bname of (?:the )?person\b"),
        ("reading_text", r"\bwhat does\b|\bwhat is written\b|\bread\b|\bsay\b|\btext\b|\bword\b|\bletter\b"),
        ("object", r"\bwhat is this\b|\bwhat type\b|\bwhat kind\b|\bwhat item\b"),
    ]
    for label, pat in patterns:
        if re.search(pat, q):
            return label
    if q.startswith("what"):
        return "what_general"
    if q.startswith("which"):
        return "which"
    return "other"


def compute_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in results:
        pred = row.get("prediction", "")
        gts = row.get("ground_truths", [])
        sub_p, sub_r = substring_scores(pred, gts)
        rows.append(
            {
                "accuracy": textvqa_accuracy(pred, gts),
                "exact_match": exact_match(pred, gts),
                "f1": token_f1(pred, gts),
                "bleu": bleu_score(pred, gts),
                "meteor": meteor_score_best(pred, gts),
                "rouge_l": rouge_l_best(pred, gts),
                "substring_precision": sub_p,
                "substring_recall": sub_r,
            }
        )
    metrics = {"num_samples": len(rows)}
    for key in ["accuracy", "exact_match", "f1", "bleu", "meteor", "rouge_l", "substring_precision", "substring_recall"]:
        metrics[key] = float(np.mean([r[key] for r in rows])) if rows else 0.0
    judge_scores = [r.get("llm_judge_score") for r in results if r.get("llm_judge_score") is not None]
    if judge_scores:
        metrics["llm_judge_similarity"] = float(np.mean(judge_scores))

    by_cat: Dict[str, List[float]] = defaultdict(list)
    for result in results:
        by_cat[categorize_question(result.get("question", ""))].append(
            textvqa_accuracy(result.get("prediction", ""), result.get("ground_truths", []))
        )
    metrics["per_category"] = {
        cat: {"count": len(vals), "accuracy": float(np.mean(vals)) if vals else 0.0}
        for cat, vals in sorted(by_cat.items())
    }
    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Metrics") -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Samples: {metrics.get('num_samples', 0)}")
    for key in ["accuracy", "exact_match", "f1", "bleu", "meteor", "rouge_l", "substring_precision", "substring_recall"]:
        print(f"{key:>20}: {metrics.get(key, 0.0) * 100:6.2f}")
    if "per_category" in metrics:
        print("\nPer-category accuracy:")
        for cat, stat in sorted(metrics["per_category"].items(), key=lambda x: -x[1]["count"]):
            print(f"{cat:>20}: {stat['accuracy'] * 100:6.2f}  n={stat['count']}")


def _preferred_torch_dtype():
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def get_model_and_processor(config: RunConfig):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    dtype = _preferred_torch_dtype()
    quantization_config = None
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor


def build_messages(question: str, ocr_tokens: Optional[Sequence[str]] = None, prompt_style: str = "concise", answer: Optional[str] = None):
    if prompt_style == "ocr" and ocr_tokens:
        short_ocr = ", ".join(list(ocr_tokens)[:80])
        prompt = (
            f"OCR tokens detected in the image: {short_ocr}\n"
            f"Question: {question}\n"
            "Answer with only the exact text, number, name, or short phrase requested."
        )
    elif prompt_style == "strict_ocr" and ocr_tokens:
        short_ocr = ", ".join(list(ocr_tokens)[:50])
        prompt = (
            "Use the image and the OCR token list to answer. "
            "Prefer a token from the list when it directly answers the question.\n"
            f"OCR tokens: {short_ocr}\n"
            f"Question: {question}\n"
            "Final answer only:"
        )
    elif prompt_style == "plain":
        prompt = question
    else:
        prompt = f"{question}\nAnswer with only the exact text, number, name, or short phrase requested."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def generation_kwargs(config: RunConfig) -> Dict[str, Any]:
    kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "pad_token_id": 151643,
    }
    if config.temperature > 0:
        kwargs.update({"temperature": config.temperature, "top_p": config.top_p})
    return kwargs


def run_inference(
    model,
    processor,
    dataset,
    config: RunConfig,
    output_prefix: str,
    prompt_style: str = "concise",
    max_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch
    from tqdm.auto import tqdm

    dirs = ensure_dirs(config.project_dir)
    n = min(len(dataset), max_samples or len(dataset))
    batch_size = batch_size or config.inference_batch_size
    results: List[Dict[str, Any]] = []
    next_partial_save = 250
    model.eval()

    for start in tqdm(range(0, n, batch_size), desc=f"inference:{output_prefix}"):
        batch_indices = list(range(start, min(start + batch_size, n)))
        samples = [dataset[int(idx)] for idx in batch_indices]
        images = []
        texts = []
        metadata = []
        for idx, sample in zip(batch_indices, samples):
            image = get_image(sample)
            question = str(sample.get("question", ""))
            answers = get_answers(sample)
            ocr_tokens = get_ocr_tokens(sample)
            messages = build_messages(question, ocr_tokens, prompt_style=prompt_style)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images.append(image)
            texts.append(text)
            metadata.append((idx, sample, question, answers, ocr_tokens))

        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        device = _model_device(model)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs(config))
        generated_ids = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs["input_ids"], output_ids)
        ]
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for prediction, (idx, sample, question, answers, ocr_tokens) in zip(predictions, metadata):
            results.append(
                {
                    "index": int(idx),
                    "image_id": str(sample.get("image_id", "")),
                    "question_id": str(sample.get("question_id", "")),
                    "question": question,
                    "prediction": prediction.strip(),
                    "ground_truths": answers,
                    "ocr_tokens": ocr_tokens,
                    "prompt_style": prompt_style,
                }
            )
        if len(results) >= next_partial_save:
            save_json(results, dirs["results"] / f"{output_prefix}_predictions.partial.json")
            next_partial_save += 250

    metrics = compute_metrics(results)
    save_json(results, dirs["results"] / f"{output_prefix}_predictions.json")
    save_json(metrics, dirs["results"] / f"{output_prefix}_metrics.json")
    print_metrics(metrics, output_prefix)
    return results, metrics


def analyze_predictions(prediction_path: str | Path, output_prefix: str, project_dir: str) -> Dict[str, Any]:
    results = load_json(prediction_path)
    error_counts = Counter()
    examples = defaultdict(list)
    category_errors = defaultdict(Counter)

    for row in results:
        pred = row.get("prediction", "")
        gts = row.get("ground_truths", [])
        ocr_tokens = [normalize_answer(t) for t in row.get("ocr_tokens", [])]
        pred_norm = normalize_answer(pred)
        gt_norms = [normalize_answer(gt) for gt in gts]
        cat = categorize_question(row.get("question", ""))
        if textvqa_accuracy(pred, gts) > 0:
            etype = "correct"
        elif any(pred_norm and (pred_norm in gt or gt in pred_norm) for gt in gt_norms):
            etype = "format_or_partial_match"
        elif any(gt and any(gt in tok or tok in gt for tok in ocr_tokens) for gt in gt_norms) and not any(pred_norm and pred_norm in tok for tok in ocr_tokens):
            etype = "failed_to_use_visible_ocr"
        elif pred_norm and any(pred_norm in tok for tok in ocr_tokens):
            etype = "selected_wrong_ocr_token"
        elif pred_norm:
            etype = "hallucination_or_visual_reasoning"
        else:
            etype = "empty_answer"

        error_counts[etype] += 1
        category_errors[cat][etype] += 1
        if len(examples[etype]) < 8:
            examples[etype].append(
                {
                    "question": row.get("question", ""),
                    "prediction": pred,
                    "ground_truths": gts[:5],
                    "ocr_tokens": row.get("ocr_tokens", [])[:20],
                }
            )

    analysis = {
        "total": len(results),
        "error_distribution": dict(error_counts),
        "category_error_distribution": {k: dict(v) for k, v in category_errors.items()},
        "examples": dict(examples),
    }
    dirs = ensure_dirs(project_dir)
    save_json(analysis, dirs["results"] / f"{output_prefix}_analysis.json")
    return analysis


def compare_prediction_files(path_a: str | Path, path_b: str | Path, name_a: str, name_b: str, project_dir: str) -> Dict[str, Any]:
    a_rows = load_json(path_a)
    b_rows = load_json(path_b)
    by_qid_b = {row.get("question_id") or row.get("index"): row for row in b_rows}
    transitions = Counter()
    examples = defaultdict(list)

    for a in a_rows:
        key = a.get("question_id") or a.get("index")
        b = by_qid_b.get(key)
        if not b:
            continue
        a_ok = textvqa_accuracy(a.get("prediction", ""), a.get("ground_truths", [])) > 0
        b_ok = textvqa_accuracy(b.get("prediction", ""), b.get("ground_truths", [])) > 0
        label = f"{name_a}_{'correct' if a_ok else 'wrong'}__{name_b}_{'correct' if b_ok else 'wrong'}"
        transitions[label] += 1
        if len(examples[label]) < 8:
            examples[label].append(
                {
                    "question": a.get("question", ""),
                    f"{name_a}_prediction": a.get("prediction", ""),
                    f"{name_b}_prediction": b.get("prediction", ""),
                    "ground_truths": a.get("ground_truths", [])[:5],
                }
            )

    comparison = {"transitions": dict(transitions), "examples": dict(examples)}
    dirs = ensure_dirs(project_dir)
    save_json(comparison, dirs["results"] / f"comparison_{name_a}_vs_{name_b}.json")
    return comparison


def make_plots(metric_files: Sequence[str | Path], project_dir: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    dirs = ensure_dirs(project_dir)
    rows = []
    for path in metric_files:
        metrics = load_json(path)
        name = Path(path).name.replace("_metrics.json", "")
        for key in ["accuracy", "exact_match", "f1", "bleu", "meteor", "rouge_l"]:
            rows.append({"model": name, "metric": key, "score": metrics.get(key, 0.0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="metric", y="score", hue="model")
    plt.ylim(0, max(0.05, min(1.0, df["score"].max() * 1.25)))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / "metric_comparison.png", dpi=200)
    plt.close()


def create_report_assets(
    prefixes: Sequence[str],
    project_dir: str,
    baseline_prefix: Optional[str] = None,
    best_prefix: Optional[str] = None,
    output_name: str = "prompt_report_assets",
) -> Dict[str, Any]:
    """Create report-ready tables, plots, and qualitative examples.

    Expects files named `{prefix}_metrics.json`, `{prefix}_predictions.json`,
    and optionally `{prefix}_analysis.json` in the results directory.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    dirs = ensure_dirs(project_dir)
    tables_dir = dirs["results"] / "report_tables"
    examples_dir = dirs["results"] / "report_examples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "accuracy",
        "exact_match",
        "f1",
        "bleu",
        "meteor",
        "rouge_l",
        "substring_precision",
        "substring_recall",
    ]
    display_names = {
        "accuracy": "TextVQA Accuracy",
        "exact_match": "Exact Match",
        "f1": "Token F1",
        "bleu": "BLEU",
        "meteor": "METEOR",
        "rouge_l": "ROUGE-L",
        "substring_precision": "Substring Precision",
        "substring_recall": "Substring Recall",
    }

    metrics_by_prefix: Dict[str, Dict[str, Any]] = {}
    for prefix in prefixes:
        path = dirs["results"] / f"{prefix}_metrics.json"
        if path.exists():
            metrics_by_prefix[prefix] = load_json(path)

    if not metrics_by_prefix:
        raise FileNotFoundError("No metrics files found for the provided prefixes.")

    baseline_prefix = baseline_prefix or prefixes[0]
    best_prefix = best_prefix or max(metrics_by_prefix, key=lambda p: metrics_by_prefix[p].get("accuracy", 0.0))

    metric_rows = []
    for prefix, metrics in metrics_by_prefix.items():
        row = {"run": prefix, "num_samples": metrics.get("num_samples", 0)}
        for key in metric_keys:
            row[display_names[key]] = 100 * metrics.get(key, 0.0)
        metric_rows.append(row)

    metric_df = pd.DataFrame(metric_rows).sort_values("TextVQA Accuracy", ascending=False)
    metric_csv = tables_dir / f"{output_name}_metrics_table.csv"
    metric_md = tables_dir / f"{output_name}_metrics_table.md"
    metric_tex = tables_dir / f"{output_name}_metrics_table.tex"
    metric_df.to_csv(metric_csv, index=False)
    metric_df.to_markdown(metric_md, index=False, floatfmt=".2f")
    metric_df.to_latex(metric_tex, index=False, float_format="%.2f")

    if baseline_prefix in metrics_by_prefix:
        base_metrics = metrics_by_prefix[baseline_prefix]
        improvement_rows = []
        for prefix, metrics in metrics_by_prefix.items():
            if prefix == baseline_prefix:
                continue
            row = {"baseline": baseline_prefix, "run": prefix}
            for key in metric_keys:
                row[f"{display_names[key]} Delta"] = 100 * (metrics.get(key, 0.0) - base_metrics.get(key, 0.0))
            improvement_rows.append(row)
        improvement_df = pd.DataFrame(improvement_rows)
        improvement_csv = tables_dir / f"{output_name}_improvements_vs_baseline.csv"
        improvement_md = tables_dir / f"{output_name}_improvements_vs_baseline.md"
        improvement_df.to_csv(improvement_csv, index=False)
        improvement_df.to_markdown(improvement_md, index=False, floatfmt=".2f")

    category_rows = []
    for prefix, metrics in metrics_by_prefix.items():
        for category, stat in metrics.get("per_category", {}).items():
            category_rows.append(
                {
                    "run": prefix,
                    "category": category,
                    "count": stat.get("count", 0),
                    "accuracy": 100 * stat.get("accuracy", 0.0),
                }
            )
    category_df = pd.DataFrame(category_rows)
    category_csv = tables_dir / f"{output_name}_category_accuracy.csv"
    category_md = tables_dir / f"{output_name}_category_accuracy.md"
    category_df.to_csv(category_csv, index=False)
    category_df.to_markdown(category_md, index=False, floatfmt=".2f")

    sample_rows = []
    predictions_by_prefix: Dict[str, List[Dict[str, Any]]] = {}
    for prefix in prefixes:
        pred_path = dirs["results"] / f"{prefix}_predictions.json"
        if not pred_path.exists():
            continue
        rows = load_json(pred_path)
        predictions_by_prefix[prefix] = rows
        for row in rows:
            gts = row.get("ground_truths", [])
            pred = row.get("prediction", "")
            sample_rows.append(
                {
                    "run": prefix,
                    "index": row.get("index"),
                    "question_id": row.get("question_id"),
                    "image_id": row.get("image_id"),
                    "category": categorize_question(row.get("question", "")),
                    "accuracy": textvqa_accuracy(pred, gts),
                    "exact_match": exact_match(pred, gts),
                    "prediction_token_length": len(normalize_answer(pred).split()),
                    "reference_token_length": np.mean([len(normalize_answer(gt).split()) for gt in gts]) if gts else 0,
                    "ocr_token_count": len(row.get("ocr_tokens", [])),
                    "question": row.get("question", ""),
                    "prediction": pred,
                }
            )
    sample_df = pd.DataFrame(sample_rows)
    sample_csv = tables_dir / f"{output_name}_per_sample_diagnostics.csv"
    sample_df.to_csv(sample_csv, index=False)

    plt.figure(figsize=(11, 5))
    plot_df = metric_df.melt(
        id_vars=["run", "num_samples"],
        value_vars=[display_names[k] for k in ["accuracy", "exact_match", "f1", "meteor", "rouge_l"]],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=plot_df, x="metric", y="score", hue="run")
    plt.ylabel("Score (%)")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    metric_fig = dirs["figures"] / f"{output_name}_metric_comparison.png"
    plt.savefig(metric_fig, dpi=220)
    plt.close()

    if not category_df.empty:
        heatmap_df = category_df.pivot_table(index="category", columns="run", values="accuracy")
        ordered_categories = (
            category_df.groupby("category")["count"].max().sort_values(ascending=False).index.tolist()
        )
        heatmap_df = heatmap_df.reindex(ordered_categories)
        plt.figure(figsize=(10, max(4, 0.45 * len(heatmap_df))))
        sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="viridis", cbar_kws={"label": "Accuracy (%)"})
        plt.xlabel("")
        plt.ylabel("Question Category")
        plt.tight_layout()
        category_heatmap = dirs["figures"] / f"{output_name}_category_heatmap.png"
        plt.savefig(category_heatmap, dpi=220)
        plt.close()

        plt.figure(figsize=(11, 5))
        top_categories = ordered_categories[:8]
        sns.barplot(data=category_df[category_df["category"].isin(top_categories)], x="category", y="accuracy", hue="run")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Question Category")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        category_bar = dirs["figures"] / f"{output_name}_category_bars.png"
        plt.savefig(category_bar, dpi=220)
        plt.close()

        if baseline_prefix in metrics_by_prefix and best_prefix in metrics_by_prefix:
            base_cat = {
                cat: stat.get("accuracy", 0.0)
                for cat, stat in metrics_by_prefix[baseline_prefix].get("per_category", {}).items()
            }
            best_cat = {
                cat: stat.get("accuracy", 0.0)
                for cat, stat in metrics_by_prefix[best_prefix].get("per_category", {}).items()
            }
            delta_rows = []
            for cat in sorted(set(base_cat) | set(best_cat)):
                count = metrics_by_prefix[best_prefix].get("per_category", {}).get(cat, {}).get("count", 0)
                delta_rows.append(
                    {
                        "category": cat,
                        "count": count,
                        "baseline_accuracy": 100 * base_cat.get(cat, 0.0),
                        "best_accuracy": 100 * best_cat.get(cat, 0.0),
                        "absolute_delta": 100 * (best_cat.get(cat, 0.0) - base_cat.get(cat, 0.0)),
                    }
                )
            delta_df = pd.DataFrame(delta_rows).sort_values("absolute_delta", ascending=False)
            delta_csv = tables_dir / f"{output_name}_category_improvement.csv"
            delta_md = tables_dir / f"{output_name}_category_improvement.md"
            delta_df.to_csv(delta_csv, index=False)
            delta_df.to_markdown(delta_md, index=False, floatfmt=".2f")
            plt.figure(figsize=(10, 5))
            sns.barplot(data=delta_df, y="category", x="absolute_delta", color="#2a9d8f")
            plt.axvline(0, color="black", linewidth=0.8)
            plt.xlabel("Accuracy Improvement Over Concise Baseline (percentage points)")
            plt.ylabel("")
            plt.tight_layout()
            category_delta_fig = dirs["figures"] / f"{output_name}_category_improvement.png"
            plt.savefig(category_delta_fig, dpi=220)
            plt.close()

    if not sample_df.empty:
        binned = sample_df.copy()
        binned["ocr_bin"] = pd.cut(
            binned["ocr_token_count"],
            bins=[-1, 0, 5, 15, 40, 10_000],
            labels=["0", "1-5", "6-15", "16-40", "41+"],
        )
        ocr_table = (
            binned.groupby(["run", "ocr_bin"], observed=False)["accuracy"]
            .mean()
            .reset_index()
        )
        ocr_table["accuracy"] *= 100
        ocr_csv = tables_dir / f"{output_name}_accuracy_by_ocr_count.csv"
        ocr_md = tables_dir / f"{output_name}_accuracy_by_ocr_count.md"
        ocr_table.to_csv(ocr_csv, index=False)
        ocr_table.to_markdown(ocr_md, index=False, floatfmt=".2f")
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=ocr_table, x="ocr_bin", y="accuracy", hue="run", marker="o")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Number of OCR Tokens in Image")
        plt.tight_layout()
        ocr_fig = dirs["figures"] / f"{output_name}_accuracy_by_ocr_count.png"
        plt.savefig(ocr_fig, dpi=220)
        plt.close()

        length_df = sample_df.copy()
        length_df["reference_length_bin"] = pd.cut(
            length_df["reference_token_length"],
            bins=[-0.1, 1, 3, 6, 100],
            labels=["1 token", "2-3 tokens", "4-6 tokens", "7+ tokens"],
        )
        length_table = (
            length_df.groupby(["run", "reference_length_bin"], observed=False)["accuracy"]
            .mean()
            .reset_index()
        )
        length_table["accuracy"] *= 100
        length_csv = tables_dir / f"{output_name}_accuracy_by_answer_length.csv"
        length_md = tables_dir / f"{output_name}_accuracy_by_answer_length.md"
        length_table.to_csv(length_csv, index=False)
        length_table.to_markdown(length_md, index=False, floatfmt=".2f")
        plt.figure(figsize=(9, 5))
        sns.barplot(data=length_table, x="reference_length_bin", y="accuracy", hue="run")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Average Reference Answer Length")
        plt.tight_layout()
        length_fig = dirs["figures"] / f"{output_name}_accuracy_by_answer_length.png"
        plt.savefig(length_fig, dpi=220)
        plt.close()

    error_rows = []
    for prefix in prefixes:
        analysis_path = dirs["results"] / f"{prefix}_analysis.json"
        if not analysis_path.exists():
            continue
        analysis = load_json(analysis_path)
        total = max(1, analysis.get("total", 0))
        for error_type, count in analysis.get("error_distribution", {}).items():
            error_rows.append(
                {
                    "run": prefix,
                    "error_type": error_type,
                    "count": count,
                    "percent": 100 * count / total,
                }
            )
    error_df = pd.DataFrame(error_rows)
    if not error_df.empty:
        error_csv = tables_dir / f"{output_name}_error_distribution.csv"
        error_md = tables_dir / f"{output_name}_error_distribution.md"
        error_df.to_csv(error_csv, index=False)
        error_df.to_markdown(error_md, index=False, floatfmt=".2f")
        pivot = error_df.pivot_table(index="run", columns="error_type", values="percent", fill_value=0)
        pivot.plot(kind="bar", stacked=True, figsize=(11, 5), colormap="tab20")
        plt.ylabel("Share of Samples (%)")
        plt.xlabel("")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        error_fig = dirs["figures"] / f"{output_name}_error_distribution.png"
        plt.savefig(error_fig, dpi=220)
        plt.close()

    transition_csv = None
    transition_fig = None
    improved_examples_csv = None
    if baseline_prefix in predictions_by_prefix and best_prefix in predictions_by_prefix:
        base_rows = predictions_by_prefix[baseline_prefix]
        best_rows = predictions_by_prefix[best_prefix]
        best_by_key = {row.get("question_id") or row.get("index"): row for row in best_rows}
        transition_rows = []
        example_rows = []
        for base in base_rows:
            key = base.get("question_id") or base.get("index")
            best = best_by_key.get(key)
            if not best:
                continue
            gts = base.get("ground_truths", [])
            base_acc = textvqa_accuracy(base.get("prediction", ""), gts)
            best_acc = textvqa_accuracy(best.get("prediction", ""), gts)
            base_status = "correct" if base_acc > 0 else "wrong"
            best_status = "correct" if best_acc > 0 else "wrong"
            transition_rows.append({"baseline": base_status, "best_prompt": best_status})
            if base_status != best_status or abs(best_acc - base_acc) > 0:
                example_rows.append(
                    {
                        "change": f"{base_status}_to_{best_status}",
                        "question": base.get("question", ""),
                        "baseline_prediction": base.get("prediction", ""),
                        "best_prediction": best.get("prediction", ""),
                        "ground_truths": gts[:5],
                        "ocr_tokens": base.get("ocr_tokens", [])[:20],
                        "baseline_accuracy": base_acc,
                        "best_accuracy": best_acc,
                    }
                )
        transition_df = pd.DataFrame(transition_rows)
        if not transition_df.empty:
            transition_table = pd.crosstab(transition_df["baseline"], transition_df["best_prompt"])
            transition_csv = tables_dir / f"{output_name}_baseline_to_best_transition.csv"
            transition_table.to_csv(transition_csv)
            plt.figure(figsize=(5, 4))
            sns.heatmap(transition_table, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Best Prompt")
            plt.ylabel("Concise Baseline")
            plt.tight_layout()
            transition_fig = dirs["figures"] / f"{output_name}_baseline_to_best_transition.png"
            plt.savefig(transition_fig, dpi=220)
            plt.close()
        if example_rows:
            example_df = pd.DataFrame(example_rows)
            example_df["sort_key"] = example_df["best_accuracy"] - example_df["baseline_accuracy"]
            example_df = example_df.sort_values("sort_key", ascending=False).drop(columns=["sort_key"])
            improved_examples_csv = examples_dir / f"{output_name}_improved_and_regressed_examples.csv"
            example_df.to_csv(improved_examples_csv, index=False)

    qualitative_md = examples_dir / f"{output_name}_qualitative_examples.md"
    lines = ["# Qualitative Examples\n"]
    for prefix in [best_prefix, baseline_prefix]:
        pred_path = dirs["results"] / f"{prefix}_predictions.json"
        if not pred_path.exists():
            continue
        rows = load_json(pred_path)
        lines.append(f"\n## {prefix}\n")
        correct = []
        incorrect = []
        for row in rows:
            item = {
                "question": row.get("question", ""),
                "prediction": row.get("prediction", ""),
                "ground_truths": row.get("ground_truths", [])[:5],
                "ocr_tokens": row.get("ocr_tokens", [])[:12],
            }
            if textvqa_accuracy(row.get("prediction", ""), row.get("ground_truths", [])) > 0:
                correct.append(item)
            else:
                incorrect.append(item)
        for label, examples in [("Correct", correct[:6]), ("Incorrect", incorrect[:8])]:
            lines.append(f"\n### {label}\n")
            for ex in examples:
                lines.append(f"- Q: {ex['question']}")
                lines.append(f"  - Prediction: `{ex['prediction']}`")
                lines.append(f"  - Ground truth examples: {ex['ground_truths']}")
                lines.append(f"  - OCR tokens: {ex['ocr_tokens']}")
    qualitative_md.write_text("\n".join(lines), encoding="utf-8")

    assets = {
        "metrics_table_csv": str(metric_csv),
        "metrics_table_markdown": str(metric_md),
        "metrics_table_latex": str(metric_tex),
        "per_sample_diagnostics_csv": str(sample_csv),
        "category_accuracy_csv": str(category_csv),
        "category_accuracy_markdown": str(category_md),
        "metric_comparison_figure": str(metric_fig),
        "qualitative_examples_markdown": str(qualitative_md),
        "best_prefix": best_prefix,
        "baseline_prefix": baseline_prefix,
    }
    for name, value in {
        "transition_table_csv": transition_csv,
        "transition_figure": transition_fig,
        "improved_and_regressed_examples_csv": improved_examples_csv,
    }.items():
        if value:
            assets[name] = str(value)
    save_json(assets, dirs["results"] / f"{output_name}_manifest.json")
    return assets


def summarize_prompt_runs(prefixes: Sequence[str], project_dir: str, output_name: str = "prompt_ablation_summary") -> Dict[str, Any]:
    import pandas as pd

    dirs = ensure_dirs(project_dir)
    rows = []
    for prefix in prefixes:
        metrics_path = dirs["results"] / f"{prefix}_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        rows.append(
            {
                "run": prefix,
                "accuracy": metrics.get("accuracy", 0.0),
                "exact_match": metrics.get("exact_match", 0.0),
                "f1": metrics.get("f1", 0.0),
                "bleu": metrics.get("bleu", 0.0),
                "meteor": metrics.get("meteor", 0.0),
                "rouge_l": metrics.get("rouge_l", 0.0),
                "substring_precision": metrics.get("substring_precision", 0.0),
                "substring_recall": metrics.get("substring_recall", 0.0),
                "num_samples": metrics.get("num_samples", 0),
            }
        )
    rows = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    summary = {"runs": rows, "best_run": rows[0] if rows else None}
    save_json(summary, dirs["results"] / f"{output_name}.json")
    if rows:
        pd.DataFrame(rows).to_csv(dirs["results"] / f"{output_name}.csv", index=False)
    return summary


def build_llm_judge_prompt(question: str, prediction: str, references: Sequence[str]) -> str:
    refs = "; ".join(str(ref) for ref in references[:10])
    return (
        "You are a strict evaluator for TextVQA, a short-answer visual question answering benchmark.\n"
        "Decide whether the model prediction actually answers the question with the same meaning as "
        "at least one reference answer.\n\n"
        "Important grading rules:\n"
        "- Answer YES only if the prediction contains the requested answer or a clearly equivalent paraphrase.\n"
        "- Answer NO for incomplete sentence fragments that merely restate the question.\n"
        "- Answer NO for generic descriptions that do not include the actual answer string.\n"
        "- Answer NO if the prediction is the wrong entity, wrong number, wrong time, or wrong yes/no polarity.\n"
        "- Minor casing, punctuation, articles, currency symbols, and harmless units may still be YES.\n"
        "- For yes/no questions, a prediction must clearly imply the correct yes/no answer.\n\n"
        "Calibration examples:\n"
        "Question: what is the brand of this camera?\n"
        "References: dakota; dakota digital\n"
        "Prediction: The brand of the camera in the picture\n"
        "Judgment: NO\n\n"
        "Question: what number is on the player's jersey?\n"
        "References: 22\n"
        "Prediction: The number on the player's jersey is 22\n"
        "Judgment: YES\n\n"
        "Question: how many watts for this powersupply?\n"
        "References: 400\n"
        "Prediction: 400W\n"
        "Judgment: YES\n\n"
        "Question: are these bottles of pepsi?\n"
        "References: yes\n"
        "Prediction: Denny's\n"
        "Judgment: NO\n\n"
        "Now evaluate the real example below. Reply with exactly one token: YES or NO.\n\n"
        f"Question: {question}\n"
        f"References: {refs}\n"
        f"Prediction: {prediction}\n"
        "Judgment:"
    )


_JUDGE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "been",
    "being",
    "by",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _judge_content_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_answer(text).split()
        if token and token not in _JUDGE_STOPWORDS
    }


def _has_reference_evidence(prediction: str, references: Sequence[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    pred_tokens = set(pred_norm.split())
    for ref in references:
        ref_norm = normalize_answer(ref)
        if not ref_norm:
            continue
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return True
        ref_tokens = _judge_content_tokens(ref_norm)
        if ref_tokens and pred_tokens.intersection(ref_tokens):
            return True
    return False


def strict_judge_prefilter(
    question: str,
    prediction: str,
    references: Sequence[str],
) -> Tuple[Optional[float], str]:
    """Catch obvious non-answers before asking the LLM judge.

    The previous small judge was too permissive for outputs that repeated the
    question without giving the answer. This prefilter only rejects cases with
    no lexical evidence for any reference answer and strong signs of an
    incomplete or question-restating prediction; ambiguous cases still go to
    the LLM judge.
    """
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return 0.0, "AUTO_EMPTY_PREDICTION"

    if _has_reference_evidence(prediction, references):
        return None, "NEEDS_LLM"

    pred_tokens = _judge_content_tokens(prediction)
    question_tokens = _judge_content_tokens(question)
    if not pred_tokens:
        return 0.0, "AUTO_NO_CONTENT"

    question_overlap = len(pred_tokens.intersection(question_tokens)) / max(1, len(pred_tokens))
    last_token = pred_norm.split()[-1]
    incomplete_endings = {
        "about",
        "approximately",
        "are",
        "called",
        "has",
        "in",
        "is",
        "named",
        "of",
        "on",
        "reads",
        "says",
        "set",
        "spells",
        "was",
        "were",
        "with",
    }
    if len(pred_tokens) >= 3 and question_overlap >= 0.40:
        return 0.0, "AUTO_QUESTION_RESTATEMENT"
    if len(pred_norm.split()) >= 3 and last_token in incomplete_endings:
        return 0.0, "AUTO_INCOMPLETE_FRAGMENT"
    return None, "NEEDS_LLM"


def judge_predictions_with_llm(
    prefixes: Sequence[str],
    project_dir: str,
    judge_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_size: int = 16,
    output_name: str = "llm_judge_similarity",
    use_strict_prefilter: bool = True,
    resume: bool = True,
    checkpoint_every_batches: int = 50,
) -> Dict[str, Any]:
    """Run a lightweight text-only LLM-as-a-Judge pass on saved predictions.

    Exact TextVQA matches are assigned a judge score of 1 without an LLM call.
    Empty predictions are assigned 0. The judge model is called only for
    non-empty, non-exact cases.
    """
    import torch
    import pandas as pd
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dirs = ensure_dirs(project_dir)
    tables_dir = dirs["results"] / "report_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _preferred_torch_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_rows: List[Dict[str, Any]] = []
    summary_rows = []

    for prefix in prefixes:
        pred_path = dirs["results"] / f"{prefix}_predictions.json"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        predictions = load_json(pred_path)
        detail_path = dirs["results"] / f"{prefix}_{output_name}_details.json"
        partial_detail_path = dirs["results"] / f"{prefix}_{output_name}_partial_details.json"

        rows = []
        if resume:
            resume_path = detail_path if detail_path.exists() else partial_detail_path
            if resume_path.exists():
                rows = load_json(resume_path)
                print(f"Resuming {prefix} from {resume_path} ({len(rows)} rows already judged).")
        completed_indices = {str(row.get("index", "")) for row in rows}
        pending = []
        for idx, row in enumerate(predictions):
            row_index = str(row.get("index", idx))
            if row_index in completed_indices:
                continue
            pred = row.get("prediction", "")
            gts = row.get("ground_truths", [])
            exact_score = textvqa_accuracy(pred, gts)
            if exact_score > 0:
                rows.append(
                    {
                        "run": prefix,
                        "index": row.get("index", idx),
                        "question_id": row.get("question_id", ""),
                        "question": row.get("question", ""),
                        "prediction": pred,
                        "ground_truths": gts,
                        "textvqa_accuracy": exact_score,
                        "llm_judge_score": 1.0,
                        "judge_raw": "AUTO_EXACT_MATCH",
                        "judge_source": "AUTO_EXACT_MATCH",
                    }
                )
                continue

            if use_strict_prefilter:
                prefilter_score, prefilter_reason = strict_judge_prefilter(
                    row.get("question", ""),
                    pred,
                    gts,
                )
                if prefilter_score is not None:
                    rows.append(
                        {
                            "run": prefix,
                            "index": row.get("index", idx),
                            "question_id": row.get("question_id", ""),
                            "question": row.get("question", ""),
                            "prediction": pred,
                            "ground_truths": gts,
                            "textvqa_accuracy": exact_score,
                            "llm_judge_score": float(prefilter_score),
                            "judge_raw": prefilter_reason,
                            "judge_source": prefilter_reason,
                        }
                    )
                    continue

            if not normalize_answer(pred):
                rows.append(
                    {
                        "run": prefix,
                        "index": row.get("index", idx),
                        "question_id": row.get("question_id", ""),
                        "question": row.get("question", ""),
                        "prediction": pred,
                        "ground_truths": gts,
                        "textvqa_accuracy": exact_score,
                        "llm_judge_score": 0.0,
                        "judge_raw": "AUTO_EMPTY_PREDICTION",
                        "judge_source": "AUTO_EMPTY_PREDICTION",
                    }
                )
            else:
                pending.append((idx, row, build_llm_judge_prompt(row.get("question", ""), pred, gts)))

        for batch_idx, start in enumerate(tqdm(range(0, len(pending), batch_size), desc=f"judge:{prefix}"), start=1):
            chunk = pending[start : start + batch_size]
            messages = [
                [
                    {
                        "role": "system",
                        "content": "You are a strict evaluator for visual question answering. Reply only YES or NO.",
                    },
                    {"role": "user", "content": prompt},
                ]
                for _, _, prompt in chunk
            ]
            texts = [
                tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs["input_ids"], output_ids)
            ]
            judge_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for (_, row, _), judge_raw in zip(chunk, judge_outputs):
                clean = judge_raw.strip().lower()
                score = 1.0 if clean.startswith("yes") else 0.0
                rows.append(
                    {
                        "run": prefix,
                        "index": row.get("index", ""),
                        "question_id": row.get("question_id", ""),
                        "question": row.get("question", ""),
                        "prediction": row.get("prediction", ""),
                        "ground_truths": row.get("ground_truths", []),
                        "textvqa_accuracy": textvqa_accuracy(row.get("prediction", ""), row.get("ground_truths", [])),
                        "llm_judge_score": score,
                        "judge_raw": judge_raw.strip(),
                        "judge_source": "LLM_CALL",
                    }
                )
            if checkpoint_every_batches and batch_idx % checkpoint_every_batches == 0:
                save_json(rows, partial_detail_path)

        rows = sorted(rows, key=lambda item: int(item["index"]) if str(item["index"]).isdigit() else 0)
        run_score = float(np.mean([row["llm_judge_score"] for row in rows])) if rows else 0.0
        source_counts = Counter(row.get("judge_source", "") for row in rows)
        summary_rows.append(
            {
                "run": prefix,
                "num_samples": len(rows),
                "llm_judge_similarity": run_score,
                "llm_judge_similarity_percent": 100 * run_score,
                "judge_model": judge_model_name,
                "llm_calls": source_counts["LLM_CALL"],
                "auto_exact_or_empty": source_counts["AUTO_EXACT_MATCH"] + source_counts["AUTO_EMPTY_PREDICTION"],
                "auto_question_restatement": source_counts["AUTO_QUESTION_RESTATEMENT"],
                "auto_incomplete_fragment": source_counts["AUTO_INCOMPLETE_FRAGMENT"],
                "auto_no_content": source_counts["AUTO_NO_CONTENT"],
                "strict_prefilter": use_strict_prefilter,
            }
        )
        all_rows.extend(rows)
        save_json(rows, detail_path)
        save_json(rows, partial_detail_path)

    summary = {
        "judge_model": judge_model_name,
        "runs": summary_rows,
    }
    save_json(summary, dirs["results"] / f"{output_name}_summary.json")

    summary_df = pd.DataFrame(summary_rows).sort_values("llm_judge_similarity", ascending=False)
    detail_df = pd.DataFrame(all_rows)
    summary_df.to_csv(tables_dir / f"{output_name}_summary.csv", index=False)
    summary_df.to_markdown(tables_dir / f"{output_name}_summary.md", index=False, floatfmt=".4f")
    detail_df.to_csv(tables_dir / f"{output_name}_details.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 4.5))
        plot_df = summary_df.copy()
        plot_df["run"] = plot_df["run"].str.replace("prompt_", "", regex=False).str.replace("_valfull", "", regex=False)
        sns.barplot(data=plot_df, x="run", y="llm_judge_similarity_percent", color="#4c78a8")
        plt.ylabel("LLM Judge Similarity (%)")
        plt.xlabel("")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(dirs["figures"] / f"{output_name}_summary.png", dpi=220)
        plt.close()
    except Exception:
        pass

    metrics_table_path = tables_dir / "full_prompt_report_assets_metrics_table.csv"
    if metrics_table_path.exists():
        metrics_df = pd.read_csv(metrics_table_path)
        metrics_df = metrics_df.merge(
            summary_df[["run", "llm_judge_similarity_percent"]],
            on="run",
            how="left",
        )
        metrics_df = metrics_df.rename(columns={"llm_judge_similarity_percent": "LLM Judge Similarity"})
        metric_base = f"full_prompt_report_assets_metrics_table_with_{output_name}"
        metrics_df.to_csv(tables_dir / f"{metric_base}.csv", index=False)
        metrics_df.to_markdown(
            tables_dir / f"{metric_base}.md",
            index=False,
            floatfmt=".2f",
        )
        metrics_df.to_latex(
            tables_dir / f"{metric_base}.tex",
            index=False,
            float_format="%.2f",
        )
        metrics_df.to_csv(tables_dir / "full_prompt_report_assets_metrics_table_with_judge.csv", index=False)
        metrics_df.to_markdown(
            tables_dir / "full_prompt_report_assets_metrics_table_with_judge.md",
            index=False,
            floatfmt=".2f",
        )
        metrics_df.to_latex(
            tables_dir / "full_prompt_report_assets_metrics_table_with_judge.tex",
            index=False,
            float_format="%.2f",
        )

    return summary


def preflight() -> None:
    import torch

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_gb:.1f} GB")
    else:
        raise RuntimeError("No CUDA GPU detected. In Colab, set Runtime > Change runtime type > GPU.")
