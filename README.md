# SHBT261-Final-Project
SHBT Final Proejct Code - Helena (Yun Zhu) Li and Yike Cheng

# SHBT261 TextVQA Final Pipeline

This GitHub contains the final prompt-engineering pipeline for the SHBT261 TextVQA project. It runs Qwen2.5-VL-3B-Instruct on the full 5,000-example TextVQA validation split and compares four prompt settings:

- `plain`: raw image + question baseline
- `concise`: answer-format-controlled baseline
- `ocr`: OCR-token-aware prompt
- `strict_ocr`: OCR-token-aware prompt with stronger answer-selection instructions

The assignment allows **fine-tuning or prompt engineering**. This project uses prompt engineering as the required method and produces reproducible metrics, ablations, visualizations, and qualitative error analysis.

## Files

- `colab_prompt_engineering_final.ipynb`: Colab notebook to run the full experiment and generate report assets
- `textvqa_pipeline.py`: reusable pipeline code for loading data/model, inference, metrics, analysis, tables, and plots
- `README.md`: this guide

## Upload Location

Upload the entire `textvqa_final` folder directly to Google Drive:

```text
MyDrive/textvqa_final/
```

The notebook expects this exact location:

```python
PROJECT_DIR = '/content/drive/MyDrive/textvqa_final'
```

All outputs are saved inside the same folder, not at the root of Drive.

## Colab Settings

Use:

```text
Runtime type: Python 3
Hardware accelerator: GPU
GPU type: A100 preferred, L4 acceptable
Runtime shape: High-RAM preferred
```

## Run Order

Open `colab_prompt_engineering_final.ipynb` and run every cell from top to bottom. There are no optional experiment cells.

The notebook will:

1. Install dependencies
2. Mount Drive
3. Import the local pipeline and verify CUDA
4. Load Qwen2.5-VL-3B-Instruct and the full TextVQA validation split
5. Run all four prompt styles on all 5,000 validation examples
6. Generate all report-ready outputs

## Expected Runtime

On A100 High-RAM, expect roughly:

- Setup + model/dataset loading: 15-30 minutes
- Four full-validation prompt runs: about 6-12 hours, depending on Colab speed and batching
- Report asset generation: 5-10 minutes

If Colab is faster than expected, it may finish sooner. Partial prediction files are saved every 250 samples.

## Output Structure

All outputs are created under:

```text
MyDrive/textvqa_final/results/
MyDrive/textvqa_final/figures/
```

Important report assets:

```text
results/full_prompt_ablation_summary.csv
results/full_prompt_ablation_summary.json

results/report_tables/full_prompt_report_assets_metrics_table.csv
results/report_tables/full_prompt_report_assets_metrics_table.md
results/report_tables/full_prompt_report_assets_metrics_table.tex
results/report_tables/full_prompt_report_assets_improvements_vs_baseline.csv
results/report_tables/full_prompt_report_assets_improvements_vs_baseline.md
results/report_tables/full_prompt_report_assets_category_accuracy.csv
results/report_tables/full_prompt_report_assets_category_accuracy.md
results/report_tables/full_prompt_report_assets_category_improvement.csv
results/report_tables/full_prompt_report_assets_error_distribution.csv
results/report_tables/full_prompt_report_assets_accuracy_by_ocr_count.csv
results/report_tables/full_prompt_report_assets_accuracy_by_answer_length.csv
results/report_tables/full_prompt_report_assets_baseline_to_best_transition.csv

results/report_examples/full_prompt_report_assets_qualitative_examples.md
results/report_examples/full_prompt_report_assets_improved_and_regressed_examples.csv

figures/full_prompt_report_assets_metric_comparison.png
figures/full_prompt_report_assets_category_heatmap.png
figures/full_prompt_report_assets_category_bars.png
figures/full_prompt_report_assets_category_improvement.png
figures/full_prompt_report_assets_error_distribution.png
figures/full_prompt_report_assets_accuracy_by_ocr_count.png
figures/full_prompt_report_assets_accuracy_by_answer_length.png
figures/full_prompt_report_assets_baseline_to_best_transition.png
```

These files are intended to support the final report sections: methodology, experimental design, quantitative results, category analysis, error analysis, and qualitative examples.
