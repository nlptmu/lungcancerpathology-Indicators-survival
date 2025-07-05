# Extracting Critical Clinical Indicators and Survival Prediction of Lung Cancer from Pathology Reports using Large Language Models

## Data Process
```
python process_data.py \
  --data_path <data_path> \
  --output_dir <output_dir> \
  --split_train_test
```

## Run NER
```
CUDA_VISIBLE_DEVICES=0 python run_ner.py \
  --train_file <train_file> \
  --test_file <test_file> \
  --ntu_file <ntu_file> \
  --output_dir <output_dir> \
  --model_name_or_path UFNLP/gatortron-base
```

## Compute Metrics
```
python compute_metrics.py \
  --predictions_file <prediction_file> \
  --outputs_dir <output_dir> \
  --model_type bert
```
