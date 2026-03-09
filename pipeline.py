# --- Model-Dependent Steps (Loop Through Each Model) ---
for model_basename in MODELS_TO_RUN:
    model_name = os.path.splitext(model_basename)[0]
    print(f"\n-- Running predictions for model: {model_name} --")
    model_path = resource_path(os.path.join('models', model_basename))

    prediction_npy_file = os.path.join(batch_prediction_output_dir, f"predictions__{model_name}.npy")
    batch_final_csv     = os.path.join(batch_prediction_output_dir, f"predictions__{model_name}.csv")

    print(f"STEP 4: Running predictions...")
    run_predictions(input_file=master_h5_file, model_file=model_path, output_file=prediction_npy_file)

    print(f"STEP 5: Converting NPY predictions to CSV...")
    labels = MODEL_LABELS.get(model_basename, [])
    convert_npy_to_csv(
        input_file=prediction_npy_file,
        output_file=batch_final_csv,
        demo_file=demographics_csv,
        records_file=records_file,
        labels=labels
    )

    # NEW: append this batch to the pooled per-model CSV right away
    if os.path.exists(batch_final_csv):
        pooled_csv = os.path.join(output_csv_dir, f"{model_name}_predictions.csv")
        df = pd.read_csv(batch_final_csv)
        append_df_safely(df, pooled_csv)
        print(f"[Batch {batch_num}] Appended {len(df)} rows to: {pooled_csv}")
        # optional: reclaim space
        try:
            os.remove(batch_final_csv)
        except OSError:
            pass
    else:
        print(f"WARNING: No CSV output was generated for {model_name} in batch {batch_num}.")
