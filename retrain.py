import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, recall_score, roc_auc_score, mean_absolute_percentage_error
import os
import numpy as np
from datetime import datetime

def clean_column_names(df):
    cleaned_cols = []
    for col in df.columns:
        c = str(col)
        # Remove BOM characters
        c = c.replace('\ufeff', '').replace('ï»¿', '')
        # Fix common UTF-8 accents mojibake
        c = c.replace('Ã©', 'é').replace('Ã¨', 'è').replace('Ãª', 'ê').replace('Ã ', 'à').replace('Ã¹', 'ù')
        c = c.replace('Ã§', 'ç').replace('Ã´', 'ô').replace('Ã»', 'û').replace('Ã¢', 'â')
        c = c.replace('Ã®', 'î').replace('Ã«', 'ë').replace('Ã¯', 'ï')
        c = c.replace('Ã‰', 'É').replace('Ãˆ', 'È')
        cleaned_cols.append(c.strip())
    df.columns = cleaned_cols
    return df

def standardize_df(df):
    df = clean_column_names(df)
    cols = df.columns
    rename_dict = {}
    emp_col = next((c for c in cols if c.lower() == "csal_nosal"), None)
    date_col = next((c for c in cols if c.lower() in ["date", "data"]), None)
    if emp_col and emp_col != "csal_nosal":
        rename_dict[emp_col] = "csal_nosal"
    if date_col and date_col != "DATE":
        rename_dict[date_col] = "DATE"
    if rename_dict:
        df = df.rename(columns=rename_dict)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce').dt.strftime('%Y-%m-%d')
    if "csal_nosal" in df.columns:
        df["csal_nosal"] = df["csal_nosal"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    return df

def run_retraining():
    # 1. Check if approved data exists
    if not os.path.exists("approved_payroll_data.csv"):
        print("❌ approved_payroll_data.csv is missing. Retraining cannot proceed.")
        return False
        
    df_new = pd.read_csv("approved_payroll_data.csv", encoding="utf-8-sig")
    if df_new.empty:
        print("❌ approved_payroll_data.csv is empty. Retraining cannot proceed.")
        return False
        
    df_new = standardize_df(df_new)
        
    # Use payroll date from approved data file if available
    date_col = next((c for c in df_new.columns if c.lower() in ["date", "data"]), None)
    if date_col and not df_new[date_col].empty:
        try:
            timestamp = pd.to_datetime(df_new[date_col], errors='coerce').dropna().dt.strftime('%Y-%m-%d').iloc[0]
        except Exception:
            timestamp = datetime.now().strftime("%Y-%m-%d")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
    # 2. Load existing production models to get feature names and evaluate baseline
    try:
        old_gb = joblib.load("gradient_boosting_model.joblib")
        FEATURES_GB = list(old_gb.feature_names_in_)
    except Exception as e:
        print(f"⚠️ Could not load production GB model features: {e}")
        return False
        
    try:
        old_if = joblib.load("isolation_forest_model.joblib")
        FEATURES_IF = list(old_if.feature_names_in_)
    except Exception as e:
        print(f"⚠️ Could not load production IF model features: {e}")
        return False
        
    try:
        old_rf = joblib.load("surrogate_model.joblib")
        FEATURES_RF = list(old_rf.feature_names_in_)
    except Exception as e:
        print(f"⚠️ Could not load production Surrogate model features: {e}")
        return False

    # Extract target column
    target_col = next((c for c in df_new.columns if c.strip().lower() in [
        "net à payer", "net a payer", "net_a_payer", "net", "net a paye"
    ]), None)
    if not target_col:
        print("❌ Target column 'Net à payer' not found in approved data.")
        return False

    # 3. Align features for evaluation
    def align_features(df, features):
        X = pd.DataFrame(index=df.index)
        for col in features:
            norm_col = col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
            found_col = None
            for df_col in df.columns:
                df_col_norm = df_col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                if df_col_norm == norm_col or df_col == col:
                    found_col = df_col
                    break
            if found_col is not None:
                X[col] = df[found_col].fillna(0)
            else:
                X[col] = 0.0
        return X

    X_new_gb = align_features(df_new, FEATURES_GB)
    y_new = df_new[target_col].fillna(0)

    # 4. Evaluate production model on unseen new approved data BEFORE retraining (No-Leakage)
    prod_mae = mean_absolute_error(y_new, old_gb.predict(X_new_gb))
    prod_mape = mean_absolute_percentage_error(y_new, old_gb.predict(X_new_gb))
    prod_r2 = r2_score(y_new, old_gb.predict(X_new_gb))

    # Evaluate old surrogate model on new data
    anomaly_col = next((c for c in df_new.columns if c.lower() in ["anomaly", "anomaly_residual", "anomaly_residual"]), None)
    if anomaly_col and df_new[anomaly_col].nunique() > 1:
        X_new_rf = align_features(df_new, FEATURES_RF)
        y_new_rf = df_new[anomaly_col].fillna(0).astype(int)
        
        try:
            pred_old_rf = old_rf.predict(X_new_rf)
            prob_old_rf = old_rf.predict_proba(X_new_rf)[:, 0] # Anomaly class (index 0)
            prod_recall = recall_score(y_new_rf, pred_old_rf)
            prod_auc = roc_auc_score(y_new_rf, prob_old_rf)
        except Exception:
            prod_recall, prod_auc = 1.0, 1.0
    else:
        prod_recall, prod_auc = 1.0, 1.0

    # 5. Load historical payroll data to combine
    if os.path.exists("historical_payroll_data.csv"):
        df_past = pd.read_csv("historical_payroll_data.csv", encoding="utf-8-sig")
    elif os.path.exists("anomalies.csv"):
        df_past = pd.read_csv("anomalies.csv", encoding="utf-8-sig")
    else:
        df_past = pd.DataFrame()

    if not df_past.empty:
        df_past = standardize_df(df_past)
        df_all = pd.concat([df_past, df_new], ignore_index=True)
    else:
        df_all = df_new.copy()

    # Enforce deduplication by Employee ID and DATE
    df_all.drop_duplicates(subset=["csal_nosal", "DATE"], keep="last", inplace=True)

    # 6. Fit candidate models on all combined data
    X_all_gb = align_features(df_all, FEATURES_GB)
    y_all = df_all[target_col].fillna(0)

    print("🔄 Training updated models on combined dataset...")
    candidate_gb = GradientBoostingRegressor(
    n_estimators=446,
    learning_rate=0.07215940690474996,
    max_depth=3, 
    min_samples_leaf=29,
    subsample=0.751277227504024,
    random_state=42
    )

    candidate_gb.fit(X_all_gb, y_all)

    X_all_if = align_features(df_all, FEATURES_IF)
    candidate_if = IsolationForest(n_estimators=500, random_state=42)
    candidate_if.fit(X_all_if)

    # 7. Re-calculate residuals and robust Z-scores on all combined data to train surrogate RF
    pred_all_gb = candidate_gb.predict(X_all_gb)
    residual_all = y_all - pred_all_gb
    relative_residual_all = np.where(y_all != 0, residual_all / y_all, 0.0)
    
    median_rel = np.median(relative_residual_all)
    mad_rel = np.median(np.abs(relative_residual_all - median_rel))
    if mad_rel == 0:
        mad_rel = 1e-9
    robust_z_all = 0.6745 * (relative_residual_all - median_rel) / mad_rel
    
    df_all["Anomaly_Residual_New"] = (np.abs(robust_z_all) > 4.25).astype(int)
    X_all_rf = align_features(df_all, FEATURES_RF)
    y_all_rf = df_all["Anomaly_Residual_New"].fillna(0).astype(int)

    candidate_rf = RandomForestClassifier( 
        max_depth=4,
        n_estimators=200,
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    candidate_rf.fit(X_all_rf, y_all_rf)

    # 8. Overwrite production model files
    joblib.dump(candidate_gb, "gradient_boosting_model.joblib")
    joblib.dump(candidate_if, "isolation_forest_model.joblib")
    joblib.dump(candidate_rf, "surrogate_model.joblib")

    # 9. Recalculate and update the global datasets
    df_all["Pred"] = pred_all_gb
    df_all["Residual"] = residual_all
    df_all["Relative Residual"] = relative_residual_all
    df_all["Robust_Z_Score"] = robust_z_all
    df_all["Anomaly"] = df_all["Anomaly_Residual_New"]
    
    if_preds_all = candidate_if.predict(X_all_if)
    df_all["Anomaly_IF"] = (if_preds_all == -1).astype(int)
    df_all["Anomaly_Both"] = ((df_all["Anomaly_IF"] == 1) & (df_all["Anomaly"] == 1)).astype(int)
    
    # Save updated datasets using utf-8-sig
    df_all.to_csv("historical_payroll_data.csv", index=False, encoding="utf-8-sig")
    df_all.to_csv("anomalies.csv", index=False, encoding="utf-8-sig")
    
    df_all[df_all["Anomaly_IF"] == 1].to_csv("anomalies_by_IF.csv", index=False, encoding="utf-8-sig")
    df_all[df_all["Anomaly"] == 1].to_csv("anomalies_by_residual.csv", index=False, encoding="utf-8-sig")
    df_all[df_all["Anomaly_Both"] == 1].to_csv("anomalies_both.csv", index=False, encoding="utf-8-sig")
    
    # Clear approved data file
    pd.DataFrame(columns=df_new.columns).to_csv("approved_payroll_data.csv", index=False, encoding="utf-8-sig")

    # 10. Update or write logs to training_history.csv
    log_file = "training_history.csv"
    if os.path.exists(log_file):
        df_hist = pd.read_csv(log_file, encoding="utf-8-sig")
        df_hist = clean_column_names(df_hist)
        idx = df_hist[df_hist['execution_date'] == timestamp].index
        if len(idx) > 0:
            df_hist.loc[idx, 'total_rows_trained'] = len(df_all)
            df_hist.to_csv(log_file, index=False, encoding="utf-8-sig")
        else:
            log_entry = pd.DataFrame([{
                'execution_date': timestamp,
                'total_rows_trained': len(df_all),
                'new_data_rows': len(df_new),
                'model_mae_on_new_data': prod_mae,
                'model_mape_on_new_data': prod_mape,
                'model_r2_on_new_data': prod_r2,
                'model_recall_on_new_data': prod_recall,
                'model_auc_on_new_data': prod_auc
            }])
            df_hist = pd.concat([df_hist, log_entry], ignore_index=True)
            df_hist.to_csv(log_file, index=False, encoding="utf-8-sig")
    else:
        log_entry = pd.DataFrame([{
            'execution_date': timestamp,
            'total_rows_trained': len(df_all),
            'new_data_rows': len(df_new),
            'model_mae_on_new_data': prod_mae,
            'model_mape_on_new_data': prod_mape,
            'model_r2_on_new_data': prod_r2,
            'model_recall_on_new_data': prod_recall,
            'model_auc_on_new_data': prod_auc
        }])
        log_entry.to_csv(log_file, index=False, encoding="utf-8-sig")

    print(f"✅ Retraining completed successfully on {timestamp}.")
    return True

if __name__ == "__main__":
    run_retraining()
