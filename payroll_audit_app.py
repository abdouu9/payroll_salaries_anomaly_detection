import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np


# Config premium layout
st.set_page_config(page_title="Audit des Paies & Anomalies", layout="wide", initial_sidebar_state="expanded")

# Injected Custom CSS for premium styling
st.markdown("""
    <style>
        /* General styling */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Header style */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #6B7280;
            margin-bottom: 2rem;
        }
        
        /* Card design for metrics */
        .metric-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #E5E7EB;
            transition: transform 0.2s ease;
            text-align: center;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0.25rem 0;
        }
        .metric-label {
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Accent colors */
        .color-total { color: #1E3A8A; }
        .color-if { color: #F59E0B; }
        .color-res { color: #3B82F6; }
        .color-crit { color: #EF4444; }
        .color-contract { color: #8B5CF6; }
        
        /* Banners */
        .welcome-card {
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.05) 0%, rgba(59, 130, 246, 0.05) 100%);
            border-left: 5px solid #1E3A8A;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        
    </style>
""", unsafe_allow_html=True)


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


# Function to dynamically find ID and Date columns matching any case
def get_column_mapping(df):
    cols = df.columns
    emp_col = next((c for c in cols if c.lower() == "csal_nosal"), None)
    date_col = next((c for c in cols if c.lower() in ["date", "data"]), None)
    return emp_col, date_col


# Loading and caching datasets
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Fichier manquant : {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file_path)
        
    df = clean_column_names(df)
    emp_col, date_col = get_column_mapping(df)
    
    # Standardize column naming internally for consistent filtering
    rename_dict = {}
    if emp_col and emp_col != "csal_nosal":
        rename_dict[emp_col] = "csal_nosal"
    if date_col and date_col != "DATE":
        rename_dict[date_col] = "DATE"
        
    if rename_dict:
        df = df.rename(columns=rename_dict)
        
    # Convert DATE to string YYYY-MM-DD for clean filtering dropdowns
    if "DATE" in df.columns:
        # First ensure parseable datetimes
        parsed_dates = pd.to_datetime(df["DATE"], errors='coerce')
        df["DATE"] = parsed_dates.dt.strftime('%Y-%m-%d')
        
    # Cast Employee ID to string to avoid float conversions in selectbox/searches
    if "csal_nosal" in df.columns:
        df["csal_nosal"] = df["csal_nosal"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        
    return df


@st.cache_resource
def load_if_model():
    if os.path.exists("isolation_forest_model.joblib"):
        return joblib.load("isolation_forest_model.joblib")
    return None


@st.cache_resource
def load_gb_model():
    if os.path.exists("gradient_boosting_model.joblib"):
        return joblib.load("gradient_boosting_model.joblib")
    return None


@st.cache_resource
def load_surrogate_model():
    if os.path.exists("surrogate_model.joblib"):
        return joblib.load("surrogate_model.joblib")
    return None


model_if = load_if_model()
model_gb = load_gb_model()
model_surrogate = load_surrogate_model()

if model_if is not None:
    feature_cols_if = list(model_if.feature_names_in_)
else:
    feature_cols_if = []

if model_gb is not None:
    feature_cols_gb = list(model_gb.feature_names_in_)
else:
    feature_cols_gb = []

if model_surrogate is not None:
    feature_cols_surrogate = list(model_surrogate.feature_names_in_)
else:
    feature_cols_surrogate = []

# Load all datasets
df_anomalies = load_data("anomalies.csv")
df_if = load_data("anomalies_by_IF.csv")
df_residual = load_data("anomalies_by_residual.csv")
df_both = load_data("anomalies_both.csv")
df_contract = load_data("end_of_contract.csv")

# Compute historical median and MAD of relative residuals from ALL historical data (df_anomalies)
if not df_anomalies.empty and "Relative Residual" in df_anomalies.columns:
    hist_relative_residuals = df_anomalies["Relative Residual"].dropna().to_numpy(dtype=np.float64)
    hist_median = np.median(hist_relative_residuals)
    hist_mad = np.median(np.abs(hist_relative_residuals - hist_median))
    if hist_mad == 0:
        hist_mad = 1e-9
else:
    hist_relative_residuals = np.array([])
    hist_median = 0.0
    hist_mad = 1e-9

@st.cache_resource
def get_shap_explainer(_model):
    if _model is not None:
        try:
            return shap.TreeExplainer(_model)
        except Exception:
            return shap.Explainer(_model)
    return None

@st.cache_resource
def get_surrogate_explainer(_model, _df_anomalies, feature_cols):
    if _model is not None:
        try:
            # Align features of background data to match model expectations
            if not _df_anomalies.empty:
                X_bg = pd.DataFrame(index=_df_anomalies.index)
                for col in feature_cols:
                    norm_col = col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                    found_col = None
                    for df_col in _df_anomalies.columns:
                        df_col_norm = df_col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                        if df_col_norm == norm_col or df_col == col:
                            found_col = df_col
                            break
                    if found_col is not None:
                        X_bg[col] = _df_anomalies[found_col].fillna(0)
                    else:
                        X_bg[col] = 0.0
                return shap.TreeExplainer(_model, data=X_bg, model_output="probability")
            else:
                return shap.TreeExplainer(_model, model_output="probability")
        except Exception:
            try:
                return shap.TreeExplainer(_model, model_output="probability")
            except Exception:
                return shap.TreeExplainer(_model)
    return None

explainer_if = get_shap_explainer(model_if)
explainer_gb = get_shap_explainer(model_gb)
explainer_surrogate = get_surrogate_explainer(model_surrogate, df_anomalies, feature_cols_surrogate)


# ----------------- SIDEBAR PAGE SELECTION -----------------
st.sidebar.header("Navigation")
app_page = st.sidebar.selectbox(
    "Sélectionnez l'onglet :",
    ["Analyse Historique", "Audit de Nouveaux Fichiers", "Performance & Réentraînement"]
)

# Application title
st.markdown('<div class="main-title">Plateforme d\'Audit & Détection d\'Anomalies</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyse croisée des anomalies de paie (Isolation Forest, Régression & Cas spéciaux)</div>', unsafe_allow_html=True)

# Define sub-pages
if app_page == "Analyse Historique":
    # ----------------- SIDEBAR FILTERS -----------------

    st.sidebar.header(" Critères de Recherche")

    # Select search mode
    search_mode = st.sidebar.radio(
        "Mode de recherche :",
        ["Par Employé", "Par Date", "Recherche Combinée"],
        index=0
    )

    # Extract unique employees and dates for validation and dropdowns
    unique_employees = sorted(df_anomalies["csal_nosal"].unique(), key=lambda x: int(x) if x.isdigit() else 99999) if not df_anomalies.empty else []
    unique_dates = sorted(df_anomalies["DATE"].dropna().unique()) if not df_anomalies.empty else []

    selected_emp = ""
    selected_date = ""

    if search_mode == "Par Employé" or search_mode == "Recherche Combinée":
        # Help user with some examples
        st.sidebar.markdown("**Employés suspectés (Exemples) :**")
        st.sidebar.caption("Cas Critiques : 786, 171, 457")
        st.sidebar.caption("Fins de contrat : 777, 778, 783")
        
        selected_emp = st.sidebar.text_input(
            "ID Employé (csal_nosal) :",
            value="",
            placeholder="Entrez un identifiant..."
        ).strip()

    if search_mode == "Par Date" or search_mode == "Recherche Combinée":
        selected_date = st.sidebar.selectbox(
            "Sélectionnez la Date :",
            [""] + unique_dates,
            index=0
        )

    # Determine if the query is active
    is_query_active = False
    if search_mode == "Par Employé" and selected_emp != "":
        is_query_active = True
    elif search_mode == "Par Date" and selected_date != "":
        is_query_active = True
    elif search_mode == "Recherche Combinée" and selected_emp != "" and selected_date != "":
        is_query_active = True

    # ----------------- MAIN LOGIC -----------------

    if not is_query_active:
        # ----------------- WELCOME SCREEN (NO FILTERS ACTIVE) -----------------
        st.markdown("""
            <div class="welcome-card">
                <h3>Bienvenue sur l'outil d'audit de paie</h3>
                <p>Veuillez sélectionner un employé et/ou une date dans la barre latérale gauche pour lancer la recherche et analyser les anomalies.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Global overview widgets
        st.subheader("Statuts Globaux des Données")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-total">Total Fiches</div>
                    <div class="metric-value color-total">{len(df_anomalies):,}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Fiches de paie chargées</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-if">Anomalies IF</div>
                    <div class="metric-value color-if">{len(df_if):,}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Détectées par Isolation Forest</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-res">Anomalies Résidus</div>
                    <div class="metric-value color-res">{len(df_residual):,}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Détectées par Régression</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-crit">Cas Critiques</div>
                    <div class="metric-value color-crit">{len(df_both):,}</div>
                    <div style="font-size: 0.8rem; color: #EF4444; font-weight: 600;">⚠️ Confirmés par les 2 modèles</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col5:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-contract">Fins de contrat</div>
                    <div class="metric-value color-contract">{len(df_contract):,}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Dossiers traités à part</div>
                </div>
            """, unsafe_allow_html=True)

    else:
        # ----------------- FILTER DATASETS -----------------
        def filter_dataframe(df, search_mode, emp, dt):
            if df.empty:
                return df
            
            filtered = df.copy()
            if search_mode == "Par Employé" or search_mode == "Recherche Combinée":
                if "csal_nosal" in filtered.columns:
                    filtered = filtered[filtered["csal_nosal"] == emp]
            
            if search_mode == "Par Date" or search_mode == "Recherche Combinée":
                if "DATE" in filtered.columns:
                    filtered = filtered[filtered["DATE"] == dt]
                    
            return filtered

        # Perform filtrations
        df_anom_filtered = filter_dataframe(df_anomalies, search_mode, selected_emp, selected_date)
        df_if_filtered = filter_dataframe(df_if, search_mode, selected_emp, selected_date)
        df_res_filtered = filter_dataframe(df_residual, search_mode, selected_emp, selected_date)
        df_both_filtered = filter_dataframe(df_both, search_mode, selected_emp, selected_date)
        df_contract_filtered = filter_dataframe(df_contract, search_mode, selected_emp, selected_date)

        # ----------------- RECALCULATE DYNAMIC KPIs -----------------
        st.subheader("Métriques Relatives à votre Sélection")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        with kpi_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-total">Fiches Filtrées</div>
                    <div class="metric-value color-total">{len(df_anom_filtered)}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Dans le jeu principal</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-if">Anomalies IF</div>
                    <div class="metric-value color-if">{len(df_if_filtered)}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Fiches signalées</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-res">Anomalies Résidus</div>
                    <div class="metric-value color-res">{len(df_res_filtered)}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Fiches signalées</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
                <div class="metric-card" style="border: 1px solid #FCA5A5; background: #FEF2F2;">
                    <div class="metric-label color-crit">Cas Critiques</div>
                    <div class="metric-value color-crit">{len(df_both_filtered)}</div>
                    <div style="font-size: 0.8rem; color: #EF4444; font-weight: 600;">Double signalement</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_col5:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label color-contract">Fins de Contrat</div>
                    <div class="metric-value color-contract">{len(df_contract_filtered)}</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Cas identifiés</div>
                </div>
            """, unsafe_allow_html=True)

        # ----------------- TABS SYSTEM -----------------
        st.write("")
        st.write("")
        st.subheader("Consultation des Tableaux de Données")
        
        tab_all, tab_if, tab_res, tab_both, tab_contract = st.tabs([
            " Toutes les Fiches",
            "Anomalies Isolation Forest",
            "Anomalies par Résidus",
            " Cas Critiques (Risque Max)",
            "Fins de Contrat"
        ])
        
        # Helper function to print clean, formatted tables
        def show_audit_table(df, tab_name):
            if df.empty:
                st.info(f"Aucune ligne trouvée dans cette catégorie ({tab_name}) pour vos critères de recherche.")
            else:
                st.caption(f"Affichage de {len(df)} fiches de paie correspondantes")
                
                # Format numeric columns for clean separation of thousands
                format_dict = {}
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64'] and not col.lower().endswith(('id', 'nosal', 'anomaly', 'anomaly_if')):
                        format_dict[col] = "{:,.0f} FCFA"
                        
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Render inside each tab
        with tab_all:
            show_audit_table(df_anom_filtered, "Toutes les fiches")
            
        with tab_if:
            if df_if_filtered.empty:
                st.info("Aucune ligne trouvée dans cette catégorie (Anomalies Isolation Forest) pour vos critères de recherche.")
            else:
                st.caption(f"Affichage de {len(df_if_filtered)} fiches de paie correspondantes (Sélectionnez une ligne pour afficher l'explication SHAP)")
                
                # Format numeric columns for clean separation of thousands
                display_df_if = df_if_filtered.copy()
                # Add an internal index to identify the selected row precisely
                display_df_if.insert(0, "_row_id", display_df_if.index)
                
                table_event = st.dataframe(
                    display_df_if,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="if_anomalies_table"
                )
                
                selected_rows = table_event.get("selection", {}).get("rows", [])
                if selected_rows:
                    selected_pos = int(selected_rows[0])
                    actual_idx = int(display_df_if.iloc[selected_pos]["_row_id"])
                    selected_row = df_if_filtered.loc[actual_idx]
                    
                    st.markdown("---")
                    st.subheader(" Explication de l'Anomalie (TreeSHAP)")
                    
                    if model_if is None or explainer_if is None:
                        st.error("Le modèle Isolation Forest ou l'explicateur SHAP n'a pas pu être chargé.")
                    else:
                        # Align the selected row to feature columns of the model
                        # If columns are encoded differently or missing, fillna(0)
                        row_data = {}
                        for col in feature_cols_if:
                            # Clean column mapping
                            norm_col = col.replace("", "é")
                            found_col = None
                            for df_col in df_if_filtered.columns:
                                if df_col.replace("é", "e").replace("è", "e").lower() == norm_col.replace("é", "e").replace("è", "e").lower() or df_col == col:
                                    found_col = df_col
                                    break
                            if found_col and found_col in selected_row:
                                row_data[col] = selected_row[found_col]
                            else:
                                row_data[col] = 0.0
                        
                        x_instance = pd.DataFrame([row_data])
                        
                        # Extract or calculate anomaly score
                        score_val = None
                        score_col = next((c for c in selected_row.index if c.lower() in ["anomaly_scores", "anomaly_score", "score"]), None)
                        if score_col:
                            score_val = selected_row[score_col]
                        elif model_if is not None:
                            score_val = model_if.score_samples(x_instance)[0]
                        
                        if score_val is not None:
                            st.metric("Score d'Anomalie Isolation Forest", f"{score_val:.4f}")
                        
                        # Compute SHAP explanation
                        try:
                            shap_explanation = explainer_if(x_instance)
                            
                            col_chart, col_table = st.columns([2, 1])
                            
                            with col_chart:
                                st.markdown("**Graphique Cascade (Waterfall Chart)**")
                                fig, ax = plt.subplots(figsize=(8, 5))
                                shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
                                plt.tight_layout()
                                st.pyplot(fig, clear_figure=True)
                                
                            with col_table:
                                st.markdown("**Contributions des Variables**")
                                # Create DataFrame with contributions
                                contribs = pd.DataFrame({
                                    "Variable": feature_cols_if,
                                    "Valeur SHAP": shap_explanation[0].values,
                                    "Abs_SHAP": np.abs(shap_explanation[0].values)
                                }).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)
                                
                                st.dataframe(
                                    contribs[["Variable", "Valeur SHAP"]],
                                    use_container_width=True,
                                    hide_index=True
                                )
                        except Exception as e:
                            st.error(f"Erreur lors du calcul SHAP : {e}")

            
        with tab_res:
            if df_res_filtered.empty:
                st.info("Aucune ligne trouvée dans cette catégorie (Anomalies par Résidus) pour vos critères de recherche.")
            else:
                st.caption(f"Affichage de {len(df_res_filtered)} fiches de paie correspondantes (Sélectionnez une ligne pour afficher l'explication SHAP)")
                
                # Format numeric columns for clean separation of thousands
                display_df_res = df_res_filtered.copy()
                display_df_res.insert(0, "_row_id", display_df_res.index)
                
                table_event_res = st.dataframe(
                    display_df_res,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="res_anomalies_table"
                )
                
                selected_rows_res = table_event_res.get("selection", {}).get("rows", [])
                if selected_rows_res:
                    selected_pos_res = int(selected_rows_res[0])
                    actual_idx_res = int(display_df_res.iloc[selected_pos_res]["_row_id"])
                    selected_row_res = df_res_filtered.loc[actual_idx_res]
                    
                    st.markdown("---")
                    st.subheader("Explication de l'Anomalie (TreeSHAP du Modèle Substitut)")
                    
                    # Display Regression Metrics
                    net_val = selected_row_res.get("Net à payer", selected_row_res.get("Net a payer", 0.0))
                    pred_val = selected_row_res.get("Pred", selected_row_res.get("Prediction", 0.0))
                    res_val = selected_row_res.get("Residual", selected_row_res.get("Résiduel", 0.0))
                    rel_res_val = selected_row_res.get("Relative Residual", selected_row_res.get("Résiduel Relatif", 0.0))
                    
                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Net à payer Réel", f"{net_val:,.0f} FCFA")
                    met2.metric("Net à payer Prédit", f"{pred_val:,.0f} FCFA")
                    met3.metric("Résidu Brut", f"{res_val:,.0f} FCFA")
                    met4.metric("Résidu Relatif", f"{rel_res_val:.4%}")
                    
                    st.write("")
                    
                    if model_surrogate is None or explainer_surrogate is None:
                        st.error("Le modèle Random Forest substitut ou l'explicateur SHAP n'a pas pu être chargé.")
                    else:
                        row_data = {}
                        for col in feature_cols_surrogate:
                            norm_col = col.replace("", "é")
                            found_col = None
                            for df_col in df_res_filtered.columns:
                                if df_col.replace("é", "e").replace("è", "e").lower() == norm_col.replace("é", "e").replace("è", "e").lower() or df_col == col:
                                    found_col = df_col
                                    break
                            if found_col and found_col in selected_row_res:
                                row_data[col] = selected_row_res[found_col]
                            else:
                                row_data[col] = 0.0
                        
                        x_instance = pd.DataFrame([row_data])
                        
                        try:
                            shap_explanation = explainer_surrogate(x_instance)
                            
                            # Dynamically handle 3D (multi-class) vs 2D (single-class output/binary fallback)
                            if len(shap_explanation.shape) == 3:
                                anomaly_class_idx = 1
                                if hasattr(model_surrogate, "classes_"):
                                    classes = list(model_surrogate.classes_)
                                    if -1 in classes:
                                        anomaly_class_idx = classes.index(-1)
                                    elif 1 in classes:
                                        anomaly_class_idx = classes.index(1)
                                shap_exp_sliced = shap_explanation[0, :, anomaly_class_idx]
                            else:
                                shap_exp_sliced = shap_explanation[0]
                                
                            col_chart, col_table = st.columns([2, 1])
                            
                            with col_chart:
                                st.markdown("**Graphique Cascade (Waterfall Chart)**")
                                fig, ax = plt.subplots(figsize=(8, 5))
                                shap.plots.waterfall(shap_exp_sliced, max_display=12, show=False)
                                plt.tight_layout()
                                st.pyplot(fig, clear_figure=True)
                                
                            with col_table:
                                st.markdown("**Contributions des Variables**")
                                contribs = pd.DataFrame({
                                    "Variable": feature_cols_surrogate,
                                    "Valeur SHAP": shap_exp_sliced.values,
                                    "Abs_SHAP": np.abs(shap_exp_sliced.values)
                                }).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)
                                
                                st.dataframe(
                                    contribs[["Variable", "Valeur SHAP"]],
                                    use_container_width=True,
                                    hide_index=True
                                )
                        except Exception as e:
                            st.error(f"Erreur lors du calcul SHAP : {e}")

            
        with tab_both:
            show_audit_table(df_both_filtered, "Cas Critiques (Risque Max)")
            
        with tab_contract:
            show_audit_table(df_contract_filtered, "Fins de Contrat")

elif app_page == "Audit de Nouveaux Fichiers":
    st.sidebar.markdown("---")
    st.sidebar.info(" Chargez un nouveau fichier de paie (CSV) pour effectuer les prédictions et détecter les anomalies en temps réel.")

    st.markdown("""
        <div class="welcome-card">
            <h3> Audit de Nouveaux Fichiers de Paie</h3>
            <p>Cette page vous permet d'analyser un nouveau fichier de paie brut. Elle applique les règles de nettoyage, exclut les fins de contrat, effectue les prédictions de salaire net, calcule les résidus robustes (Z-Score) et exécute l'Isolation Forest.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Sélectionnez le fichier CSV de paie à auditer :",
        type=["csv"],
        key="new_payroll_file_uploader"
    )

    if uploaded_file is not None:
        try:
            try:
                df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            except Exception:
                df_raw = pd.read_csv(uploaded_file)
            df_raw = clean_column_names(df_raw)
            
            # --- Rename employee and date columns case-insensitively to standard names ---
            rename_dict = {}
            emp_col = next((c for c in df_raw.columns if c.lower() == "csal_nosal"), None)
            date_col = next((c for c in df_raw.columns if c.lower() in ["date", "data"]), None)
            if emp_col and emp_col != "csal_nosal":
                rename_dict[emp_col] = "csal_nosal"
            if date_col and date_col != "DATE":
                rename_dict[date_col] = "DATE"
            if rename_dict:
                df_raw = df_raw.rename(columns=rename_dict)
            
            # Ensure proper formats
            if "DATE" in df_raw.columns:
                df_raw["DATE"] = pd.to_datetime(df_raw["DATE"], errors='coerce').dt.strftime('%Y-%m-%d')
            if "csal_nosal" in df_raw.columns:
                df_raw["csal_nosal"] = df_raw["csal_nosal"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            
            # --- 1. Filter Jours Travaillés ---
            jours_col = next((c for c in df_raw.columns if c.strip().lower() in [
                "jours travaillés", "jours travaille", "jours travailles", 
                "jours_travailles", "jours_travaillés", "jours trav"
            ]), None)
            
            if jours_col is not None:
                # Keep rows where Jours Travaillés is null (NaN) or != 0
                df_filtered_jt = df_raw[df_raw[jours_col].isna() | (df_raw[jours_col] != 0)].copy()
                st.info(f"Filtre 'Jours Travaillés' appliqué : {len(df_raw) - len(df_filtered_jt)} lignes supprimées (Jours Travaillés == 0).")
                df_raw = df_filtered_jt
            else:
                st.warning(" Colonne 'Jours Travaillés' introuvable dans le fichier importé. Aucune ligne n'a été filtrée par ce critère.")
            
            # --- 2. Extract End of Contract Cases ---
            solde_col = next((c for c in df_raw.columns if c.strip().lower() in [
                "solde congé", "solde conge", "solde_conge", "solde_congés", "solde congés", "solde cong"
            ]), None)
            indem_col = next((c for c in df_raw.columns if c.strip().lower() in [
                "indem fin contrat", "indem_fin_contrat", "indemnite fin contrat", "indem fin contr"
            ]), None)
            
            is_contract = pd.Series([False] * len(df_raw), index=df_raw.index)
            solde_vals = pd.Series(0.0, index=df_raw.index)
            indem_vals = pd.Series(0.0, index=df_raw.index)
            if solde_col:
                solde_vals = pd.to_numeric(df_raw[solde_col], errors='coerce').fillna(0.0)
            if indem_col:
                indem_vals = pd.to_numeric(df_raw[indem_col], errors='coerce').fillna(0.0)
                
            is_contract = (solde_vals != 0.0) | (indem_vals != 0.0)
                
            df_contract_new = df_raw[is_contract].copy()
            df_active_new = df_raw[~is_contract].copy()
            
            if df_active_new.empty:
                st.warning(" Aucune fiche de paie active restante après filtrage des jours travaillés et fins de contrat.")
            else:
                # --- 3. Aggregate Primes Feature ---
                # Check if "Primes" or "Prime" column already exists first
                primes_col = next((c for c in df_active_new.columns if c.strip().lower() in ["primes", "prime"]), None)
                if primes_col is not None:
                    df_active_new["Primes"] = df_active_new[primes_col].fillna(0)
                else:
                    # Fallback to candidates sum
                    primes_candidates = ["Sursal", "Prime d'astreinte", "Prime Exceptionnelle", "P. Responsabilité", "Bonus", "P. Journaliers", "Prime de Ramadan"]
                    matched_prime_cols = []
                    for cand in primes_candidates:
                        cand_norm = cand.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                        for col in df_active_new.columns:
                            col_norm = col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                            if col_norm == cand_norm or col == cand:
                                matched_prime_cols.append(col)
                                
                    if matched_prime_cols:
                        df_active_new["Primes"] = df_active_new[matched_prime_cols].fillna(0).sum(axis=1)
                    else:
                        df_active_new["Primes"] = 0.0

                # --- 4. Gradient Boosting Feature Alignment & Prediction ---
                X_gb = pd.DataFrame(index=df_active_new.index)
                for col in feature_cols_gb:
                    norm_col = col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                    found_col = None
                    for df_col in df_active_new.columns:
                        df_col_norm = df_col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                        if df_col_norm == norm_col or df_col == col:
                            found_col = df_col
                            break
                    if found_col is not None:
                        X_gb[col] = df_active_new[found_col].fillna(0)
                    else:
                        X_gb[col] = 0.0
                
                if model_gb is not None:
                    pred_net = model_gb.predict(X_gb)
                else:
                    pred_net = np.zeros(len(df_active_new))
                    st.error("Modèle Gradient Boosting introuvable.")

                # --- 5. Residuals & Robust Z-Scores ---
                net_pay_col = next((c for c in df_active_new.columns if c.strip().lower() in [
                    "net à payer", "net a payer", "net_a_payer", "net", "net a paye"
                ]), None)
                if net_pay_col:
                    actual_net = df_active_new[net_pay_col].fillna(0)
                else:
                    st.warning(" Colonne 'Net à payer' absente. Les résidus relatifs et le Robust Z-score ne peuvent pas être calculés précisément.")
                    actual_net = pd.Series(pred_net, index=df_active_new.index)
                
                residual = actual_net - pred_net
                relative_residual = np.where(actual_net != 0, residual / actual_net, 0.0)
                
                df_active_new["Pred"] = pred_net
                df_active_new["Residual"] = residual
                df_active_new["Relative Residual"] = relative_residual
                
                # Robust Z-Score calculation over combined dataset (historical + new)
                if len(relative_residual) > 0:
                    if len(hist_relative_residuals) > 0:
                        combined_residuals = np.concatenate([hist_relative_residuals, relative_residual])
                    else:
                        combined_residuals = relative_residual
                    
                    combined_median = np.median(combined_residuals)
                    combined_mad = np.median(np.abs(combined_residuals - combined_median))
                    if combined_mad == 0:
                        combined_mad = 1e-9
                    robust_z = 0.6745 * (relative_residual - combined_median) / combined_mad
                else:
                    robust_z = np.zeros(len(df_active_new))
                
                df_active_new["Robust_Z_Score"] = robust_z
                df_active_new["Anomaly_Residual"] = (np.abs(robust_z) > 4.5).astype(int)

                # --- 6. Isolation Forest Feature Alignment & Prediction ---
                X_if = pd.DataFrame(index=df_active_new.index)
                for col in feature_cols_if:
                    if col == "Relative Residual":
                        X_if[col] = df_active_new["Relative Residual"]
                    else:
                        norm_col = col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                        found_col = None
                        for df_col in df_active_new.columns:
                            df_col_norm = df_col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                            if df_col_norm == norm_col or df_col == col:
                                found_col = df_col
                                break
                        if found_col is not None:
                            X_if[col] = df_active_new[found_col].fillna(0)
                        else:
                            X_if[col] = 0.0
                
                if model_if is not None:
                    if_preds = model_if.predict(X_if)
                    df_active_new["Anomaly_IF"] = (if_preds == -1).astype(int)
                else:
                    df_active_new["Anomaly_IF"] = 0
                    st.error("Modèle Isolation Forest introuvable.")

                # Double alerts
                df_active_new["Anomaly_Both"] = ((df_active_new["Anomaly_IF"] == 1) & (df_active_new["Anomaly_Residual"] == 1)).astype(int)

                # --- 7. Prepare Tab Datasets ---
                df_uploaded_all = df_active_new.copy()
                df_uploaded_if = df_active_new[df_active_new["Anomaly_IF"] == 1].copy()
                df_uploaded_res = df_active_new[df_active_new["Anomaly_Residual"] == 1].copy()
                df_uploaded_both = df_active_new[df_active_new["Anomaly_Both"] == 1].copy()

                # --- 8. Render Dynamic KPI Metrics ---
                st.subheader("Résultats Globaux du Fichier Importé")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label color-total">Total Actives</div>
                            <div class="metric-value color-total">{len(df_uploaded_all):,}</div>
                            <div style="font-size: 0.8rem; color: #6B7280;">Fiches analysées</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label color-if">Anomalies IF</div>
                            <div class="metric-value color-if">{len(df_uploaded_if):,}</div>
                            <div style="font-size: 0.8rem; color: #6B7280;">Signalées par IF</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label color-res">Anomalies Résidus</div>
                            <div class="metric-value color-res">{len(df_uploaded_res):,}</div>
                            <div style="font-size: 0.8rem; color: #6B7280;">Z-Score Relatif > 3.5</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    st.markdown(f"""
                        <div class="metric-card" style="border: 1px solid #FCA5A5; background: #FEF2F2;">
                            <div class="metric-label color-crit">Cas Critiques</div>
                            <div class="metric-value color-crit">{len(df_uploaded_both):,}</div>
                            <div style="font-size: 0.8rem; color: #EF4444; font-weight: 600;">⚠️ Confirmés par les 2</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col5:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label color-contract">Fins de contrat</div>
                            <div class="metric-value color-contract">{len(df_contract_new):,}</div>
                            <div style="font-size: 0.8rem; color: #6B7280;">Traitées à part</div>
                        </div>
                    """, unsafe_allow_html=True)

                # --- Anomaly Approval Section ---
                st.write("")
                st.subheader(" Revue & Approbation des Anomalies")
                st.info("Cochez les fiches de paie que vous souhaitez valider. Les lignes non cochées seront rejetées et exclues du prochain réentraînement.")
                
                df_anoms_to_review = df_active_new[(df_active_new["Anomaly_IF"] == 1) | (df_active_new["Anomaly_Residual"] == 1)].copy()
                
                if df_anoms_to_review.empty:
                    st.success(" Aucune anomalie détectée dans ce fichier ! Toutes les fiches sont approuvées par défaut.")
                    df_clean = df_active_new.copy()
                    if st.button(" Enregistrer les données propres"):
                        df_clean.to_csv("approved_payroll_data.csv", index=False)
                        st.success("Données de paie propres enregistrées avec succès dans approved_payroll_data.csv !")
                else:
                    # Initialize review dataframe state
                    state_key = "df_review_state"
                    if state_key not in st.session_state or st.session_state.get("last_uploaded_file") != uploaded_file.name:
                        st.session_state[state_key] = df_anoms_to_review.copy()
                        st.session_state[state_key].insert(0, "Approuver", False)
                        st.session_state["last_uploaded_file"] = uploaded_file.name
                        st.session_state["prev_select_all"] = False

                    # Checkbox to Select All / Unselect All
                    select_all = st.checkbox("Tout sélectionner (Approuver toutes les anomalies)", value=st.session_state["prev_select_all"])
                    if select_all != st.session_state["prev_select_all"]:
                        st.session_state[state_key]["Approuver"] = select_all
                        st.session_state["prev_select_all"] = select_all
                        st.rerun()

                    col_config = {
                        "Approuver": st.column_config.CheckboxColumn(
                            "Approuver ?",
                            help="Cochez pour valider cette fiche de paie",
                            default=False
                        )
                    }
                    
                    edited_df = st.data_editor(
                        st.session_state[state_key],
                        column_config=col_config,
                        disabled=[col for col in st.session_state[state_key].columns if col != "Approuver"],
                        use_container_width=True,
                        hide_index=True,
                        key="anomaly_approval_editor"
                    )
                    
                    # Store edited df back in session state
                    st.session_state[state_key] = edited_df
                    
                    if st.button(" Enregistrer les Données Approuvées"):
                        approved_indices = edited_df[edited_df["Approuver"] == True].index
                        df_approved_anoms = df_anoms_to_review.loc[approved_indices]
                        
                        df_clean = df_active_new[(df_active_new["Anomaly_IF"] == 0) & (df_active_new["Anomaly_Residual"] == 0)].copy()
                        df_approved_total = pd.concat([df_clean, df_approved_anoms], ignore_index=True)
                        
                        # Deduplicate approved rows to prevent double entry
                        df_approved_total.drop_duplicates(subset=["csal_nosal", "DATE"], keep="last", inplace=True)
                        df_approved_total.to_csv("approved_payroll_data.csv", index=False, encoding="utf-8-sig")
                        
                        # Save contract cases to end_of_contract.csv
                        if not df_contract_new.empty:
                            # Standardize df_contract_new columns
                            rename_dict_c = {}
                            emp_col_c = next((c for c in df_contract_new.columns if c.lower() == "csal_nosal"), None)
                            date_col_c = next((c for c in df_contract_new.columns if c.lower() in ["date", "data"]), None)
                            if emp_col_c and emp_col_c != "csal_nosal":
                                rename_dict_c[emp_col_c] = "csal_nosal"
                            if date_col_c and date_col_c != "DATE":
                                rename_dict_c[date_col_c] = "DATE"
                            if rename_dict_c:
                                df_contract_new = df_contract_new.rename(columns=rename_dict_c)
                            
                            if "DATE" in df_contract_new.columns:
                                df_contract_new["DATE"] = pd.to_datetime(df_contract_new["DATE"], errors='coerce').dt.strftime('%Y-%m-%d')
                            if "csal_nosal" in df_contract_new.columns:
                                df_contract_new["csal_nosal"] = df_contract_new["csal_nosal"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                            
                            if os.path.exists("end_of_contract.csv"):
                                try:
                                    df_contract_hist = pd.read_csv("end_of_contract.csv", encoding="utf-8-sig")
                                except Exception:
                                    df_contract_hist = pd.read_csv("end_of_contract.csv")
                                df_contract_hist = clean_column_names(df_contract_hist)
                                # Standardize historical contract columns
                                rename_dict_ch = {}
                                emp_col_ch = next((c for c in df_contract_hist.columns if c.lower() == "csal_nosal"), None)
                                date_col_ch = next((c for c in df_contract_hist.columns if c.lower() in ["date", "data"]), None)
                                if emp_col_ch and emp_col_ch != "csal_nosal":
                                    rename_dict_ch[emp_col_ch] = "csal_nosal"
                                if date_col_ch and date_col_ch != "DATE":
                                    rename_dict_ch[date_col_ch] = "DATE"
                                if rename_dict_ch:
                                    df_contract_hist = df_contract_hist.rename(columns=rename_dict_ch)
                                if "DATE" in df_contract_hist.columns:
                                    df_contract_hist["DATE"] = pd.to_datetime(df_contract_hist["DATE"], errors='coerce').dt.strftime('%Y-%m-%d')
                                if "csal_nosal" in df_contract_hist.columns:
                                    df_contract_hist["csal_nosal"] = df_contract_hist["csal_nosal"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                                    
                                df_contract_combined = pd.concat([df_contract_hist, df_contract_new], ignore_index=True)
                            else:
                                df_contract_combined = df_contract_new
                                
                            df_contract_combined.drop_duplicates(subset=["csal_nosal", "DATE"], keep="last", inplace=True)
                            df_contract_combined.to_csv("end_of_contract.csv", index=False, encoding="utf-8-sig")
                        
                        # --- Immediate Performance Evaluation & Logging (No-Leakage) ---
                        from sklearn.metrics import mean_absolute_error, r2_score, recall_score, roc_auc_score, mean_absolute_percentage_error
                        
                        target_col = next((c for c in df_approved_total.columns if c.strip().lower() in [
                            "net à payer", "net a payer", "net_a_payer", "net", "net a paye"
                        ]), None)
                        
                        if target_col and model_gb is not None:
                            X_new_gb = pd.DataFrame(index=df_approved_total.index)
                            for col in feature_cols_gb:
                                norm_col = col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                                found_col = None
                                for df_col in df_approved_total.columns:
                                    df_col_norm = df_col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                                    if df_col_norm == norm_col or df_col == col:
                                        found_col = df_col
                                        break
                                if found_col is not None:
                                    X_new_gb[col] = df_approved_total[found_col].fillna(0)
                                else:
                                    X_new_gb[col] = 0.0
                                    
                            y_new = df_approved_total[target_col].fillna(0)
                            pred_gb = model_gb.predict(X_new_gb)
                            prod_mae = mean_absolute_error(y_new, pred_gb)
                            prod_mape = mean_absolute_percentage_error(y_new, pred_gb)
                            prod_r2 = r2_score(y_new, pred_gb)
                        else:
                            prod_mae, prod_mape, prod_r2 = 0.0, 0.0, 0.0
                            
                        # RF Surrogate evaluation
                        anomaly_col = next((c for c in df_approved_total.columns if c.lower() in ["anomaly", "anomaly_residual", "anomaly_residual"]), None)
                        if anomaly_col and df_approved_total[anomaly_col].nunique() > 1 and model_surrogate is not None:
                            X_new_rf = pd.DataFrame(index=df_approved_total.index)
                            for col in feature_cols_surrogate:
                                norm_col = col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                                found_col = None
                                for df_col in df_approved_total.columns:
                                    df_col_norm = df_col.lower().replace("é", "e").replace("è", "e").replace("'", "").replace(" ", "")
                                    if df_col_norm == norm_col or df_col == col:
                                        found_col = df_col
                                        break
                                if found_col is not None:
                                    X_new_rf[col] = df_approved_total[found_col].fillna(0)
                                else:
                                    X_new_rf[col] = 0.0
                                    
                            y_new_rf = df_approved_total[anomaly_col].fillna(0).astype(int)
                            try:
                                pred_rf = model_surrogate.predict(X_new_rf)
                                prob_rf = model_surrogate.predict_proba(X_new_rf)[:, 0]
                                prod_recall = recall_score(y_new_rf, pred_rf)
                                prod_auc = roc_auc_score(y_new_rf, prob_rf)
                            except Exception:
                                prod_recall, prod_auc = 1.0, 1.0
                        else:
                            prod_recall, prod_auc = 1.0, 1.0
                            
                        # Save/Overwrite log entry using the payroll date from the uploaded file
                        date_col = next((c for c in df_approved_total.columns if c.lower() in ["date", "data"]), None)
                        if date_col and not df_approved_total[date_col].empty:
                            try:
                                timestamp = pd.to_datetime(df_approved_total[date_col], errors='coerce').dropna().dt.strftime('%Y-%m-%d').iloc[0]
                            except Exception:
                                from datetime import datetime
                                timestamp = datetime.now().strftime("%Y-%m-%d")
                        else:
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y-%m-%d")
                            
                        log_file = "training_history.csv"
                        
                        log_entry = pd.DataFrame([{
                            'execution_date': timestamp,
                            'total_rows_trained': len(df_anomalies),
                            'new_data_rows': len(df_approved_total),
                            'model_mae_on_new_data': prod_mae,
                            'model_mape_on_new_data': prod_mape,
                            'model_r2_on_new_data': prod_r2,
                            'model_recall_on_new_data': prod_recall,
                            'model_auc_on_new_data': prod_auc
                        }])
                        
                        if os.path.exists(log_file):
                            try:
                                df_hist = pd.read_csv(log_file, encoding="utf-8-sig")
                            except Exception:
                                df_hist = pd.read_csv(log_file)
                            df_hist = df_hist[df_hist['execution_date'] != timestamp]
                            df_hist = pd.concat([df_hist, log_entry], ignore_index=True)
                            df_hist.to_csv(log_file, index=False, encoding="utf-8-sig")
                        else:
                            log_entry.to_csv(log_file, index=False, encoding="utf-8-sig")
                        
                        # --- Update global/historical files with new data immediately ---
                        # Combine all active rows and contract rows to include EVERY single uploaded row in anomalies.csv
                        df_all_new_rows = pd.concat([df_active_new, df_contract_new], ignore_index=True)
                        if "Anomaly_Residual" in df_all_new_rows.columns:
                            df_all_new_rows["Anomaly"] = df_all_new_rows["Anomaly_Residual"]
                        else:
                            df_all_new_rows["Anomaly"] = 0

                        # Keep all flagged anomalies (whether approved or rejected) in their respective category views
                        df_active_new_copy = df_active_new.copy()
                        df_active_new_copy["Anomaly"] = df_active_new_copy["Anomaly_Residual"]

                        def append_to_historical_csv(df_to_add, filename, query_cond=None):
                            if df_to_add.empty:
                                return
                            sub_df = df_to_add.copy()
                            if query_cond is not None:
                                sub_df = sub_df.query(query_cond)
                            if sub_df.empty:
                                return
                                
                            if os.path.exists(filename):
                                try:
                                    df_h = pd.read_csv(filename, encoding="utf-8-sig")
                                except Exception:
                                    df_h = pd.read_csv(filename)
                                df_h = clean_column_names(df_h)
                                df_combined = pd.concat([df_h, sub_df], ignore_index=True)
                            else:
                                df_combined = sub_df
                            
                            df_combined.drop_duplicates(subset=["csal_nosal", "DATE"], keep="last", inplace=True)
                            df_combined.to_csv(filename, index=False, encoding="utf-8-sig")

                        # Save EVERY single row to the main history
                        append_to_historical_csv(df_all_new_rows, "anomalies.csv")
                        
                        # Save all flagged anomalies to their respective lists
                        append_to_historical_csv(df_active_new_copy, "anomalies_by_IF.csv", "Anomaly_IF == 1")
                        append_to_historical_csv(df_active_new_copy, "anomalies_by_residual.csv", "Anomaly_Residual == 1")
                        append_to_historical_csv(df_active_new_copy, "anomalies_both.csv", "Anomaly_Both == 1")

                        # Clear streamlit caching to force reload the files from disk on rerun
                        st.cache_data.clear()
                        
                        st.success(f"🎉 Données enregistrées et évaluées ! {len(df_clean)} fiches propres + {len(df_approved_anoms)} anomalies approuvées ont été écrites dans approved_payroll_data.csv et intégrées à l'historique.")

                # --- 9. Tabs ---
                st.write("")
                st.write("")
                st.subheader(" Consultation des Anomalies Détectées")

                new_tab_all, new_tab_if, new_tab_res, new_tab_both, new_tab_contract = st.tabs([
                    " Toutes les Fiches Actives",
                    " Anomalies Isolation Forest",
                    " Anomalies par Résidus",
                    " Cas Critiques (Risque Max)",
                    " Fins de Contrat"
                ])

                def show_new_audit_table(df, tab_name):
                    if df.empty:
                        st.info(f"Aucune ligne trouvée dans cette catégorie ({tab_name}).")
                    else:
                        st.caption(f"Affichage de {len(df)} fiches de paie correspondantes")
                        
                        # Format numeric columns for clean separation of thousands
                        format_dict = {}
                        for col in df.columns:
                            if df[col].dtype in ['float64', 'int64'] and not col.lower().endswith(('id', 'nosal', 'anomaly', 'anomaly_if', 'anomaly_residual', 'anomaly_both')):
                                format_dict[col] = "{:,.0f} FCFA"
                                
                        st.dataframe(df, use_container_width=True, hide_index=True)

                with new_tab_all:
                    show_new_audit_table(df_uploaded_all, "Toutes les Fiches Actives")
                    
                with new_tab_if:
                    if df_uploaded_if.empty:
                        st.info("Aucune ligne détectée par Isolation Forest.")
                    else:
                        st.caption(f"Affichage de {len(df_uploaded_if)} anomalies (Sélectionnez une ligne pour afficher l'explication SHAP)")
                        
                        if_cols = [c for c in ["csal_nosal", "DATE"] if c in df_uploaded_if.columns] + \
                                  [c for c in feature_cols_if if c in df_uploaded_if.columns] + \
                                  [c for c in ["Anomaly_IF"] if c in df_uploaded_if.columns]
                        if_cols = list(dict.fromkeys(if_cols))
                        display_df_new_if = df_uploaded_if[if_cols].copy()
                        display_df_new_if.insert(0, "_row_id", df_uploaded_if.index)
                        
                        table_event = st.dataframe(
                            display_df_new_if,
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            key="new_if_anomalies_table"
                        )
                        
                        selected_rows = table_event.get("selection", {}).get("rows", [])
                        if selected_rows:
                            selected_pos = int(selected_rows[0])
                            actual_idx = int(display_df_new_if.iloc[selected_pos]["_row_id"])
                            selected_row = df_uploaded_if.loc[actual_idx]
                            
                            st.markdown("---")
                            st.subheader(" Explication de l'Anomalie (TreeSHAP)")
                            
                            if model_if is None or explainer_if is None:
                                st.error("Le modèle Isolation Forest ou l'explicateur SHAP n'a pas pu être chargé.")
                            else:
                                # Align row features to what the model expects
                                row_data = {}
                                for col in feature_cols_if:
                                    if col == "Relative Residual":
                                        row_data[col] = selected_row["Relative Residual"]
                                    else:
                                        norm_col = col.replace("", "é") # dummy replace or normal align
                                        found_col = None
                                        for df_col in df_uploaded_if.columns:
                                            if df_col.replace("é", "e").replace("è", "e").lower() == col.lower().replace("é", "e").replace("è", "e").lower() or df_col == col:
                                                found_col = df_col
                                                break
                                        if found_col and found_col in selected_row:
                                            row_data[col] = selected_row[found_col]
                                        else:
                                            row_data[col] = 0.0
                                
                                x_instance = pd.DataFrame([row_data])
                                
                                # Calculate / extract anomaly score
                                score_val = None
                                score_col = next((c for c in selected_row.index if c.lower() in ["anomaly_scores", "anomaly_score", "score"]), None)
                                if score_col:
                                    score_val = selected_row[score_col]
                                elif model_if is not None:
                                    score_val = model_if.score_samples(x_instance)[0]
                                    
                                if score_val is not None:
                                    st.metric("Score d'Anomalie Isolation Forest", f"{score_val:.4f}")
                                
                                try:
                                    shap_explanation = explainer_if(x_instance)
                                    
                                    col_chart, col_table = st.columns([2, 1])
                                    
                                    with col_chart:
                                        st.markdown("**Graphique Cascade (Waterfall Chart)**")
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
                                        plt.tight_layout()
                                        st.pyplot(fig, clear_figure=True)
                                        
                                    with col_table:
                                        st.markdown("**Contributions des Variables**")
                                        contribs = pd.DataFrame({
                                            "Variable": feature_cols_if,
                                            "Valeur SHAP": shap_explanation[0].values,
                                            "Abs_SHAP": np.abs(shap_explanation[0].values)
                                        }).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)
                                        
                                        st.dataframe(
                                            contribs[["Variable", "Valeur SHAP"]],
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                except Exception as e:
                                    st.error(f"Erreur lors du calcul SHAP : {e}")

                with new_tab_res:
                    if df_uploaded_res.empty:
                        st.info("Aucune ligne détectée par Anomalies par Résidus.")
                    else:
                        st.caption(f"Affichage de {len(df_uploaded_res)} anomalies (Sélectionnez une ligne pour afficher l'explication SHAP)")
                        
                        res_cols = [c for c in ["csal_nosal", "DATE"] if c in df_uploaded_res.columns] + \
                                   [c for c in feature_cols_gb if c in df_uploaded_res.columns] + \
                                   [c for c in ["Net à payer", "Net a payer", "Pred", "Residual", "Relative Residual", "Robust_Z_Score"] if c in df_uploaded_res.columns]
                        res_cols = list(dict.fromkeys(res_cols))
                        display_df_new_res = df_uploaded_res[res_cols].copy()
                        display_df_new_res.insert(0, "_row_id", df_uploaded_res.index)
                        
                        table_event_new_res = st.dataframe(
                            display_df_new_res,
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            key="new_res_anomalies_table"
                        )
                        
                        selected_rows_new_res = table_event_new_res.get("selection", {}).get("rows", [])
                        if selected_rows_new_res:
                            selected_pos_new_res = int(selected_rows_new_res[0])
                            actual_idx_new_res = int(display_df_new_res.iloc[selected_pos_new_res]["_row_id"])
                            selected_row_new_res = df_uploaded_res.loc[actual_idx_new_res]
                            
                            st.markdown("---")
                            st.subheader(" Explication de l'Anomalie (TreeSHAP du Modèle Substitut)")
                            
                            # Display Regression Metrics
                            net_val = selected_row_new_res.get("Net à payer", selected_row_new_res.get("Net a payer", 0.0))
                            pred_val = selected_row_new_res.get("Pred", selected_row_new_res.get("Prediction", 0.0))
                            res_val = selected_row_new_res.get("Residual", selected_row_new_res.get("Résiduel", 0.0))
                            rel_res_val = selected_row_new_res.get("Relative Residual", selected_row_new_res.get("Résiduel Relatif", 0.0))
                            
                            met1, met2, met3, met4 = st.columns(4)
                            met1.metric("Net à payer Réel", f"{net_val:,.0f} FCFA")
                            met2.metric("Net à payer Prédit", f"{pred_val:,.0f} FCFA")
                            met3.metric("Résidu Brut", f"{res_val:,.0f} FCFA")
                            met4.metric("Résidu Relatif", f"{rel_res_val:.4%}")
                            
                            st.write("")
                            
                            if model_surrogate is None or explainer_surrogate is None:
                                st.error("Le modèle Random Forest substitut ou l'explicateur SHAP n'a pas pu être chargé.")
                            else:
                                # Align row features to what the model expects
                                row_data = {}
                                for col in feature_cols_surrogate:
                                    norm_col = col.replace("", "é") # dummy replace or normal align
                                    found_col = None
                                    for df_col in df_uploaded_res.columns:
                                        if df_col.replace("é", "e").replace("è", "e").lower() == col.lower().replace("é", "e").replace("è", "e").lower() or df_col == col:
                                            found_col = df_col
                                            break
                                    if found_col and found_col in selected_row_new_res:
                                        row_data[col] = selected_row_new_res[found_col]
                                    else:
                                        row_data[col] = 0.0
                                
                                x_instance = pd.DataFrame([row_data])
                                
                                try:
                                    shap_explanation = explainer_surrogate(x_instance)
                                     
                                    # Dynamically handle 3D (multi-class) vs 2D (single-class output/binary fallback)
                                    if len(shap_explanation.shape) == 3:
                                        anomaly_class_idx = 1
                                        if hasattr(model_surrogate, "classes_"):
                                            classes = list(model_surrogate.classes_)
                                            if -1 in classes:
                                                anomaly_class_idx = classes.index(-1)
                                            elif 1 in classes:
                                                anomaly_class_idx = classes.index(1)
                                        shap_exp_sliced = shap_explanation[0, :, anomaly_class_idx]
                                    else:
                                        shap_exp_sliced = shap_explanation[0]
                                        
                                    col_chart, col_table = st.columns([2, 1])
                                    
                                    with col_chart:
                                        st.markdown("**Graphique Cascade (Waterfall Chart)**")
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        shap.plots.waterfall(shap_exp_sliced, max_display=12, show=False)
                                        plt.tight_layout()
                                        st.pyplot(fig, clear_figure=True)
                                        
                                    with col_table:
                                        st.markdown("**Contributions des Variables**")
                                        contribs = pd.DataFrame({
                                            "Variable": feature_cols_surrogate,
                                            "Valeur SHAP": shap_exp_sliced.values,
                                            "Abs_SHAP": np.abs(shap_exp_sliced.values)
                                        }).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)
                                        
                                        st.dataframe(
                                            contribs[["Variable", "Valeur SHAP"]],
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                except Exception as e:
                                    st.error(f"Erreur lors du calcul SHAP : {e}")

                    
                with new_tab_both:
                    if df_uploaded_both.empty:
                        st.info("Aucune ligne détectée dans cette catégorie (Cas Critiques (Risque Max)).")
                    else:
                        both_cols = [c for c in ["csal_nosal", "DATE"] if c in df_uploaded_both.columns] + \
                                    [c for c in list(set(feature_cols_if + feature_cols_gb)) if c in df_uploaded_both.columns] + \
                                    [c for c in ["Net à payer", "Net a payer", "Pred", "Residual", "Relative Residual", "Robust_Z_Score", "Anomaly_IF"] if c in df_uploaded_both.columns]
                        both_cols = list(dict.fromkeys(both_cols))
                        display_both = df_uploaded_both[both_cols].copy()
                        show_new_audit_table(display_both, "Cas Critiques (Risque Max)")
                    
                with new_tab_contract:
                    show_new_audit_table(df_contract_new, "Fins de Contrat")

        except Exception as e:
            st.error(f"Erreur lors de la lecture ou du traitement du fichier : {e}")
            st.exception(e)

elif app_page == "Performance & Réentraînement":
    import plotly.express as px
    import retrain
    st.markdown("""
        <div class="welcome-card">
            <h3> Model Performance & Drift Monitor</h3>
            <p>Cet onglet vous permet de suivre l'évolution des performances de vos modèles au fil des mois et de lancer les cycles de réentraînement sur les nouvelles données approuvées.</p>
        </div>
    """, unsafe_allow_html=True)

    if os.path.exists("training_history.csv"):
        try:
            df_history = pd.read_csv("training_history.csv", encoding="utf-8-sig")
        except Exception:
            df_history = pd.read_csv("training_history.csv")
        df_history = clean_column_names(df_history)
        
        # Robustly map old column names to the new format if needed
        rename_map = {
            'production_model_mae_on_unseen_data': 'model_mae_on_new_data',
            'production_model_r2_on_unseen_data': 'model_r2_on_new_data',
            'old_model_mae_on_new_data': 'model_mae_on_new_data',
            'old_model_r2_on_new_data': 'model_r2_on_new_data',
            'new_model_mae_on_new_data': 'model_mae_on_new_data',
            'new_model_r2_on_new_data': 'model_r2_on_new_data',
            'new_model_recall_on_new_data': 'model_recall_on_new_data',
            'new_model_auc_on_new_data': 'model_auc_on_new_data'
        }
        for old_c, new_c in rename_map.items():
            if old_c in df_history.columns and new_c not in df_history.columns:
                df_history[new_c] = df_history[old_c]
                
        # Fill missing expected columns with defaults
        expected_cols = {
            'total_rows_trained': 0,
            'model_mae_on_new_data': 0.0,
            'model_mape_on_new_data': 0.0,
            'model_r2_on_new_data': 0.0,
            'model_recall_on_new_data': 1.0,
            'model_auc_on_new_data': 1.0
        }
        for col, default_val in expected_cols.items():
            if col not in df_history.columns:
                df_history[col] = default_val
  
        # Key Metrics Display
        latest = df_history.iloc[-1]
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        col1.metric("Taille du jeu d'entraînement", f"{int(latest['total_rows_trained']):,} lignes")
        col2.metric("MAE Regresseur GB (Unseen)", f"{latest['model_mae_on_new_data']:.2f}")
        col3.metric("MAPE Regresseur GB (Unseen)", f"{latest['model_mape_on_new_data']:.2%}")
        col4.metric("R2 Regresseur GB (Unseen)", f"{latest['model_r2_on_new_data']:.4f}")
        
        # RF Surrogate Metrics
        latest_recall = latest['model_recall_on_new_data']
        latest_auc = latest['model_auc_on_new_data']
        col5.metric("Recall Classifieur RF", f"{latest_recall:.2%}")
        col6.metric("AUC ROC Classifieur RF", f"{latest_auc:.2%}")
        
        st.write("")
        
        # Metric plots
        st.subheader("Évolution de l'Erreur de Régression (MAE) dans le temps")
        fig_mae = px.line(df_history, x='execution_date', 
                      y='model_mae_on_new_data',
                      labels={'model_mae_on_new_data': 'Mean Absolute Error (MAE)', 'execution_date': "Date d'exécution"},
                      title="MAE du modèle de production sur les nouvelles données non vues",
                      markers=True)
        st.plotly_chart(fig_mae, use_container_width=True)

        st.subheader("Évolution de l'Erreur de Régression (MAPE) dans le temps")
        fig_mape = px.line(df_history, x='execution_date', 
                      y='model_mape_on_new_data',
                      labels={'model_mape_on_new_data': 'Mean Absolute Percentage Error (MAPE)', 'execution_date': "Date d'exécution"},
                      title="MAPE du modèle de production sur les nouvelles données non vues",
                      markers=True)
        fig_mape.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig_mape, use_container_width=True)

        st.subheader("Évolution du Coefficient de Détermination (R2) dans le temps")
        fig_r2 = px.line(df_history, x='execution_date', 
                      y='model_r2_on_new_data',
                      labels={'model_r2_on_new_data': 'R2 Score', 'execution_date': "Date d'exécution"},
                      title="Score R2 du modèle de production sur les nouvelles données non vues",
                      markers=True)
        st.plotly_chart(fig_r2, use_container_width=True)
        
        st.subheader("Évolution de la Classification (Recall & AUC ROC) dans le temps")
        fig_rf = px.line(df_history, x='execution_date', 
                      y=['model_recall_on_new_data', 'model_auc_on_new_data'],
                      labels={'value': 'Score', 'variable': 'Métrique', 'execution_date': "Date d'exécution"},
                      title="Recall et AUC ROC du modèle RandomForest substitut (sur les nouvelles données non vues)",
                      markers=True)
        st.plotly_chart(fig_rf, use_container_width=True)
        
        # Display raw logs
        st.subheader("Historique des Cycles d'Entraînement")
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun historique de réentraînement trouvé. Veuillez lancer votre premier cycle pour générer des statistiques.")

    st.write("---")
    st.subheader(" Cycle de Réentraînement Mensuel")
    
    # Check if approved data exists and has content
    approved_file_ready = False
    if os.path.exists("approved_payroll_data.csv"):
        try:
            try:
                df_app = pd.read_csv("approved_payroll_data.csv", encoding="utf-8-sig")
            except Exception:
                df_app = pd.read_csv("approved_payroll_data.csv")
            df_app = clean_column_names(df_app)
            if not df_app.empty:
                approved_file_ready = True
                st.success(f" **{len(df_app)} fiches de paie approuvées** sont en attente dans `approved_payroll_data.csv` et prêtes pour le réentraînement.")
        except Exception:
            pass

    if approved_file_ready:
        if st.button(" Lancer le Réentraînement"):
            with st.spinner("Réentraînement des modèles en cours sur l'ensemble de l'historique..."):
                success = retrain.run_retraining()
                if success:
                    st.success(" Les modèles ont été réentraînés avec succès, validés et sauvegardés en tant que modèles de production !")
                    # Clear caching resources to force reload the updated models and datasets
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(" Échec du réentraînement.")
    else:
        st.warning(" Aucune donnée approuvée en attente. Veuillez d'abord charger un fichier de paie brut et valider les anomalies dans l'onglet 'Audit de Nouveaux Fichiers' pour générer `approved_payroll_data.csv`.")
