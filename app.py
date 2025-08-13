# app like telling the AI to do fewer practice rounds. For a dataset of this size, it can learn the main patterns very quickly, and extra rounds provide diminishing returns at the cost of more waiting time.
3.  **Introduce Subsampling:** We will.py (v53 - FINAL, HIGH-PERFORMANCE MODEL TUNING)
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px
import gc
import tell the model to use a random 80% of the data (`bagging_fraction`) and 80% of os
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from passlib.context the features (`feature_fraction`) for each training cycle. This is a powerful technique that dramatically speeds up training without losing accuracy, as it forces the model to learn more robust patterns.

These are standard, professional techniques for tuning a model for a import CryptContext
import smtplib
from email.message import EmailMessage

# --- Page Configuration & State --- high-speed production environment.

---

### The Final, High-Performance `app.py` Script

Please
st.set_page_config(layout="wide", page_title="ITC AI Budget Allocation Portal")
if replace the entire contents of your `app.py` file with this definitive, high-performance version. The changes are all 'resolved_issues' not in st.session_state: st.session_state.resolved_issues = set()
if inside the `get_allocation_recommendations` function.

```python
# app.py (v53 - 'resolved_oos' not in st.session_state: st.session_state.resolved_oos = set FINAL, MAXIMUM PERFORMANCE TUNING)
import streamlit as st
import pandas as pd
import numpy as np
import light()

# --- Hashing Context ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#gbm as lgb
import plotly.express as px
import gc
import os
import streamlit.components.v1 as components ----------------- The Backend "Engine" -----------------
@st.cache_data
def load_data(url
from datetime import datetime, timedelta
from passlib.context import CryptContext
import smtplib
from email):
    file_id = url.split('/d/')[1].split('/')[0]
    csv.message import EmailMessage

# --- Page Configuration & State ---
st.set_page_config(layout="wide_url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format", page_title="ITC AI Budget Allocation Portal")
if 'resolved_issues' not in st.session_=csv'
    df = pd.read_csv(csv_url)
    df.columns = [state: st.session_state.resolved_issues = set()
if 'resolved_oos' not in st.session_state: st.session_state.resolved_oos = set()

# --- Hashing Context ---
pwdc.strip() for c in df.columns]
    return df

@st.cache_data
def load_demo_data(input_file):
    xls = pd.ExcelFile(input_file); df_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ----------------- The Backend "Engine" _s1 = pd.read_excel(xls, sheet_name='s1'); df_s2 = pd.read_excel(xls, sheet_name='s2')
    if 'Pincode' in-----------------
@st.cache_data
def load_data(url):
    file_id = url.split('/ df_s1.columns: df_s1.rename(columns={'Pincode': 'Pin Code'},d/')[1].split('/')[0]
    csv_url = f'https://docs.google.com/spread inplace=True)
    if 'Pincode' in df_s2.columns: df_s2.rename(columns={'Pincode': 'Pin Code'}, inplace=True)
    if 'Pin Code' in df_sheets/d/{file_id}/export?format=csv'
    df = pd.read_csv(csv_s1.columns: df_s1['Pin Code'] = df_s1['Pin Code'].astype(strurl)
    df.columns = [c.strip() for c in df.columns]
    return df).str.strip()
    if 'Pin Code' in df_s2.columns: df_s2['Pin

@st.cache_data
def load_demo_data(input_file):
    xls = pd.Excel Code'] = df_s2['Pin Code'].astype(str).str.strip()
    return df_File(input_file); df_s1 = pd.read_excel(xls, sheet_name='s1'); dfs1, df_s2

@st.cache_data
def get_allocation_recommendations(df,_s2 = pd.read_excel(xls, sheet_name='s2')
    if 'P budget_multiplier, roas_weight):
    ntb_weight = 1.0 - roas_weightincode' in df_s1.columns: df_s1.rename(columns={'Pincode':
    grouping_keys = ["Time Slot", "Pin Code", "Tier", "Platform", "Brand", "SK 'Pin Code'}, inplace=True)
    if 'Pincode' in df_s2.columns: dfU", "Ad Type", "OOS Flag", "Content Issue Flag"]
    grouping_keys = [_s2.rename(columns={'Pincode': 'Pin Code'}, inplace=True)
    if 'key for key in df.columns if key in grouping_keys]
    if pd.api.types.isPin Code' in df_s1.columns: df_s1['Pin Code'] = df_s1['Pin Code'].astype(str).str.strip()
    if 'Pin Code' in df_s2_string_dtype(df["NTB (%)"]): df['Clean_NTB'] = pd.to_numeric(df.columns: df_s2['Pin Code'] = df_s2['Pin Code'].astype(str).str.strip()
    return df_s1, df_s2

@st.cache_data
["NTB (%)"].str.replace('%', '', regex=False), errors='coerce')
    else: df['Cleandef get_allocation_recommendations(df, budget_multiplier, roas_weight):
    ntb_weight = _NTB'] = pd.to_numeric(df["NTB (%)"], errors='coerce')
    agg1.0 - roas_weight
    grouping_keys = ["Time Slot", "Pin Code", "Tier_dict = {'Budget Spent': 'sum', 'Direct Sales': 'sum', 'Clean_NTB': 'mean'}", "Platform", "Brand", "SKU", "Ad Type", "OOS Flag", "Content Issue Flag"]
    df_agg = df.groupby(grouping_keys, as_index=False).agg(agg_dict
    grouping_keys = [key for key in df.columns if key in grouping_keys]
    if pd)
    df_agg['Aggregated_ROAS'] = df_agg['Direct Sales'] / (df_agg.api.types.is_string_dtype(df["NTB (%)"]): df['Clean_NTB'] =['Budget Spent'] + 1e-6)
    df_agg['Optimization_Score'] = (roas_weight * df_agg['Aggregated_ROAS']) + (ntb_weight * df_agg[' pd.to_numeric(df["NTB (%)"].str.replace('%', '', regex=False), errors='Clean_NTB'])
    df_agg.fillna(0, inplace=True)
    features = groupingcoerce')
    else: df['Clean_NTB'] = pd.to_numeric(df["NTB (%)_keys; X = df_agg[features]; y = df_agg['Optimization_Score']
    for"], errors='coerce')
    agg_dict = {'Budget Spent': 'sum', 'Direct Sales': 'sum', col in features: X[col] = X[col].astype("category")
    lgb_data = l 'Clean_NTB': 'mean'}
    df_agg = df.groupby(grouping_keys, asgb.Dataset(X, label=y, categorical_feature=features)
    
    # <<< --- PERFORMANCE_index=False).agg(agg_dict)
    df_agg['Aggregated_ROAS'] = df_agg['Direct Sales'] / (df_agg['Budget Spent'] + 1e-6)
    df_agg[' TUNING: Faster model parameters --- >>>
    params = {
        "objective": "regression_l1", "metric": "rmse", "verbosity": -1, "seed": 42,
        "n_estimators": Optimization_Score'] = (roas_weight * df_agg['Aggregated_ROAS']) + (nt50,          # Reduced from 80 for much faster training
        "num_leaves": 15b_weight * df_agg['Clean_NTB'])
    df_agg.fillna(0, inplace=True)
    features = grouping_keys; X = df_agg[features]; y = df_agg,          # Keeps model simple
        "learning_rate": 0.1,
        "feature_fraction": 0.8,     # New: Use 80% of features for each tree, boosting speed
        "bagging_['Optimization_Score']
    for col in features: X[col] = X[col].astype("category")
    fraction": 0.8,     # New: Use 80% of data for each tree, boosting speed
        lgb_data = lgb.Dataset(X, label=y, categorical_feature=features, free_raw"bagging_freq": 1
    }
    model = lgb.train(params, lgb_data)_data=False)
    
    # <<< --- MAXIMUM PERFORMANCE TUNING --- >>>
    params = {
        
    
    df_agg['Predicted_Score'] = model.predict(X).clip(min=0)"objective": "regression_l1", "metric": "rmse", "verbosity": -1, "seed": 4
    current_total_budget = df_agg['Budget Spent'].sum(); new_total_budget = current2,
        "n_estimators": 60,          # Further reduced training rounds for speed
        "_total_budget * budget_multiplier
    brand_sales = df_agg.groupby('Brand')['Direct Sales'].sumnum_leaves": 10,            # Further simplified trees for speed
        "learning_rate": 0.(); brand_proportions = brand_sales / max(1, brand_sales.sum()); brand_budgets1,
        "feature_fraction": 0.8,     # Use a random 80% of features for each tree
        "bagging_fraction": 0.8,     # Use a random 8 = brand_proportions * new_total_budget
    df_agg['Brand_Allocated_Budget'] = df_0% of data for each tree
        "bagging_freq": 1,
        "n_jobsagg['Brand'].map(brand_budgets); df_agg['Brand_Total_Predicted_Score'] =": -1                 # Use all available CPU cores
    }
    model = lgb.train(params, df_agg.groupby('Brand')['Predicted_Score'].transform('sum')
    df_agg['Final_Allocated_Budget'] = df_agg['Brand_Allocated_Budget'] * (df_agg['Predicted_Score'] lgb_data)
    
    df_agg['Predicted_Score'] = model.predict(X).clip / (df_agg['Brand_Total_Predicted_Score'] + 1e-6)); df_agg.(min=0)
    current_total_budget = df_agg['Budget Spent'].sum(); new_total_budgetfillna(0, inplace=True); df_agg['Final_Allocated_Budget'] = df_agg[' = current_total_budget * budget_multiplier
    brand_sales = df_agg.groupby('Brand')['Final_Allocated_Budget'] * (new_total_budget / max(1, df_agg['Final_AllocatedDirect Sales'].sum(); brand_proportions = brand_sales / max(1, brand_sales.sum());_Budget'].sum()))
    return df_agg

def send_oos_email(manager_email, brand brand_budgets = brand_proportions * new_total_budget
    df_agg['Brand_Alloc, sku, pincode, stock_left):
    try:
        sender = st.secrets["email_credentials"]["ated_Budget'] = df_agg['Brand'].map(brand_budgets); df_agg['Brand_Totalsender_email"]; password = st.secrets["email_credentials"]["sender_password"]
        subject = f"_Predicted_Score'] = df_agg.groupby('Brand')['Predicted_Score'].transform('sum')
    df_agg['Final_Allocated_Budget'] = df_agg['Brand_Allocated_Budget'] * (🚨 URGENT: Low Stock Alert for {brand}"; body = f"Hello,\n\nThis is an automated alertdf_agg['Predicted_Score'] / (df_agg['Brand_Total_Predicted_Score'] + 1e-6)); df_agg.fillna(0, inplace=True); df_agg['Final_Allocated_Budget'] =.\n\nThe following item is running low on stock in your area:\n\n- Brand: {brand}\ df_agg['Final_Allocated_Budget'] * (new_total_budget / max(1, df_agg['n- SKU: {sku}\n- Pin Code: {pincode}\n- Stock Left: {stockFinal_Allocated_Budget'].sum()))
    return df_agg

def send_oos_email(manager_left}\n\nPlease take action to restock.\n\nThank you,\nITC AI Operations Bot"
        msg_email, brand, sku, pincode, stock_left):
    # (This function is unchanged)
    try: = EmailMessage(); msg.set_content(body); msg['Subject'] = subject; msg['From'] = sender
        sender = st.secrets["email_credentials"]["sender_email"]; password = st.secrets["email_credentials; msg['To'] = manager_email
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465); server.login(sender, password); server.send_message(msg); server"]["sender_password"]
        subject = f"🚨 URGENT: Low Stock Alert for {brand}"; body.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: = f"Hello,\n\nThis is an automated alert.\n\nThe following item is running low on stock in {e}"); return False

@st.cache_data
def to_csv(df):
    return df your area:\n\n- Brand: {brand}\n- SKU: {sku}\n- Pin Code: {p.to_csv(index=False).encode('utf-8')

# --- Authentication and Main App UI ---
def checkincode}\n- Stock Left: {stock_left}\n\nPlease take action to restock.\n\nThank_password(username, password):
    if username in st.secrets["passwords"]: return pwd_context.verify(password, st.secrets["passwords"][username])
    return False

def login_form():
    col1, col2, col3 = st.columns([1,1,1]);
    with col2:
        if os.path.exists("itc_logo.png"): st.image("itc_logo.png", you,\nITC AI Operations Bot"
        msg = EmailMessage(); msg.set_content(body); msg['Subject'] = subject; msg['From'] = sender; msg['To'] = manager_email
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465); server.login(sender, password); server.send_message(msg); server.quit()
        return True
    except Exception as e width=150)
        st.title("ITC Allocation Portal Login")
        with st.form(":
        st.error(f"Failed to send email: {e}"); return False

@st.cachelogin_form"):
            username = st.text_input("Username"); password = st.text_input("Password",_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Authentication and Main App UI ---
def check_password(username, password):
 type="password"); submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if check_password(username, password): st.session_state["authentication_status    if username in st.secrets["passwords"]: return pwd_context.verify(password, st.secrets["passwords"][username])
    return False

def login_form():
    col1, col2, col3 ="] = True; st.session_state["username"] = username; st.rerun()
                else: st.error("Incorrect username or password")
    return False

if not st.session_state.get("authentication_status", False):
    login_form()
else:
    st.markdown("<h1 style='text-align st.columns([1,1,1]);
    with col2:
        if os.path.exists("itc_logo.png"): st.image("itc_logo.png", width=150)
        st.title("ITC Allocation Portal Login")
        with st.form("login_form"):
            username = st.: center; color: #2c3e50;'>ITC AI-BASED BUDGET ALLOCATION PORTAL</h1>", unsafe_allow_html=True)
    st.sidebar.success(f"Welcome, {st.session_text_input("Username"); password = st.text_input("Password", type="password"); submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if checkstate['username']}!")
    st.sidebar.header("⚙️ Scenario Controls")
    try:
        INPUT_password(username, password): st.session_state["authentication_status"] = True; st.session_state["_DATA_URL = "https://docs.google.com/spreadsheets/d/1jnF_Jusername"] = username; st.rerun()
                else: st.error("Incorrect username or password")
    return False

if not st.session_state.get("authentication_status", False):
    login_form()1X4oq6-f-heomauZrRuFpjlbik9_tYGL5ceoNM/edit?usp=sharing"
        DEMO_XLSX = "demo.xlsx"
        original_df
else:
    st.markdown("<h1 style='text-align: center; color: #2c3e5 = load_data(INPUT_DATA_URL); oos_df, manager_df = load_demo_0;'>ITC AI-BASED BUDGET ALLOCATION PORTAL</h1>", unsafe_allow_html=True)
    st.sidebardata(DEMO_XLSX)
        budget_mult = st.sidebar.slider("Budget Multiplier", 0.5, 2.5, 1.2, 0.1)
        roas_w.success(f"Welcome, {st.session_state['username']}!")
    st.sidebar.header = st.sidebar.slider("ROAS / NTB Weight", 0.0, 1.0, 0("⚙️ Scenario Controls")
    try:
        INPUT_DATA_URL = "https://docs.google.com/.5, 0.05)
        st.sidebar.metric("Resulting NTB % Weight", fspreadsheets/d/1jnF_J1X4oq6-f-heomauZrRuF"{(1.0 - roas_w):.0%}")
        if st.sidebar.button("🚀 Run Predictive Allocation", type="primary", use_container_width=True):
            with st.spinner("🧠pjlbik9_tYGL5ceoNM/edit?usp=sharing"
        DEMO_XLSX Running predictive model..."):
                st.session_state.final_df = get_allocation_recommendations( = "demo.xlsx"
        original_df = load_data(INPUT_DATA_URL); oos_df,original_df.copy(), budget_mult, roas_w)
            st.toast("✅ Allocation complete manager_df = load_demo_data(DEMO_XLSX)
        budget_mult = st.sidebar!", icon="🎉")
        if st.sidebar.button("Logout"):
            st.session_state.clear();.slider("Budget Multiplier", 0.5, 2.5, 1.2, 0.1) st.rerun()
        
        # (The rest of the app UI is unchanged and complete)
        recent_issues
        roas_w = st.sidebar.slider("ROAS / NTB Weight", 0.0, 1 = pd.DataFrame()
        if all(col in original_df.columns for col in ['Date', 'Content.0, 0.5, 0.05)
        st.sidebar.metric("Resulting NTB Issue Flag']): df_copy = original_df.copy(); df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce'); df_copy.dropna(subset=['Date'], inplace=True); end % Weight", f"{(1.0 - roas_w):.0%}")
        if st.sidebar.button_date = df_copy['Date'].max(); start_date = end_date - timedelta(days=2("🚀 Run Predictive Allocation", type="primary", use_container_width=True):
            with st.spinner("🧠 Running predictive model..."):
                st.session_state.final_df = get_allocation_recommendations(); recent_issues = df_copy[(df_copy['Date'] >= start_date) & (df_original_df.copy(), budget_mult, roas_w)
            st.toast("✅ Allocation completecopy['Content Issue Flag'].astype(str).str.lower() == 'yes')]
        unresolved_issues_!", icon="🎉")
        if st.sidebar.button("Logout"):
            st.session_state.clear(); stcount = len(recent_issues[~recent_issues.index.isin(st.session_state.get('resolved_issues', set()))])
        recent_oos = pd.DataFrame()
        if all(col in.rerun()
        
        recent_issues = pd.DataFrame()
        if all(col in original_df.columns for col in ['Date', 'Content Issue Flag']): df_copy = original_df.copy(); df oos_df.columns for col in ['Time', 'Stock_Left']):
            df_oos_copy = oos_df.copy(); df_oos_copy['Parsed_Timestamp'] = pd.to_datetime(df_oos__copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce'); df_copy.copy['Time'].astype(str), errors='coerce'); df_oos_copy['Simulated_Timestamp'] = df_oosdropna(subset=['Date'], inplace=True); end_date = df_copy['Date'].max(); start_date = end_copy['Parsed_Timestamp'].dt.time.apply(lambda t: datetime.combine(datetime.now().date_date - timedelta(days=2); recent_issues = df_copy[(df_copy['Date'] >= start_date) & (df_copy['Content Issue Flag'].astype(str).str.lower() == 'yes')](), t) if pd.notna(t) else pd.NaT)
            time_filter = datetime
        unresolved_issues_count = len(recent_issues[~recent_issues.index.isin(st.now() - timedelta(minutes=30)
            recent_oos = df_oos_copy[(df_oos_copy['Simulated_Timestamp'] >= time_filter) & (df_oos_copy['Stock_Left'] <=.session_state.get('resolved_issues', set()))])
        recent_oos = pd.DataFrame()
        if all(col in oos_df.columns for col in ['Time', 'Stock_Left']):
            df_oos 5)]
        unresolved_oos_count = len(recent_oos[~recent_oos.index._copy = oos_df.copy(); df_oos_copy['Parsed_Timestamp'] = pd.to_datetime(isin(st.session_state.get('resolved_oos', set()))])
        
        tab1, tabdf_oos_copy['Time'].astype(str), errors='coerce'); df_oos_copy['Simulated_Timestamp'] = df_oos_copy['Parsed_Timestamp'].dt.time.apply(lambda t: datetime.combine(datetime2, tab3, tab4, tab5 = st.tabs(["📊 Predictive Allocation", "📋 Raw Data", f.now().date(), t) if pd.notna(t) else pd.NaT)
            time"🚨 Content Issues ({unresolved_issues_count})", f"📉 Low Stock Alerts ({unresolved_oos_count})_filter = datetime.now() - timedelta(minutes=30)
            recent_oos = df_oos_", "🌐 ITC E-COMMERCE"])
        
        with tab1:
            if 'final_df' in st.session_state:
                final_df = st.session_state.final_df;copy[(df_oos_copy['Simulated_Timestamp'] >= time_filter) & (df_oos_copy['Stock st.expander("🔍 Filter Dashboard Results", expanded=True); col1, col2, col3, col4, col_Left'] <= 5)]
        unresolved_oos_count = len(recent_oos[~recent_oos.index.isin(st.session_state.get('resolved_oos', set()))])
        5 = st.columns(5); brands = sorted(final_df['Brand'].unique()); selected_brands = col
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Predictive Allocation1.multiselect("Brand", brands, default=brands); platforms = sorted(final_df['Platform'].unique()); selected", "📋 Raw Data", f"🚨 Content Issues ({unresolved_issues_count})", f"📉 Low_platforms = col2.multiselect("Platform", platforms, default=platforms); ad_types = sorted(final_df Stock Alerts ({unresolved_oos_count})", "🌐 ITC E-COMMERCE"])
        
        with tab1:
            if 'final_df' in st.session_state:
                final_df = st.session_state['Ad Type'].unique()); selected_ad_types = col3.multiselect("Ad Type", ad_types,.final_df; st.expander("🔍 Filter Dashboard Results", expanded=True); col1, col2 default=ad_types); tiers = sorted(final_df['Tier'].unique()); selected_tiers = col4.multiselect("Tier", tiers, default=tiers); time_slots = sorted(final_df['Time Slot'].unique());, col3, col4, col5 = st.columns(5); brands = sorted(final_df['Brand selected_slots = col5.multiselect("Time Slot", time_slots, default=time_slots)
'].unique()); selected_brands = col1.multiselect("Brand", brands, default=brands); platforms = sorted(final                filtered_df = final_df[(final_df['Brand'].isin(selected_brands)) & (final__df['Platform'].unique()); selected_platforms = col2.multiselect("Platform", platforms, default=platformsdf['Platform'].isin(selected_platforms)) & (final_df['Ad Type'].isin(selected_ad_types)) & (final_df['Tier'].isin(selected_tiers)) & (final_df['Time); ad_types = sorted(final_df['Ad Type'].unique()); selected_ad_types = col3 Slot'].isin(selected_slots))]; st.header("Financial Summary"); kpi_cols = st.columns(3);.multiselect("Ad Type", ad_types, default=ad_types); tiers = sorted(final_df['Tier'].unique()); selected_tiers = col4.multiselect("Tier", tiers, default=tiers); time_slots original_budget = filtered_df['Budget Spent'].sum(); new_budget = filtered_df['Final_Allocated_ = sorted(final_df['Time Slot'].unique()); selected_slots = col5.multiselect("TimeBudget'].sum(); sales = filtered_df['Direct Sales'].sum(); kpi_cols[0].metric("Original Slot", time_slots, default=time_slots)
                filtered_df = final_df[(final_df[' Budget", f"${original_budget:,.0f}"); kpi_cols[1].metric("Optimized Budget", fBrand'].isin(selected_brands)) & (final_df['Platform'].isin(selected_platforms)) & ("${new_budget:,.0f}", f"{(new_budget - original_budget):,.0f}");final_df['Ad Type'].isin(selected_ad_types)) & (final_df['Tier'].isin kpi_cols[2].metric("Historical Sales", f"${sales:,.0f}"); st.markdown("---(selected_tiers)) & (final_df['Time Slot'].isin(selected_slots))]; st.header"); st.header("Allocation Visualizations"); viz_cols = st.columns(2); brand_summary = filtered_df("Financial Summary"); kpi_cols = st.columns(3); original_budget = filtered_df['Budget.groupby('Brand')['Final_Allocated_Budget'].sum().sort_values(ascending=False); fig_brand Spent'].sum(); new_budget = filtered_df['Final_Allocated_Budget'].sum(); sales = filtered_df['Direct Sales'].sum(); kpi_cols[0].metric("Original Budget", f"${original_budget: = px.bar(brand_summary, x=brand_summary.index, y='Final_Allocated_Budget',,.0f}"); kpi_cols[1].metric("Optimized Budget", f"${new_budget:,.0f}", f"{(new_budget - original_budget):,.0f}"); kpi_cols[2 title="Optimized Budget by Brand", labels={'Final_Allocated_Budget': 'Budget ($)', 'index': 'Brand].metric("Historical Sales", f"${sales:,.0f}"); st.markdown("---"); st.header("Allocation'}, text_auto='.2s'); fig_brand.update_traces(textposition='outside'); viz_cols[0].plotly_chart(fig_brand, use_container_width=True)
                platform_summary = filtered_df Visualizations"); viz_cols = st.columns(2); brand_summary = filtered_df.groupby('Brand')['.groupby('Platform')['Final_Allocated_Budget'].sum(); fig_platform = px.pie(platform_summary, valuesFinal_Allocated_Budget'].sum().sort_values(ascending=False); fig_brand = px.bar(='Final_Allocated_Budget', names=platform_summary.index, title="Optimized Budget by Platform", holebrand_summary, x=brand_summary.index, y='Final_Allocated_Budget', title="Optimized Budget by Brand", labels={'Final_Allocated_Budget': 'Budget ($)', 'index': 'Brand'},=.3); viz_cols[1].plotly_chart(fig_platform, use_container_width=True)
                st.markdown("---"); st.header("💡 Key AI Insights");
                if not filtered_df.empty and filtered_df['Direct Sales'].sum() > 0:
                    total_sales = filtered_df['Direct Sales'].sum(); total_new_budget = filtered_df['Final_Allocated_Budget'].sum(); insight_df text_auto='.2s'); fig_brand.update_traces(textposition='outside'); viz_cols[0 = filtered_df.groupby(['Brand', 'Platform']).agg(Historical_Sales=('Direct Sales', 'sum'), Allocated_Budget=('Final_Allocated_Budget', 'sum')).reset_index(); insight_df['Sales_Share'] = insight_df['Historical_Sales'] / total_sales; insight_df['Budget_Share'] = insight_df].plotly_chart(fig_brand, use_container_width=True)
                platform_summary = filtered_df.groupby('Platform')['Final_Allocated_Budget'].sum(); fig_platform = px.pie(platform_summary, values='Final_Allocated_Budget', names=platform_summary.index, title="Optimized['Allocated_Budget'] / total_new_budget; insight_df['Lift'] = insight_df['Budget_Share'] / (insight_df['Sales_Share'] + 1e-9); hidden_gem = insight Budget by Platform", hole=.3); viz_cols[1].plotly_chart(fig_platform, use_container_df[insight_df['Allocated_Budget'] > 0].nlargest(1, 'Lift'); overpriced_performer = insight_df[insight_df['Historical_Sales'] > 0].nsmallest(1, 'Lift')
                    if roas_w >= 0.7: strategy = "to **maximize short_width=True)
                st.markdown("---"); st.header("💡 Key AI Insights");
                if not filtered_df.empty and filtered_df['Direct Sales'].sum() > 0:
                    total_sales = filtered_df['Direct Sales'].sum(); total_new_budget = filtered_df['Final_Allocated_Budget'].sum(); insight_df = filtered_df.groupby(['Brand', 'Platform']).agg(Historical_Sales=('Direct Sales', 'sum'), Allocated_Budget=('Final_Allocated_Budget', 'sum')).reset_index(); insight_df-term profitability** by focusing on segments with the highest proven ROAS."
                    elif roas_w <= 0.3: strategy = "for **aggressive customer acquisition** by prioritizing segments with high New-to-Brand (NTB)['Sales_Share'] = insight_df['Historical_Sales'] / total_sales; insight_df['Budget_Share'] = insight_df['Allocated_Budget'] / total_new_budget; insight_df['Lift'] = insight_df['Budget_Share'] / (insight_df['Sales_Share'] + 1e-9)
                    hidden_ percentages."
                    else: strategy = "for **balanced growth**, giving equal importance to both profitability (ROAS) and customergem = insight_df[insight_df['Allocated_Budget'] > 0].nlargest(1, acquisition (NTB)."
                    st.markdown(f"- **Strategy Focus:** The current weights configure the AI { 'Lift'); overpriced_performer = insight_df[insight_df['Historical_Sales'] > 0].nsstrategy}")
                    if not hidden_gem.empty: gem_row = hidden_gem.iloc[0]; st.mallest(1, 'Lift')
                    if roas_w >= 0.7: strategy = "tomarkdown(f"- **Hidden Gem:** The model identified **{gem_row['Brand']} on {gem_row[' **maximize profitability**."
                    elif roas_w <= 0.3: strategy = "for **aggressive acquisition**."
                    else: strategy = "for **balanced growth**."
                    st.markdown(f"- **StrategyPlatform']}** as a key growth opportunity. It received **{gem_row['Budget_Share']:.1%} Focus:** The AI is configured {strategy}")
                    if not hidden_gem.empty: gem_row = hidden_** of the budget, a significant increase from its historical sales contribution of {gem_row['Sales_Share']:.1%gem.iloc[0]; st.markdown(f"- **Hidden Gem:** The model identified **{gem_row['Brand}.")
                    if not overpriced_performer.empty: op_row = overpriced_performer.iloc[0];']} on {gem_row['Platform']}** as a key growth opportunity.")
                    if not overpriced_performer.empty: op_row = overpriced_performer.iloc[0]; st.markdown(f"- **Efficiency Optimization:** The model st.markdown(f"- **Efficiency Optimization:** While **{op_row['Brand']} on {op_row[' suggests reallocating budget from **{op_row['Brand']} on {op_row['Platform']}** to morePlatform']}** was a strong historical performer ({op_row['Sales_Share']:.1%} of sales), the efficient segments.")
                st.markdown("---"); st.header("Operational Health Summary (Last 3 Days)") model suggests it has reached diminishing returns, allocating it a smaller budget share ({op_row['Budget_Share']:.
                if not recent_issues.empty:
                    unresolved_issues_df = recent_issues[~recent_issues.index.isin(st.session_state.get('resolved_issues', set()))]
                    if1%}).")
                st.markdown("---"); st.header("Operational Health Summary (Last 3 Days not unresolved_issues_df.empty:
                        issue_viz_cols = st.columns(2)
                        with issue_viz_cols[0]: brand_counts = unresolved_issues_df['Brand'].value_counts(); fig_)")
                if not recent_issues.empty:
                    unresolved_issues_df = recent_issues[~recent_issues.index.isin(st.session_state.get('resolved_issues', set()))]
                    ifbrand_issues = px.pie(brand_counts, values=brand_counts.values, names=brand_counts.index not unresolved_issues_df.empty:
                        issue_viz_cols = st.columns(2)
                        with issue, title="Content Issues by Brand", hole=0.4); st.plotly_chart(fig_brand_issues_viz_cols[0]:
                            brand_counts = unresolved_issues_df['Brand'].value_counts(); fig, use_container_width=True)
                        with issue_viz_cols[1]: pincode_counts = unresolved_issues_df['Pin Code'].value_counts().nlargest(10); fig_pincode_issues_brand_issues = px.pie(brand_counts, values=brand_counts.values, names=brand_counts.index, title="Content Issues by Brand", hole=0.4); st.plotly_chart(fig_brand = px.pie(pincode_counts, values=pincode_counts.values, names=p_issues, use_container_width=True)
                        with issue_viz_cols[1]:
                            incode_counts.index, title="Top 10 Pin Codes with Issues", hole=0.4); st.plotly_chart(fig_pincode_issues, use_container_width=True)
                    pincode_counts = unresolved_issues_df['Pin Code'].value_counts().nlargest(10); fig_else: st.success("✅ No unresolved content issues found in the last 3 days.")
                else: stpincode_issues = px.pie(pincode_counts, values=pincode_counts..success("✅ No content issues found in the last 3 days.")
            else: st.info("Clickvalues, names=pincode_counts.index, title="Top 10 Pin Codes with Issues", hole=0.4); st.plotly_chart(fig_pincode_issues, use_container_width=True 'Run' to generate an allocation.")
        
        with tab2:
            if 'final_df' in st.session_state: st.header("Full Allocation Details"); st.dataframe(st.session_state.final)
                    else:
                        st.success("✅ No unresolved content issues found in the last 3 days.")
_df); st.download_button("📥 Download Full Data", to_csv(st.session_state.final_df                else:
                    st.success("✅ No content issues found in the last 3 days.")
            else), "full_alloc.csv")
            else: st.info("Run an allocation to see data.")
        
: st.info("Click 'Run' to generate an allocation.")
        with tab2:
            if 'final_        with tab3:
            st.header("Action Center: Content Issue Flags"); st.markdown("Unresolved itemsdf' in st.session_state: st.header("Full Allocation Details"); st.dataframe(st.session_state from the **last 3 days**.");
            if st.button("🔄 Reset Resolved List"): st.session_state..final_df); st.download_button("📥 Download Full Data", to_csv(st.session_stateresolved_issues = set(); st.toast("Resolved list cleared."); st.rerun()
            st.metric("Unresolved Issues", unresolved_issues_count); st.markdown("---")
            if unresolved_issues_count == 0.final_df), "full_alloc.csv")
            else: st.info("Run an allocation to see data.")
        with tab3:
            st.header("Action Center: Content Issue Flags"); st.markdown: st.success("✅ All Clear!")
            else:
                unresolved_issues_df_cards = recent_issues[~recent_issues.index.isin(st.session_state.get('resolved_issues("Unresolved items from the **last 3 days**.");
            if st.button("🔄 Reset Resolved List"):', set()))]
                for index, row in unresolved_issues_df_cards.iterrows():
                    with st.session_state.resolved_issues = set(); st.toast("Resolved list cleared."); st.rerun st.container(): st.markdown('<div class="issue-card">', unsafe_allow_html=True); col1,()
            st.metric("Unresolved Issues", unresolved_issues_count); st.markdown("---")
 col2 = st.columns([3, 1]); col1.subheader(f"Brand: {row.            if unresolved_issues_count == 0: st.success("✅ All Clear!")
            else:
                unresolved_issues_df_cards = recent_issues[~recent_issues.index.isin(st.session_state.get('resolved_issues', set()))]
                for index, row in unresolved_issues_df_cardsget('Brand', 'N/A')} | SKU: {row.get('SKU', 'N/A').iterrows():
                    with st.container(): st.markdown('<div class="issue-card">', unsafe_allow}"); col1.text(f"Platform: {row.get('Platform', 'N/A')} | Pin Code:_html=True); col1, col2 = st.columns([3, 1]); col1.subheader {row.get('Pin Code', 'N/A')} | Date: {row['Date'].strftime('%Y-%m-%d')}"); col1.error(f"**Flag Type:** {row.get('Type of Flag', 'Unknown(f"Brand: {row.get('Brand', 'N/A')} | SKU: {row.get('SK')}")
                    if col2.button("✔️ Mark as Resolved", key=f"resolve_{index}", use_container_width=True): st.session_state.resolved_issues.add(index); st.rerun()U', 'N/A')}"); col1.text(f"Platform: {row.get('Platform',
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
             'N/A')} | Pin Code: {row.get('Pin Code', 'N/A')} | Date: {row['Date'].strftime('%Y-%m-%d')}"); col1.error(f"**Flagst.header("Action Center: Low Stock Alerts"); st.markdown("Displays items with **Stock <= 5** in Type:** {row.get('Type of Flag', 'Unknown')}")
                    if col2.button("✔️ the **last 30 minutes**.")
            if st.button("🔄 Reset Low Stock List"): st.session Mark as Resolved", key=f"resolve_{index}", use_container_width=True): st.session__state.resolved_oos = set(); st.toast("Resolved list cleared."); st.rerun()
            st.metric("Actionable Low Stock Alerts", unresolved_oos_count); st.markdown("---")
            state.resolved_issues.add(index); st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        with tab4:
            st.header("Action Center: Low Stock Alerts"); stif unresolved_oos_count == 0:
                st.success("✅ No recent low stock issues found.").markdown("Displays items with **Stock <= 5** in the **last 30 minutes**.")
            if st
            else:
                unresolved_oos_df_cards = recent_oos[~recent_oos.index.isin(st.session_state.get('resolved_oos', set()))]
                oos_with_managers.button("🔄 Reset Low Stock List"): st.session_state.resolved_oos = set(); st.toast = pd.merge(unresolved_oos_df_cards, manager_df, on='Pin Code', how='left'); oos_with_managers['contact'].fillna('Not Available', inplace=True)
                for index,("Resolved list cleared."); st.rerun()
            st.metric("Actionable Low Stock Alerts", unresolved_oos_count row in oos_with_managers.iterrows():
                    with st.container():
                        st.markdown('<div class="); st.markdown("---")
            if unresolved_oos_count == 0:
                st.success("✅ No recent low stock issues found.")
            else:
                unresolved_oos_df_cards = recent_oosissue-card" style="border-color: #fca130; border-left-color: #[~recent_oos.index.isin(st.session_state.get('resolved_oos', set()))fca130; background-color: #fffaf0;">', unsafe_allow_html=True); col1, col2 = st.columns([3, 1])
                        with col1: st.subheader(f"Brand]
                oos_with_managers = pd.merge(unresolved_oos_df_cards, manager_df, on='Pin Code', how='left'); oos_with_managers['contact'].fillna('Not Available', inplace=True)
                for index, row in oos_with_managers.iterrows():
                    with st.container():
                        st.markdown('<div class="issue-card" style="border-color: #fca130;: {row.get('Brand', 'N/A')} | SKU: {row.get('SKU', 'N/A')}"); st.text(f"Pin Code: {row.get('Pin Code', 'N border-left-color: #fca130; background-color: #fffaf0;">', unsafe_allow_html=True); col1, col2 = st.columns([3, 1])
                        with col/A')} | Manager: {row.get('contact', 'N/A')}"); st.warning(f"**Stock Left:** {row.get('Stock_Left', 0)} | **Time:** {row['1: st.subheader(f"Brand: {row.get('Brand', 'N...
