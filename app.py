# app.py (v41 - FINAL, FEATURE-COMPLETE & OPTIMIZED)
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px
import gc
import os
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from passlib.context import CryptContext
import smtplib
from email.message import EmailMessage

# --- Page Configuration & State ---
st.set_page_config(layout="wide", page_title="ITC AI Budget Allocation Portal")
if 'resolved_issues' not in st.session_state: st.session_state.resolved_issues = set()
if 'resolved_oos' not in st.session_state: st.session_state.resolved_oos = set()

# --- Hashing Context ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ----------------- The Backend "Engine" -----------------
# SPEED OPTIMIZATION: Caching is used for all data loading and processing functions.
@st.cache_data
def load_data(input_url):
    df = pd.read_csv(input_url); df.columns = [c.strip() for c in df.columns]; return df

@st.cache_data
def load_demo_data(input_file):
    xls = pd.ExcelFile(input_file); df_s1 = pd.read_excel(xls, sheet_name='s1'); df_s2 = pd.read_excel(xls, sheet_name='s2')
    if 'Pincode' in df_s1.columns: df_s1.rename(columns={'Pincode': 'Pin Code'}, inplace=True)
    if 'Pincode' in df_s2.columns: df_s2.rename(columns={'Pincode': 'Pin Code'}, inplace=True)
    if 'Pin Code' in df_s1.columns: df_s1['Pin Code'] = df_s1['Pin Code'].astype(str).str.strip()
    if 'Pin Code' in df_s2.columns: df_s2['Pin Code'] = df_s2['Pin Code'].astype(str).str.strip()
    return df_s1, df_s2

@st.cache_data
def calculate_allocation(_df, budget_multiplier, roas_weight):
    # The underscore in _df is a convention to show it's being modified by a cached function
    df = _df.copy() # Work on a copy to prevent caching issues
    ntb_weight = 1.0 - roas_weight; grouping_keys = ["Time Slot", "Pin Code", "Tier", "Platform", "Brand", "SKU", "Ad Type", "OOS Flag", "Content Issue Flag"]; grouping_keys = [key for key in grouping_keys if key in df.columns]
    if pd.api.types.is_string_dtype(df["NTB (%)"]): df['Clean_NTB'] = pd.to_numeric(df["NTB (%)"].str.replace('%', '', regex=False), errors='coerce')
    else: df['Clean_NTB'] = pd.to_numeric(df["NTB (%)"], errors='coerce')
    agg_dict = {'Budget Spent': 'sum', 'Direct Sales': 'sum', 'Clean_NTB': 'mean'}; df_agg = df.groupby(grouping_keys, as_index=False).agg(agg_dict); df_agg['Aggregated_ROAS'] = df_agg['Direct Sales'] / (df_agg['Budget Spent'] + 1e-6); df_agg['Optimization_Score'] = (roas_weight * df_agg['Aggregated_ROAS']) + (ntb_weight * df_agg['Clean_NTB']); df_agg.fillna(0, inplace=True)
    features = grouping_keys; X = df_agg[features]; y = df_agg['Optimization_Score']
    for col in features: X[col] = X[col].astype("category")
    lgb_data = lgb.Dataset(X, label=y, categorical_feature=features); params = {"objective": "regression_l1", "metric": "rmse", "verbosity": -1, "seed": 42}; model = lgb.train(params, lgb_data, num_boost_round=200); df_agg['Predicted_Score'] = model.predict(X).clip(min=0)
    current_total_budget = df_agg['Budget Spent'].sum(); new_total_budget = current_total_budget * budget_multiplier; brand_sales = df_agg.groupby('Brand')['Direct Sales'].sum(); brand_proportions = brand_sales / max(1, brand_sales.sum()); brand_budgets = brand_proportions * new_total_budget
    df_agg['Brand_Allocated_Budget'] = df_agg['Brand'].map(brand_budgets); df_agg['Brand_Total_Predicted_Score'] = df_agg.groupby('Brand')['Predicted_Score'].transform('sum')
    df_agg['Final_Allocated_Budget'] = df_agg['Brand_Allocated_Budget'] * (df_agg['Predicted_Score'] / (df_agg['Brand_Total_Predicted_Score'] + 1e-6)); df_agg.fillna(0, inplace=True); df_agg['Final_Allocated_Budget'] = df_agg['Final_Allocated_Budget'] * (new_total_budget / max(1, df_agg['Final_Allocated_Budget'].sum()))
    return df_agg

def send_oos_email(manager_email, brand, sku, pincode, stock_left):
    try:
        sender = st.secrets["email_credentials"]["sender_email"]; password = st.secrets["email_credentials"]["sender_password"]
        subject = f"🚨 URGENT: Low Stock Alert for {brand}"; body = f"Hello,\n\nThis is an automated alert.\n\nThe following item is running low on stock in your area:\n\n- Brand: {brand}\n- SKU: {sku}\n- Pin Code: {pincode}\n- Stock Left: {stock_left}\n\nPlease take action to restock.\n\nThank you,\nITC AI Operations Bot"
        msg = EmailMessage(); msg.set_content(body); msg['Subject'] = subject; msg['From'] = sender; msg['To'] = manager_email
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465); server.login(sender, password); server.send_message(msg); server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}"); return False

@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Authentication and Main App UI ---
def check_password(username, password):
    if username in st.secrets["passwords"]: return pwd_context.verify(password, st.secrets["passwords"][username])
    return False

def login_form():
    col1, col2, col3 = st.columns([1,1,1]);
    with col2:
        if os.path.exists("itc_logo.png"): st.image("itc_logo.png", width=150)
        st.title("ITC Allocation Portal Login")
        with st.form("login_form"):
            username = st.text_input("Username"); password = st.text_input("Password", type="password"); submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if check_password(username, password): st.session_state["authentication_status"] = True; st.session_state["username"] = username; st.rerun()
                else: st.error("Incorrect username or password")
    return False

if not st.session_state.get("authentication_status", False):
    login_form()
else:
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>ITC AI-BASED BUDGET ALLOCATION PORTAL</h1>", unsafe_allow_html=True)
    st.sidebar.success(f"Welcome, {st.session_state['username']}!")
    st.sidebar.header("⚙️ Scenario Controls")
    try:
        INPUT_DATA_URL = "https://docs.google.com/spreadsheets/d/1g1F863VgDK0QOR0rnAOm3pEF0QJvyg-U/export?format=csv&gid=0"; DEMO_XLSX = "demo.xlsx"
        original_df = load_data(INPUT_DATA_URL); oos_df, manager_df = load_demo_data(DEMO_XLSX)
        
        # <<< --- NEW FEATURE: Campaign Objective Selector --- >>>
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Campaign Objective")
        objective = st.sidebar.selectbox("Select the primary goal for this allocation:",("Balanced Growth (50% ROAS, 50% NTB)","Maximize Profitability (80% ROAS, 20% NTB)","Aggressive Acquisition (20% ROAS, 80% NTB)"))
        if "Balanced" in objective: roas_w = 0.50
        elif "Profitability" in objective: roas_w = 0.80
        elif "Acquisition" in objective: roas_w = 0.20
        st.sidebar.info(f"The model will now optimize for **{objective}**.")
        
        budget_mult = st.sidebar.slider("Budget Multiplier", 0.5, 2.5, 1.2, 0.1)
        
        if st.sidebar.button("🚀 Run Predictive Allocation", type="primary", use_container_width=True):
            with st.spinner("🧠 Running predictive model..."): st.session_state.final_df = calculate_allocation(original_df, budget_mult, roas_w)
            st.toast("✅ Allocation complete!", icon="🎉")
        if st.sidebar.button("Logout"): st.session_state["authentication_status"] = False; st.session_state["username"] = None; st.rerun()
        
        # --- Pre-calculation for Tab Badges ---
        unresolved_issues_count = 0; unresolved_oos_count = 0
        if all(col in original_df.columns for col in ['Date', 'Content Issue Flag']): df_copy = original_df.copy(); df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce'); df_copy.dropna(subset=['Date'], inplace=True); end_date = df_copy['Date'].max(); start_date = end_date - timedelta(days=2); recent_issues = df_copy[(df_copy['Date'] >= start_date) & (df_copy['Content Issue Flag'].astype(str).str.lower() == 'yes')]; unresolved_issues_count = len(recent_issues[~recent_issues.index.isin(st.session_state.get('resolved_issues', set()))])
        if all(col in oos_df.columns for col in ['Time', 'Stock_Left']): df_oos_copy = oos_df.copy(); df_oos_copy['Parsed_Timestamp'] = pd.to_datetime(df_oos_copy['Time'].astype(str), errors='coerce'); df_oos_copy['Simulated_Timestamp'] = df_oos_copy['Parsed_Timestamp'].dt.time.apply(lambda t: datetime.combine(datetime.now().date(), t) if pd.notna(t) else pd.NaT); time_filter = datetime.now() - timedelta(minutes=30); recent_oos = df_oos_copy[(df_oos_copy['Simulated_Timestamp'] >= time_filter) & (df_oos_copy['Stock_Left'] <= 5)]; unresolved_oos_count = len(recent_oos[~recent_oos.index.isin(st.session_state.get('resolved_oos', set()))])
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Predictive Allocation", "📋 Raw Data", f"🚨 Content Issues ({unresolved_issues_count})", f"📉 Low Stock Alerts ({unresolved_oos_count})", "🌐 ITC E-COMMERCE"])
        
        with tab1:
            if 'final_df' in st.session_state:
                final_df = st.session_state.final_df
                # <<< --- NEW FEATURE: Dayparting Filter --- >>>
                with st.expander("🔍 Filter Dashboard Results", expanded=True):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    brands = sorted(final_df['Brand'].unique()); selected_brands = col1.multiselect("Brand", brands, default=brands)
                    platforms = sorted(final_df['Platform'].unique()); selected_platforms = col2.multiselect("Platform", platforms, default=platforms)
                    ad_types = sorted(final_df['Ad Type'].unique()); selected_ad_types = col3.multiselect("Ad Type", ad_types, default=ad_types)
                    tiers = sorted(final_df['Tier'].unique()); selected_tiers = col4.multiselect("Tier", tiers, default=tiers)
                    time_slots = sorted(final_df['Time Slot'].unique()); selected_slots = col5.multiselect("Time Slot", time_slots, default=time_slots)

                filtered_df = final_df[(final_df['Brand'].isin(selected_brands)) & (final_df['Platform'].isin(selected_platforms)) & (final_df['Ad Type'].isin(selected_ad_types)) & (final_df['Tier'].isin(selected_tiers)) & (final_df['Time Slot'].isin(selected_slots))]
                
                st.header("Financial Summary"); kpi_cols = st.columns(3); original_budget = filtered_df['Budget Spent'].sum(); new_budget = filtered_df['Final_Allocated_Budget'].sum(); sales = filtered_df['Direct Sales'].sum(); kpi_cols[0].metric("Original Budget", f"${original_budget:,.0f}"); kpi_cols[1].metric("Optimized Budget", f"${new_budget:,.0f}", f"{(new_budget - original_budget):,.0f}"); kpi_cols[2].metric("Historical Sales", f"${sales:,.0f}"); st.markdown("---"); st.header("Allocation Visualizations"); viz_cols = st.columns(2)
                brand_summary = filtered_df.groupby('Brand')['Final_Allocated_Budget'].sum().sort_values(ascending=False); fig_brand = px.bar(brand_summary, x=brand_summary.index, y='Final_Allocated_Budget', title="Optimized Budget by Brand", labels={'Final_Allocated_Budget': 'Budget ($)', 'index': 'Brand'}, text_auto='.2s'); fig_brand.update_traces(textposition='outside'); viz_cols[0].plotly_chart(fig_brand, use_container_width=True)
                platform_summary = filtered_df.groupby('Platform')['Final_Allocated_Budget'].sum(); fig_platform = px.pie(platform_summary, values='Final_Allocated_Budget', names=platform_summary.index, title="Optimized Budget by Platform", hole=.3); viz_cols[1].plotly_chart(fig_platform, use_container_width=True)
                
                # <<< --- NEW FEATURE: AI Key Insights --- >>>
                st.markdown("---"); st.header("💡 Key Insights from the AI Model")
                top_brand = brand_summary.index[0]; top_platform = platform_summary.index[0]
                st.markdown(f"""Based on the **{objective}** objective, the model recommends these strategic shifts:
                - **Prioritize Brand:** The largest portion of the new budget (`${brand_summary.iloc[0]:,.0f}`) is allocated to the **{top_brand}** brand.
                - **Focus Platform:** The **{top_platform}** platform receives the largest share of spend, indicating it is the most efficient channel based on historical data.
                """)
            else: st.info("Click 'Run' to generate an allocation.")
        
        with tab2:
            if 'final_df' in st.session_state: st.header("Full Allocation Details"); st.dataframe(st.session_state.final_df); st.download_button("📥 Download Full Data", to_csv(st.session_state.final_df), "full_alloc.csv")
            else: st.info("Run an allocation to see data.")
        
        with tab3:
            st.header("Action Center: Content Issue Flags"); st.markdown("Unresolved items from the **last 3 days**.");
            if st.button("🔄 Reset Resolved List"): st.session_state.resolved_issues = set(); st.toast("Resolved list cleared."); st.rerun()
            st.metric("Unresolved Issues", unresolved_issues_count); st.markdown("---")
            if unresolved_issues_count == 0: st.success("✅ All Clear!")
            else:
                unresolved_issues_df = recent_issues[~recent_issues.index.isin(st.session_state.get('resolved_issues', set()))]
                for index, row in unresolved_issues_df.iterrows():
                    with st.container(): st.markdown('<div class="issue-card">', unsafe_allow_html=True); col1, col2 = st.columns([3, 1]); col1.subheader(f"Brand: {row.get('Brand', 'N/A')} | SKU: {row.get('SKU', 'N/A')}"); col1.text(f"Platform: {row.get('Platform', 'N/A')} | Pin Code: {row.get('Pin Code', 'N/A')} | Date: {row['Date'].strftime('%Y-%m-%d')}"); col1.error(f"**Flag Type:** {row.get('Type of Flag', 'Unknown')}")
                    if col2.button("✔️ Mark as Resolved", key=f"resolve_{index}", use_container_width=True): st.session_state.resolved_issues.add(index); st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.header("Action Center: Low Stock Alerts"); st.markdown("Displays items with **Stock <= 5** in the **last 30 minutes**.")
            if st.button("🔄 Reset Low Stock List"): st.session_state.resolved_oos = set(); st.toast("Resolved list cleared."); st.rerun()
            st.metric("Actionable Low Stock Alerts", unresolved_oos_count); st.markdown("---")
            if unresolved_oos_count == 0: st.success("✅ No recent low stock issues found.")
            else:
                unresolved_oos_df = recent_oos[~recent_oos.index.isin(st.session_state.get('resolved_oos', set()))]
                oos_with_managers = pd.merge(unresolved_oos_df, manager_df, on='Pin Code', how='left'); oos_with_managers['contact'].fillna('Not Available', inplace=True)
                for index, row in oos_with_managers.iterrows():
                    with st.container():
                        st.markdown('<div class="issue-card" style="border-color: #fca130; border-left-color: #fca130; background-color: #fffaf0;">', unsafe_allow_html=True); col1, col2 = st.columns([3, 1])
                        with col1: st.subheader(f"Brand: {row.get('Brand', 'N/A')} | SKU: {row.get('SKU', 'N/A')}"); st.text(f"Pin Code: {row.get('Pin Code', 'N/A')} | Manager: {row.get('contact', 'N/A')}"); st.warning(f"**Stock Left:** {row.get('Stock_Left', 0)} | **Time:** {row['Parsed_Timestamp'].time().strftime('%I:%M %p')}")
                        with col2:
                            if st.button("📧 Notify Manager", key=f"notify_{index}", use_container_width=True):
                                if row['contact'] != 'Not Available':
                                    if send_oos_email(row['contact'], row['Brand'], row['SKU'], row['Pin Code'], row['Stock_Left']): st.toast(f"✅ Email sent to {row['contact']}!"); st.session_state.resolved_oos.add(row.name); st.rerun()
                                else: st.warning("No manager email available.")
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.header("Open the Live E-Commerce Dashboard"); POWER_BI_URL = "https://app.powerbi.com/groups/me/reports/4d9f2e70-e22d-464c-a997-355c8559558e/4f5955ee3b04ded7b3da?experience=power-bi"
            st.markdown(f'<a href="{POWER_BI_URL}" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: #1a73e8; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 5px;">🔗 Open Secure Power BI Report</a>', unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Please make sure 'demo.xlsx' is in the same folder as the app and the Google Sheet link is correct.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
