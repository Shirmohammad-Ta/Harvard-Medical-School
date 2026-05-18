"""
CIGNA MA - In-Network Rates Dashboard
With MGB Price Highlight on Charts
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import io

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CIGNA MA Rate Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD DATA (CACHED)
# ============================================

@st.cache_data
def load_data():
    """Load the pre-processed CSV file with provider_group_id"""
    try:
        df = pd.read_csv('cigna_full_rates_with_groupid.csv')
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('cigna_full_rates.csv')
            return df
        except FileNotFoundError:
            st.error("❌ Data file not found. Please run the analysis first.")
            return None

@st.cache_data
def load_summary():
    """Load summary statistics"""
    try:
        summary = pd.read_csv('cigna_rate_summary.csv', index_col=0)
        return summary
    except FileNotFoundError:
        return None

@st.cache_data
def load_mgb_prices():
    """Load MGB prices from the dataset"""
    try:
        df = pd.read_csv('cigna_full_rates_with_groupid.csv')
        mgb_df = df[df['provider_group_id'] == 1072514]
        mgb_prices = mgb_df.groupby('billing_code')['negotiated_rate'].first().to_dict()
        return mgb_prices
    except Exception as e:
        return {}

# ============================================
# HELPER FUNCTIONS
# ============================================

def dollar_formatter(x, pos):
    return f'${x:,.0f}'

def plot_price_distribution_with_mgb(data, billing_code, mgb_price=None, title=None):
    """Create violin plot for a billing code with MGB price highlighted"""
    subset = data[data['billing_code'] == billing_code]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if subset.empty:
        ax.text(0.5, 0.5, f"No data for code: {billing_code}", 
                ha='center', va='center', fontsize=14)
        ax.set_title(f'{billing_code} - Not Found')
        return fig
    
    prices = subset['negotiated_rate'].dropna()
    
    if len(prices) == 0:
        ax.text(0.5, 0.5, "No price data available", ha='center', va='center')
        ax.set_title(f'{billing_code} - No Data')
        return fig
    
    # Violin plot
    sns.violinplot(y=prices, ax=ax, inner='box', color='#99c2ff', cut=0)
    
    # Statistics
    mean_price = prices.mean()
    median_price = prices.median()
    min_price = prices.min()
    max_price = prices.max()
    
    # Add lines
    ax.axhline(mean_price, color='red', linestyle='--', linewidth=1.5, label=f'Mean: ${mean_price:,.0f}')
    ax.axhline(median_price, color='green', linestyle=':', linewidth=1.5, label=f'Median: ${median_price:,.0f}')
    
    # 🔥 MGB price point (الالماس نارنجی)
    if mgb_price and mgb_price > 0:
        ax.scatter(0, mgb_price, color='orange', s=200, marker='D', 
                   edgecolor='black', linewidth=2, zorder=5,
                   label=f'MGB Price: ${mgb_price:,.0f}')
        ax.text(0.1, mgb_price, f'  MGB: ${mgb_price:,.0f}', 
                verticalalignment='center', fontsize=10, fontweight='bold')
    
    # Title and labels
    display_title = title or f'{billing_code} - Price Distribution'
    ax.set_title(f'{display_title}\n({len(prices)} rates from {subset["provider_name"].nunique()} providers)', fontsize=12)
    ax.set_ylabel('Negotiated Rate ($)')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    ax.legend(loc='upper right')
    
    # Add stats box
    stats_text = f'Min: ${min_price:,.0f}\nMax: ${max_price:,.0f}\nMean: ${mean_price:,.0f}\nMedian: ${median_price:,.0f}'
    if mgb_price:
        stats_text += f'\n🔸 MGB: ${mgb_price:,.0f}'
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🏥 CIGNA MA - In-Network Rates Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    summary = load_summary()
    mgb_prices = load_mgb_prices()
    
    if df is None:
        st.warning("⚠️ Please run the data extraction script first.")
        
        # Option to upload file directly
        st.subheader("📤 Or upload your CSV file:")
        uploaded_file = st.file_uploader("Upload cigna_full_rates_with_groupid.csv", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File loaded successfully!")
        else:
            return
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Filters")
        
        # Filter by billing code type
        if 'billing_code_type' in df.columns:
            code_types = ['All'] + sorted(df['billing_code_type'].dropna().unique().tolist())
            selected_type = st.selectbox("Billing Code Type", code_types)
        else:
            selected_type = 'All'
        
        st.markdown("---")
        st.header("📊 Statistics")
        st.metric("Total Providers", f"{df['provider_name'].nunique():,}")
        st.metric("Total Rates", f"{len(df):,}")
        st.metric("Unique Billing Codes", f"{df['billing_code'].nunique():,}")
        st.metric("MGB Codes Found", f"{len(mgb_prices):,}")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_type != 'All' and 'billing_code_type' in df.columns:
        filtered_df = filtered_df[filtered_df['billing_code_type'] == selected_type]
    
    st.subheader(f"📋 Current Data: {len(filtered_df):,} records")
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search by Code", "🏆 Top Codes", "🏥 Provider Analysis", "📥 Export Data"])
    
    # ============================================
    # TAB 1: Search by Code
    # ============================================
    with tab1:
        st.subheader("🔍 Search for a Specific Billing Code")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get top codes for suggestions
            top_codes = df['billing_code'].value_counts().head(50).index.tolist()
            
            # Manual code input
            col_a, col_b = st.columns([2, 1])
            with col_a:
                quick_select = st.selectbox("Quick select (top 50 codes)", options=[''] + top_codes[:30])
            with col_b:
                manual_code = st.text_input("Or type any code", placeholder="Example: 76805")
            
            search_code = manual_code.strip() if manual_code else quick_select
            
            if search_code and search_code != '':
                with st.spinner(f"Loading {search_code}..."):
                    subset = filtered_df[filtered_df['billing_code'] == search_code]
                    mgb_price = mgb_prices.get(search_code)
                
                if not subset.empty:
                    st.success(f"✅ Found {len(subset)} records for code: {search_code}")
                    if mgb_price:
                        st.info(f"🏥 MGB price found: **${mgb_price:,.0f}**")
                    else:
                        st.warning(f"⚠️ No MGB price found for this code")
                    
                    # Price statistics
                    prices = subset['negotiated_rate'].dropna()
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Count", f"{len(prices):,}")
                    with col_b:
                        st.metric("Mean", f"${prices.mean():,.0f}")
                    with col_c:
                        st.metric("Median", f"${prices.median():,.0f}")
                    with col_d:
                        st.metric("Std Dev", f"${prices.std():,.0f}")
                    
                    # Price distribution plot with MGB
                    st.pyplot(plot_price_distribution_with_mgb(filtered_df, search_code, mgb_price, f'Code: {search_code}'))
                    
                    # Show data table
                    with st.expander("📊 View Data Table (first 100 rows)"):
                        display_cols = ['provider_name', 'negotiated_rate', 'billing_code_type', 'description']
                        if 'provider_group_id' in subset.columns:
                            display_cols.insert(1, 'provider_group_id')
                        available_cols = [c for c in display_cols if c in subset.columns]
                        st.dataframe(subset[available_cols].head(100))
                else:
                    st.error(f"❌ Code '{search_code}' not found in database")
                    st.info(f"💡 Try one of these popular codes: `{', '.join(top_codes[:10])}`")
        
        with col2:
            st.info("💡 **Top 20 Billing Codes**")
            for i, code in enumerate(top_codes[:20], 1):
                mgb_mark = " 🏥" if code in mgb_prices else ""
                st.write(f"{i}. `{code}`{mgb_mark}")
    
    # ============================================
    # TAB 2: Top Codes
    # ============================================
    with tab2:
        st.subheader("🏆 Top Billing Codes by Frequency")
        
        if summary is not None:
            top_n = st.slider("Number of top codes to display", 5, 50, 20)
            
            top_codes_data = summary.head(top_n).reset_index()
            top_codes_data['has_mgb'] = top_codes_data['billing_code'].isin(mgb_prices)
            top_codes_data['mgb_price'] = top_codes_data['billing_code'].map(mgb_prices)
            top_codes_data['mgb_display'] = top_codes_data['mgb_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
            
            # Display table
            st.dataframe(
                top_codes_data[['billing_code', 'count', 'mean_rate', 'min_rate', 'max_rate', 'mgb_display']],
                column_config={
                    'billing_code': 'Billing Code',
                    'count': st.column_config.NumberColumn('Count', format='%d'),
                    'mean_rate': st.column_config.NumberColumn('Mean Rate', format='$%.0f'),
                    'min_rate': st.column_config.NumberColumn('Min Rate', format='$%.0f'),
                    'max_rate': st.column_config.NumberColumn('Max Rate', format='$%.0f'),
                    'mgb_display': 'MGB Price'
                },
                use_container_width=True
            )
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(top_codes_data)), top_codes_data['count'], color='steelblue')
            ax.set_yticks(range(len(top_codes_data)))
            ax.set_yticklabels(top_codes_data['billing_code'])
            ax.set_xlabel('Number of negotiated rates')
            ax.set_title(f'Top {top_n} Billing Codes by Rate Count')
            
            # Highlight codes with MGB price
            for i, (idx, row) in enumerate(top_codes_data.iterrows()):
                if row['has_mgb']:
                    bars[i].set_color('orange')
            
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                        f'{int(bar.get_width())}', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("🏥 Orange bars indicate codes with MGB price available")
        else:
            st.warning("Summary data not available")
    
    # ============================================
    # TAB 3: Provider Analysis
    # ============================================
    with tab3:
        st.subheader("🏥 Provider Analysis")
        
        # Check if provider_group_id exists
        if 'provider_group_id' in df.columns:
            mgb_providers = df[df['provider_group_id'] == 1072514]['provider_name'].unique()
            st.info(f"🏥 Found {len(mgb_providers)} MGB provider records")
        
        # Get top providers
        top_providers = filtered_df['provider_name'].value_counts().head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            top_providers.head(15).plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Number of negotiated rates')
            ax.set_title('Top 15 Providers by Number of Rates')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            provider_list = ['All'] + sorted(filtered_df['provider_name'].dropna().unique().tolist())[:100]
            selected_provider = st.selectbox("Select a provider", provider_list)
            
            if selected_provider != 'All':
                provider_data = filtered_df[filtered_df['provider_name'] == selected_provider]
                st.metric("Number of Rates", len(provider_data))
                st.metric("Unique Billing Codes", provider_data['billing_code'].nunique())
                st.metric("Average Rate", f"${provider_data['negotiated_rate'].mean():,.0f}")
                
                with st.expander("View provider data"):
                    st.dataframe(provider_data[['billing_code', 'negotiated_rate', 'description']].head(50))
    
    # ============================================
    # TAB 4: Export Data
    # ============================================
    with tab4:
        st.subheader("📥 Export Data")
        
        st.write("Download the current filtered dataset as CSV")
        
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="cigna_filtered_rates.csv",
            mime="text/csv"
        )
        
        if summary is not None:
            summary_buffer = io.StringIO()
            summary.to_csv(summary_buffer)
            summary_data = summary_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Summary Statistics (CSV)",
                data=summary_data,
                file_name="cigna_rate_summary.csv",
                mime="text/csv"
            )
        
        # Export MGB prices
        if mgb_prices:
            mgb_df = pd.DataFrame(list(mgb_prices.items()), columns=['billing_code', 'mgb_price'])
            mgb_buffer = io.StringIO()
            mgb_df.to_csv(mgb_buffer, index=False)
            st.download_button(
                label="📥 Download MGB Prices (CSV)",
                data=mgb_buffer.getvalue(),
                file_name="mgb_prices.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    st.caption("📊 Data source: CIGNA MA In-Network Rates | 🏥 Orange diamond = MGB price | Last updated: 2026-05-18")

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()