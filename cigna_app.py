
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import io


st.set_page_config(
    page_title="CIGNA MA Rate Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_top_codes():
    try:
        summary = pd.read_csv('cigna_rate_summary.csv', index_col=0)
        top_codes = summary.head(100).index.tolist()
        return top_codes, summary
    except FileNotFoundError:
        st.error("❌ Summary file not found.")
        return [], None

@st.cache_data
def load_all_descriptions():
    """Load unique descriptions for search"""
    try:
        chunks = pd.read_csv('cigna_full_rates.csv', chunksize=50000)
        descriptions = set()
        for chunk in chunks:
            desc = chunk['description'].dropna().unique()
            descriptions.update(desc)
        return sorted(list(descriptions))
    except Exception as e:
        st.error(f"Error loading descriptions: {e}")
        return []

def get_code_info(billing_code):
    try:
        chunks = pd.read_csv('cigna_full_rates.csv', chunksize=50000)
        matching_rows = []
        for chunk in chunks:
            mask = chunk['billing_code'] == billing_code
            if mask.any():
                matching_rows.append(chunk[mask])
        if matching_rows:
            return pd.concat(matching_rows, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def find_code_by_description(description):
    """Find billing code that matches a description"""
    try:
        chunks = pd.read_csv('cigna_full_rates.csv', chunksize=50000)
        for chunk in chunks:
            mask = chunk['description'] == description
            if mask.any():
                return chunk[mask].iloc[0]['billing_code']
        return None
    except:
        return None

def get_best_worst_providers(data, billing_code):
    """Get cheapest and most expensive providers for a code"""
    subset = data[data['billing_code'] == billing_code]
    if subset.empty:
        return None, None
    
    provider_avg = subset.groupby('provider_name')['negotiated_rate'].mean().sort_values()
    cheapest = provider_avg.head(3).to_dict()
    most_expensive = provider_avg.tail(3).to_dict()
    return cheapest, most_expensive

def plot_compare_codes(data1, data2, code1, code2):
    """Compare two billing codes side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, data, code in [(ax1, data1, code1), (ax2, data2, code2)]:
        if data.empty:
            ax.text(0.5, 0.5, f"No data for {code}", ha='center', va='center')
            ax.set_title(code)
            continue
        
        prices = data['negotiated_rate'].dropna()
        if len(prices) == 0:
            ax.text(0.5, 0.5, "No price data", ha='center', va='center')
            ax.set_title(code)
            continue
        
        sns.violinplot(y=prices, ax=ax, inner='box', color='#99c2ff', cut=0)
        ax.axhline(prices.mean(), color='red', linestyle='--', label=f'Mean: ${prices.mean():,.0f}')
        ax.axhline(prices.median(), color='green', linestyle=':', label=f'Median: ${prices.median():,.0f}')
        ax.set_title(f'{code}\n({len(prices)} rates)')
        ax.set_ylabel('Negotiated Rate ($)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_price_distribution(data, billing_code):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if data.empty:
        ax.text(0.5, 0.5, f"No data for code: {billing_code}", ha='center', va='center')
        ax.set_title(f'{billing_code} - Not Found')
        return fig
    
    prices = data['negotiated_rate'].dropna()
    if len(prices) == 0:
        ax.text(0.5, 0.5, "No price data available", ha='center', va='center')
        ax.set_title(f'{billing_code} - No Data')
        return fig
    
    sns.violinplot(y=prices, ax=ax, inner='box', color='#99c2ff', cut=0)
    mean_price = prices.mean()
    median_price = prices.median()
    min_price = prices.min()
    max_price = prices.max()
    
    ax.axhline(mean_price, color='red', linestyle='--', label=f'Mean: ${mean_price:,.0f}')
    ax.axhline(median_price, color='green', linestyle=':', label=f'Median: ${median_price:,.0f}')
    ax.set_title(f'{billing_code} - Price Distribution\n({len(prices)} rates, {data["provider_name"].nunique()} providers)', fontsize=12)
    ax.set_ylabel('Negotiated Rate ($)')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper right')
    
    stats_text = f'Min: ${min_price:,.0f}\nMax: ${max_price:,.0f}\nMean: ${mean_price:,.0f}\nMedian: ${median_price:,.0f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    return fig


def main():
    st.title("🏥 CIGNA MA - In-Network Rates Dashboard")
    st.markdown("---")
    
    top_codes, summary = load_top_codes()
    if not top_codes:
        st.warning("⚠️ Please run the data extraction script first.")
        return
    
    
    with st.sidebar:
        st.header("📊 Statistics")
        if summary is not None:
            st.metric("Total Rates", f"{summary['count'].sum():,}")
            st.metric("Unique Billing Codes", f"{len(summary):,}")
            st.metric("Avg Rate", f"${summary['mean_rate'].mean():,.0f}")
    
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Compare", "🏆 Top Codes", "💡 Best/Worst Providers"])
    
    
    with tab1:
        st.subheader("🔍 Search by Billing Code or Description")
        
        search_type = st.radio("Search by:", ["Billing Code", "Description"], horizontal=True)
        
        if search_type == "Billing Code":
            search_code = st.selectbox("Select billing code", options=[''] + top_codes[:50])
            
            if search_code:
                with st.spinner(f"Loading {search_code}..."):
                    subset = get_code_info(search_code)
                
                if not subset.empty:
                    st.success(f"✅ Found {len(subset)} records")
                    
                    prices = subset['negotiated_rate'].dropna()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Count", f"{len(prices):,}")
                    col2.metric("Mean", f"${prices.mean():,.0f}")
                    col3.metric("Median", f"${prices.median():,.0f}")
                    col4.metric("Std", f"${prices.std():,.0f}")
                    
                    st.pyplot(plot_price_distribution(subset, search_code))
                    
                    with st.expander("📊 Sample Data"):
                        st.dataframe(subset[['provider_name', 'negotiated_rate', 'negotiated_type']].head(100))
                else:
                    st.warning(f"No data for {search_code}")
        
        else:
            st.info("💡 Search by description to find the billing code")
            descriptions = load_all_descriptions()
            if descriptions:
                search_desc = st.selectbox("Select description", options=[''] + descriptions[:100])
                if search_desc:
                    code = find_code_by_description(search_desc)
                    if code:
                        st.success(f"✅ Found code: **{code}**")
                        with st.spinner(f"Loading {code}..."):
                            subset = get_code_info(code)
                        if not subset.empty:
                            st.pyplot(plot_price_distribution(subset, code))
                    else:
                        st.warning("No matching code found")
    
    
    with tab2:
        st.subheader("📊 Compare Two Billing Codes")
        
        col1, col2 = st.columns(2)
        with col1:
            code1 = st.selectbox("First code", options=top_codes[:30], key='compare1')
        with col2:
            code2 = st.selectbox("Second code", options=top_codes[:30], key='compare2')
        
        if code1 and code2:
            with st.spinner("Loading comparison data..."):
                data1 = get_code_info(code1)
                data2 = get_code_info(code2)
            
            if not data1.empty and not data2.empty:
                st.pyplot(plot_compare_codes(data1, data2, code1, code2))
                
                # Comparison stats
                st.subheader("📈 Comparison Statistics")
                stats1 = data1['negotiated_rate'].dropna()
                stats2 = data2['negotiated_rate'].dropna()
                
                comp_df = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Min', 'Max'],
                    code1: [len(stats1), f"${stats1.mean():,.0f}", f"${stats1.median():,.0f}", f"${stats1.min():,.0f}", f"${stats1.max():,.0f}"],
                    code2: [len(stats2), f"${stats2.mean():,.0f}", f"${stats2.median():,.0f}", f"${stats2.min():,.0f}", f"${stats2.max():,.0f}"]
                })
                st.dataframe(comp_df, use_container_width=True)
            else:
                st.warning("Not enough data for comparison")
    
    
    with tab3:
        st.subheader("🏆 Top Billing Codes by Frequency")
        top_n = st.slider("Number to display", 5, 50, 20)
        
        if summary is not None:
            st.dataframe(
                summary.head(top_n).reset_index()[['billing_code', 'count', 'mean_rate', 'min_rate', 'max_rate']],
                column_config={
                    'billing_code': 'Billing Code',
                    'count': st.column_config.NumberColumn('Count', format='%d'),
                    'mean_rate': st.column_config.NumberColumn('Mean Rate', format='$%.0f'),
                },
                use_container_width=True
            )
    
    
    with tab4:
        st.subheader("💡 Cheapest & Most Expensive Providers")
        
        selected_code = st.selectbox("Select billing code", options=top_codes[:30], key='bw')
        
        if selected_code:
            with st.spinner("Analyzing providers..."):
                cheapest, expensive = get_best_worst_providers(get_code_info(selected_code), selected_code)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 💚 Cheapest Providers")
                if cheapest:
                    for provider, price in cheapest.items():
                        st.metric(provider[:30], f"${price:,.0f}")
                else:
                    st.info("No data")
            
            with col2:
                st.markdown("### 🔥 Most Expensive Providers")
                if expensive:
                    for provider, price in expensive.items():
                        st.metric(provider[:30], f"${price:,.0f}")
                else:
                    st.info("No data")

if __name__ == "__main__":
    main()