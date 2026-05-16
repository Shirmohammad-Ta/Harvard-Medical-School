"""
CIGNA MA - In-Network Rates Dashboard
Enhanced with Benchmark Comparison + Anomaly Detection + PDF Export
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
import io
import base64
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


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

@st.cache_data
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
        return None, None, 0
    
    provider_avg = subset.groupby('provider_name')['negotiated_rate'].mean().sort_values()
    num_providers = len(provider_avg)
    
    if num_providers == 0:
        return None, None, 0
    elif num_providers <= 3:
        cheapest = provider_avg.head(num_providers).to_dict()
        most_expensive = {}
    elif num_providers <= 6:
        cheapest = provider_avg.head(3).to_dict()
        most_expensive = {}
        for provider, price in provider_avg.tail(3).items():
            if provider not in cheapest:
                most_expensive[provider] = price
    else:
        cheapest = provider_avg.head(3).to_dict()
        most_expensive = provider_avg.tail(3).to_dict()
    
    return cheapest, most_expensive, num_providers

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

def detect_anomalies(prices, threshold=2):
    """Detect price anomalies using standard deviation"""
    if len(prices) < 3:
        return []
    mean_price = prices.mean()
    std_price = prices.std()
    anomalies = prices[(prices > mean_price + threshold * std_price) | 
                       (prices < mean_price - threshold * std_price)]
    return anomalies.tolist()

def create_pdf_report(code, stats, anomalies, cheapest, expensive, overall_avg):
    """Create PDF report using reportlab"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=5
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4
    )
    
    story = []
    
    # Title
    story.append(Paragraph(f"CIGNA MA Rate Report: Code {code}", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Basic Statistics
    story.append(Paragraph("Basic Statistics", heading_style))
    story.append(Paragraph(f"Total Records: {stats['count']:,}", normal_style))
    story.append(Paragraph(f"Mean Price: ${stats['mean']:,.0f}", normal_style))
    story.append(Paragraph(f"Median Price: ${stats['median']:,.0f}", normal_style))
    story.append(Paragraph(f"Min Price: ${stats['min']:,.0f}", normal_style))
    story.append(Paragraph(f"Max Price: ${stats['max']:,.0f}", normal_style))
    story.append(Paragraph(f"Standard Deviation: ${stats['std']:,.0f}", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Benchmark
    story.append(Paragraph("Benchmark Analysis", heading_style))
    diff_percent = ((stats['mean'] - overall_avg) / overall_avg) * 100
    story.append(Paragraph(f"System Average: ${overall_avg:,.0f}", normal_style))
    story.append(Paragraph(f"Difference: {diff_percent:.1f}% {'above' if diff_percent > 0 else 'below'} average", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Anomalies
    if anomalies:
        story.append(Paragraph("Price Anomalies Detected", heading_style))
        for anomaly in anomalies[:10]:
            story.append(Paragraph(f"Suspicious price: ${anomaly:,.0f}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Cheapest Providers
    if cheapest:
        story.append(Paragraph("Cheapest Providers", heading_style))
        for provider, price in cheapest.items():
            provider_clean = provider.replace('&', 'and').replace('<', '').replace('>', '')
            story.append(Paragraph(f"{provider_clean[:50]}: ${price:,.0f}", normal_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Most Expensive Providers
    if expensive:
        story.append(Paragraph("Most Expensive Providers", heading_style))
        for provider, price in expensive.items():
            provider_clean = provider.replace('&', 'and').replace('<', '').replace('>', '')
            story.append(Paragraph(f"{provider_clean[:50]}: ${price:,.0f}", normal_style))
    
    # Build PDF
    doc.build(story)
    return temp_file.name


def main():
    st.title("🏥 CIGNA MA - In-Network Rates Dashboard")
    st.markdown("---")
    
    top_codes, summary = load_top_codes()
    if not top_codes:
        st.warning("⚠️ Please run the data extraction script first.")
        return
    
    overall_avg = summary['mean_rate'].mean() if summary is not None else 0
    
    with st.sidebar:
        st.header("📊 Statistics")
        if summary is not None:
            st.metric("Total Rates", f"{summary['count'].sum():,}")
            st.metric("Unique Billing Codes", f"{len(summary):,}")
            st.metric("System Avg Rate", f"${overall_avg:,.0f}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Compare", "🏆 Top Codes", "💡 Best/Worst Providers"])
    
    # ============================================
    # TAB 1: SEARCH
    # ============================================
    with tab1:
        st.subheader("🔍 Search by Billing Code or Description")
        
        search_type = st.radio("Search by:", ["Billing Code", "Description"], horizontal=True)
        
        if search_type == "Billing Code":
            st.markdown("### Enter a billing code manually")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                quick_select = st.selectbox(
                    "⚡ Quick select (top 20 codes)", 
                    options=[''] + top_codes[:20],
                    help="Choose from the most common codes"
                )
            
            with col2:
                manual_code = st.text_input(
                    "✏️ Or type any billing code manually",
                    placeholder="Example: 76805, 99213, 99214",
                    help="Enter any 5-digit CPT/HCPCS code"
                )
            
            search_code = manual_code.strip() if manual_code else quick_select
            
            if search_code and search_code != '':
                with st.spinner(f"Searching for code: {search_code}..."):
                    subset = get_code_info(search_code)
                
                if not subset.empty:
                    st.success(f"✅ Found **{len(subset)}** records for code: `{search_code}`")
                    
                    prices = subset['negotiated_rate'].dropna()
                    mean_price = prices.mean()
                    median_price = prices.median()
                    
                    current_stats = {
                        'count': len(prices),
                        'mean': mean_price,
                        'median': median_price,
                        'min': prices.min(),
                        'max': prices.max(),
                        'std': prices.std()
                    }
                    
                    # Benchmark Comparison
                    st.markdown("### 📊 Benchmark Analysis")
                    col_bench1, col_bench2 = st.columns(2)
                    
                    with col_bench1:
                        diff_percent = ((mean_price - overall_avg) / overall_avg) * 100
                        if diff_percent > 0:
                            st.warning(f"⚠️ This code is **{diff_percent:.1f}% MORE expensive** than system average (${overall_avg:,.0f})")
                        else:
                            st.success(f"✅ This code is **{abs(diff_percent):.1f}% CHEAPER** than system average (${overall_avg:,.0f})")
                    
                    with col_bench2:
                        percentile_rank = (summary['mean_rate'] < mean_price).sum() / len(summary) * 100
                        st.info(f"📈 This code is in the **{percentile_rank:.0f}th percentile** of all codes")
                    
                    st.markdown("---")
                    
                    # Anomaly Detection
                    anomalies = detect_anomalies(prices)
                    if anomalies:
                        st.markdown("### 🚨 Price Anomaly Alert")
                        for anomaly in anomalies[:5]:
                            st.error(f"⚠️ Suspicious price detected: **${anomaly:,.0f}** (outlier)")
                        if len(anomalies) > 5:
                            st.caption(f"... and {len(anomalies) - 5} more anomalies")
                        st.markdown("---")
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Count", f"{len(prices):,}")
                    col2.metric("Mean", f"${mean_price:,.0f}")
                    col3.metric("Median", f"${median_price:,.0f}")
                    col4.metric("Std", f"${prices.std():,.0f}")
                    
                    # Plot
                    st.pyplot(plot_price_distribution(subset, search_code))
                    
                    # PDF Export
                    st.markdown("---")
                    st.markdown("### 📄 Export Report")
                    
                    cheapest, expensive, _ = get_best_worst_providers(subset, search_code)
                    
                    if st.button("📥 Download PDF Report", key="pdf_btn"):
                        with st.spinner("Generating PDF report..."):
                            pdf_path = create_pdf_report(search_code, current_stats, anomalies, cheapest, expensive, overall_avg)
                            with open(pdf_path, "rb") as f:
                                pdf_data = f.read()
                            st.download_button(
                                label="✅ Click to Download PDF",
                                data=pdf_data,
                                file_name=f"cigna_report_{search_code}.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )
                            os.unlink(pdf_path)
                    
                    # Sample data
                    with st.expander("📊 View Sample Data (first 100 rows)"):
                        st.dataframe(subset[['provider_name', 'negotiated_rate', 'negotiated_type', 'description']].head(100))
                else:
                    st.error(f"❌ Code '{search_code}' not found")
                    st.info(f"💡 Try: {', '.join(top_codes[:10])}")
        
        else:
            st.info("💡 Search by description to find the billing code")
            descriptions = load_all_descriptions()
            if descriptions:
                search_desc = st.selectbox("Select description", options=[''] + descriptions[:100])
                if search_desc:
                    with st.spinner("Searching..."):
                        code = find_code_by_description(search_desc)
                    if code:
                        st.success(f"✅ Found code: **{code}**")
                        with st.spinner(f"Loading {code}..."):
                            subset = get_code_info(code)
                        if not subset.empty:
                            st.pyplot(plot_price_distribution(subset, code))
                    else:
                        st.warning("No matching code found")
    
    # ============================================
    # TAB 2: COMPARE
    # ============================================
    with tab2:
        st.subheader("📊 Compare Two Billing Codes")
        st.info("💡 You can also type custom codes (e.g., 76805, 99213)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            code1_option = st.radio("Code 1 input method", ["Select from list", "Type manually"], key="code1_method")
            if code1_option == "Select from list":
                code1 = st.selectbox("First code", options=top_codes[:30], key='compare1_select')
            else:
                code1 = st.text_input("Enter first code", placeholder="Example: 76805", key='compare1_manual')
        
        with col2:
            code2_option = st.radio("Code 2 input method", ["Select from list", "Type manually"], key="code2_method")
            if code2_option == "Select from list":
                code2 = st.selectbox("Second code", options=top_codes[:30], key='compare2_select')
            else:
                code2 = st.text_input("Enter second code", placeholder="Example: 99213", key='compare2_manual')
        
        if code1 and code2:
            with st.spinner("Loading comparison data..."):
                data1 = get_code_info(code1.strip())
                data2 = get_code_info(code2.strip())
            
            if not data1.empty and not data2.empty:
                st.pyplot(plot_compare_codes(data1, data2, code1, code2))
                
                stats1 = data1['negotiated_rate'].dropna()
                stats2 = data2['negotiated_rate'].dropna()
                
                comp_df = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Min', 'Max'],
                    code1: [len(stats1), f"${stats1.mean():,.0f}", f"${stats1.median():,.0f}", f"${stats1.min():,.0f}", f"${stats1.max():,.0f}"],
                    code2: [len(stats2), f"${stats2.mean():,.0f}", f"${stats2.median():,.0f}", f"${stats2.min():,.0f}", f"${stats2.max():,.0f}"]
                })
                st.dataframe(comp_df, use_container_width=True)
            elif data1.empty:
                st.error(f"❌ Code '{code1}' not found")
            elif data2.empty:
                st.error(f"❌ Code '{code2}' not found")
            else:
                st.warning("No data for comparison")
    
    # ============================================
    # TAB 3: TOP CODES
    # ============================================
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
    
    # ============================================
    # TAB 4: BEST/WORST PROVIDERS
    # ============================================
    with tab4:
        st.subheader("💡 Cheapest & Most Expensive Providers")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_code = st.selectbox("Select billing code", options=top_codes[:30], key='bw')
        
        with col2:
            manual_bw_code = st.text_input("Or enter any code", placeholder="Example: 76805", key='bw_manual')
        
        final_code = manual_bw_code.strip() if manual_bw_code else selected_code
        
        if final_code:
            with st.spinner("Analyzing providers..."):
                subset = get_code_info(final_code)
                cheapest, expensive, num_providers = get_best_worst_providers(subset, final_code)
            
            if cheapest:
                st.success(f"Results for code: **{final_code}** (Found {num_providers} providers)")
                
                if num_providers < 6:
                    st.info(f"ℹ️ Only {num_providers} provider(s) found. Showing cheapest only.")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("### 💚 Cheapest Providers")
                    for provider, price in cheapest.items():
                        st.metric(provider[:40], f"${price:,.0f}")
                
                with col_b:
                    st.markdown("### 🔥 Most Expensive Providers")
                    if expensive:
                        for provider, price in expensive.items():
                            st.metric(provider[:40], f"${price:,.0f}")
                    else:
                        st.caption("✨ Same as cheapest (only 1-3 providers available)")
            else:
                st.error(f"❌ No data found for code: {final_code}")

if __name__ == "__main__":
    main()