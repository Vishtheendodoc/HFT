
import requests
import pandas as pd
import time
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

# Set IST timezone
IST = pytz.timezone("Asia/Kolkata")

# Streamlit Configuration
st.set_page_config(page_title="Call/Put Writers Tracker", layout="wide")

# API Configuration
CLIENT_ID = '1100244268'
ACCESS_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU5Mjk2MzI2LCJpYXQiOjE3NTkyMDk5MjYsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMjQ0MjY4In0.4gPj992GB0WhYi3eLeTTvXXFMIKvQQ2y4uiCgUYxeStWxgrV8rdyIYxuhYuVuPUg1VGBE2_NwvPhDj2S7x2Wxw'
HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}

# Initialize session state for writer tracking
if 'writer_history' not in st.session_state:
    st.session_state.writer_history = []
if 'call_writers_timeline' not in st.session_state:
    st.session_state.call_writers_timeline = []
if 'put_writers_timeline' not in st.session_state:
    st.session_state.put_writers_timeline = []

# Writer Detection Settings
WRITER_DETECTION_SETTINGS = {
    'MIN_OI_INCREASE': 500,           # Minimum OI increase to consider
    'MIN_VOLUME': 100,                # Minimum volume for validity
    'MAX_IV_INCREASE': -2,            # IV should decrease for writing
    'MIN_TOTAL_OI': 5000,            # Minimum total OI for significance
    'INSTITUTIONAL_OI_THRESHOLD': 25000,  # OI threshold for institutional writers
    'LARGE_WRITER_THRESHOLD': 50000   # Very large writer threshold
}

def get_expiry_dates():
    """Get available expiry dates"""
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        return response.json()['data'] if response.status_code == 200 else []
    except:
        return []

def fetch_option_chain(expiry):
    """Fetch current option chain"""
    url = "https://api.dhan.co/v2/optionchain"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=20)
        time.sleep(1.5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def detect_writers(option_chain):
    """Detect call and put writers from option chain data"""
    if not option_chain or "data" not in option_chain:
        return pd.DataFrame(), 0

    data = option_chain["data"]
    underlying_price = data.get("last_price", 0)
    option_data = data.get("oc", {})

    call_writers = []
    put_writers = []
    current_time = datetime.now(IST)

    for strike_str, contracts in option_data.items():
        strike = float(strike_str)

        # Process Call Options (CE)
        ce_contract = contracts.get('ce', {})
        if ce_contract:
            oi = ce_contract.get('oi', 0)
            volume = ce_contract.get('volume', 0)
            ltp = ce_contract.get('last_price', 0)
            iv = ce_contract.get('implied_volatility', 0)

            # Writer detection logic for calls
            if (oi > WRITER_DETECTION_SETTINGS['MIN_TOTAL_OI'] and 
                volume > WRITER_DETECTION_SETTINGS['MIN_VOLUME']):

                # Simulate OI change and IV change (in real app, calculate from historical data)
                oi_change = np.random.uniform(-10, 20)  # Placeholder
                iv_change = np.random.uniform(-15, 10)   # Placeholder

                # Strong writer signal: High OI increase + Low/Decreasing IV + Good volume
                writer_strength = 0
                writer_type = "None"

                if oi_change > WRITER_DETECTION_SETTINGS['MIN_OI_INCREASE']:
                    writer_strength += 30

                if iv_change < WRITER_DETECTION_SETTINGS['MAX_IV_INCREASE']:
                    writer_strength += 40  # IV decrease is strong writer signal

                if volume > 500:
                    writer_strength += 20

                if oi > WRITER_DETECTION_SETTINGS['INSTITUTIONAL_OI_THRESHOLD']:
                    writer_strength += 10  # Large OI suggests institutional

                # Classify writer strength
                if writer_strength >= 70:
                    writer_type = "Strong Call Writer"
                elif writer_strength >= 50:
                    writer_type = "Moderate Call Writer"
                elif writer_strength >= 30:
                    writer_type = "Weak Call Writer"

                if writer_type != "None":
                    # Determine writer category
                    if oi > WRITER_DETECTION_SETTINGS['LARGE_WRITER_THRESHOLD']:
                        writer_category = "Institutional"
                    elif oi > WRITER_DETECTION_SETTINGS['INSTITUTIONAL_OI_THRESHOLD']:
                        writer_category = "Professional"
                    else:
                        writer_category = "Retail"

                    call_writers.append({
                        'Time': current_time.strftime('%H:%M:%S'),
                        'DateTime': current_time,
                        'Strike': strike,
                        'Type': 'CE',
                        'OI': oi,
                        'Volume': volume,
                        'LTP': ltp,
                        'IV': iv,
                        'OI_Change': oi_change,
                        'IV_Change': iv_change,
                        'WriterStrength': writer_strength,
                        'WriterType': writer_type,
                        'WriterCategory': writer_category,
                        'NotionalValue': oi * ltp * 75,  # Lot size 75
                        'UnderlyingPrice': underlying_price
                    })

        # Process Put Options (PE)
        pe_contract = contracts.get('pe', {})
        if pe_contract:
            oi = pe_contract.get('oi', 0)
            volume = pe_contract.get('volume', 0)
            ltp = pe_contract.get('last_price', 0)
            iv = pe_contract.get('implied_volatility', 0)

            # Writer detection logic for puts
            if (oi > WRITER_DETECTION_SETTINGS['MIN_TOTAL_OI'] and 
                volume > WRITER_DETECTION_SETTINGS['MIN_VOLUME']):

                oi_change = np.random.uniform(-10, 20)  # Placeholder
                iv_change = np.random.uniform(-15, 10)   # Placeholder

                writer_strength = 0
                writer_type = "None"

                if oi_change > WRITER_DETECTION_SETTINGS['MIN_OI_INCREASE']:
                    writer_strength += 30

                if iv_change < WRITER_DETECTION_SETTINGS['MAX_IV_INCREASE']:
                    writer_strength += 40

                if volume > 500:
                    writer_strength += 20

                if oi > WRITER_DETECTION_SETTINGS['INSTITUTIONAL_OI_THRESHOLD']:
                    writer_strength += 10

                # Classify writer strength
                if writer_strength >= 70:
                    writer_type = "Strong Put Writer"
                elif writer_strength >= 50:
                    writer_type = "Moderate Put Writer"
                elif writer_strength >= 30:
                    writer_type = "Weak Put Writer"

                if writer_type != "None":
                    # Determine writer category
                    if oi > WRITER_DETECTION_SETTINGS['LARGE_WRITER_THRESHOLD']:
                        writer_category = "Institutional"
                    elif oi > WRITER_DETECTION_SETTINGS['INSTITUTIONAL_OI_THRESHOLD']:
                        writer_category = "Professional"
                    else:
                        writer_category = "Retail"

                    put_writers.append({
                        'Time': current_time.strftime('%H:%M:%S'),
                        'DateTime': current_time,
                        'Strike': strike,
                        'Type': 'PE',
                        'OI': oi,
                        'Volume': volume,
                        'LTP': ltp,
                        'IV': iv,
                        'OI_Change': oi_change,
                        'IV_Change': iv_change,
                        'WriterStrength': writer_strength,
                        'WriterType': writer_type,
                        'WriterCategory': writer_category,
                        'NotionalValue': oi * ltp * 75,
                        'UnderlyingPrice': underlying_price
                    })

    # Combine and return
    writers_df = pd.DataFrame(call_writers + put_writers)
    return writers_df, underlying_price

def update_writer_timeline(writers_df):
    """Update continuous writer timeline data"""
    current_time = datetime.now(IST)

    # Aggregate call writers
    call_writers = writers_df[writers_df['Type'] == 'CE']
    call_writer_summary = {
        'Time': current_time.strftime('%H:%M:%S'),
        'DateTime': current_time,
        'TotalCallWriters': len(call_writers),
        'StrongCallWriters': len(call_writers[call_writers['WriterType'] == 'Strong Call Writer']),
        'InstitutionalCallWriters': len(call_writers[call_writers['WriterCategory'] == 'Institutional']),
        'CallWriterNotional': call_writers['NotionalValue'].sum(),
        'CallWriterOI': call_writers['OI'].sum(),
        'AvgCallWriterStrength': call_writers['WriterStrength'].mean() if len(call_writers) > 0 else 0
    }

    # Aggregate put writers
    put_writers = writers_df[writers_df['Type'] == 'PE']
    put_writer_summary = {
        'Time': current_time.strftime('%H:%M:%S'),
        'DateTime': current_time,
        'TotalPutWriters': len(put_writers),
        'StrongPutWriters': len(put_writers[put_writers['WriterType'] == 'Strong Put Writer']),
        'InstitutionalPutWriters': len(put_writers[put_writers['WriterCategory'] == 'Institutional']),
        'PutWriterNotional': put_writers['NotionalValue'].sum(),
        'PutWriterOI': put_writers['OI'].sum(),
        'AvgPutWriterStrength': put_writers['WriterStrength'].mean() if len(put_writers) > 0 else 0
    }

    # Store in session state
    st.session_state.call_writers_timeline.append(call_writer_summary)
    st.session_state.put_writers_timeline.append(put_writer_summary)
    st.session_state.writer_history.extend(writers_df.to_dict('records'))

    # Keep only last 50 records for performance
    for key in ['call_writers_timeline', 'put_writers_timeline']:
        if len(st.session_state[key]) > 50:
            st.session_state[key] = st.session_state[key][-50:]

    if len(st.session_state.writer_history) > 500:
        st.session_state.writer_history = st.session_state.writer_history[-500:]

def render_writer_strength_chart():
    """Render call vs put writer strength over time"""
    if len(st.session_state.call_writers_timeline) < 2:
        st.info("📊 Collecting writer data for charts...")
        return

    call_df = pd.DataFrame(st.session_state.call_writers_timeline)
    put_df = pd.DataFrame(st.session_state.put_writers_timeline)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('📉 Call Writers Activity (Bearish)', '📈 Put Writers Activity (Bullish)'),
        row_heights=[0.5, 0.5]
    )

    # Call Writers (Bearish - Red theme)
    fig.add_trace(
        go.Bar(
            x=call_df['Time'],
            y=call_df['TotalCallWriters'],
            name='Total Call Writers',
            marker_color='#FF4444',
            opacity=0.7,
            hovertemplate='<b>Call Writers</b><br>' +
                         'Total: %{y}<br>' +
                         'Time: %{x}<extra></extra>'
        ), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=call_df['Time'],
            y=call_df['InstitutionalCallWriters'],
            mode='lines+markers',
            name='Institutional Call Writers',
            line=dict(color='#8B0000', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Institutional Call Writers</b><br>' +
                         'Count: %{y}<br>' +
                         'Time: %{x}<extra></extra>'
        ), row=1, col=1
    )

    # Put Writers (Bullish - Green theme)
    fig.add_trace(
        go.Bar(
            x=put_df['Time'],
            y=put_df['TotalPutWriters'],
            name='Total Put Writers',
            marker_color='#00C851',
            opacity=0.7,
            hovertemplate='<b>Put Writers</b><br>' +
                         'Total: %{y}<br>' +
                         'Time: %{x}<extra></extra>'
        ), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=put_df['Time'],
            y=put_df['InstitutionalPutWriters'],
            mode='lines+markers',
            name='Institutional Put Writers',
            line=dict(color='#004D1F', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Institutional Put Writers</b><br>' +
                         'Count: %{y}<br>' +
                         'Time: %{x}<extra></extra>'
        ), row=2, col=1
    )

    fig.update_layout(
        title=dict(
            text="✍️ CALL vs PUT WRITERS TIMELINE",
            x=0.5,
            font=dict(size=20, color='#2C3E50')
        ),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Clean axes
    fig.update_xaxes(showgrid=False, title_text="Time", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    st.plotly_chart(fig, use_container_width=True)

def render_writer_notional_chart():
    """Render writer notional value comparison"""
    if len(st.session_state.call_writers_timeline) < 2:
        return

    call_df = pd.DataFrame(st.session_state.call_writers_timeline)
    put_df = pd.DataFrame(st.session_state.put_writers_timeline)

    fig = go.Figure()

    # Call writer notional (negative for visual distinction)
    fig.add_trace(
        go.Bar(
            x=call_df['Time'],
            y=-(call_df['CallWriterNotional'] / 1000000),  # Convert to millions, negative
            name='Call Writer Value (₹ Millions)',
            marker_color='#FF4444',
            opacity=0.8,
            hovertemplate='<b>Call Writers Notional</b><br>' +
                         'Value: ₹%{y:,.0f}M<br>' +
                         'Time: %{x}<extra></extra>'
        )
    )

    # Put writer notional (positive)
    fig.add_trace(
        go.Bar(
            x=put_df['Time'],
            y=put_df['PutWriterNotional'] / 1000000,  # Convert to millions
            name='Put Writer Value (₹ Millions)',
            marker_color='#00C851',
            opacity=0.8,
            hovertemplate='<b>Put Writers Notional</b><br>' +
                         'Value: ₹%{y:,.0f}M<br>' +
                         'Time: %{x}<extra></extra>'
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.8)

    fig.update_layout(
        title=dict(
            text="💰 WRITER NOTIONAL VALUES - CALL vs PUT",
            x=0.5,
            font=dict(size=18, color='#2C3E50')
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title="Time",
        yaxis_title="Notional Value (₹ Millions)",
        showlegend=True,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text='🔴 Call Writing (Bearish)',
                showarrow=False,
                font=dict(size=12, color='#FF4444'),
                xanchor='left'
            ),
            dict(
                x=0.02, y=0.05,
                xref='paper', yref='paper',
                text='🟢 Put Writing (Bullish)',
                showarrow=False,
                font=dict(size=12, color='#00C851'),
                xanchor='left'
            )
        ]
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    st.plotly_chart(fig, use_container_width=True)

def render_strike_wise_writers(writers_df, underlying_price):
    """Render strike-wise writer distribution"""
    if writers_df.empty:
        st.info("No writers detected in current data")
        return

    # Separate call and put writers
    call_writers = writers_df[writers_df['Type'] == 'CE'].copy()
    put_writers = writers_df[writers_df['Type'] == 'PE'].copy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🔴 Call Writers by Strike', '🟢 Put Writers by Strike'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )

    # Call writers
    if not call_writers.empty:
        # Color by writer strength
        colors = ['#8B0000' if strength >= 70 else '#FF4444' if strength >= 50 else '#FFB6C1' 
                 for strength in call_writers['WriterStrength']]

        fig.add_trace(
            go.Bar(
                x=call_writers['Strike'],
                y=call_writers['OI'],
                name='Call Writers OI',
                marker_color=colors,
                text=call_writers['WriterCategory'],
                textposition='auto',
                hovertemplate='<b>Call Writer</b><br>' +
                             'Strike: %{x}<br>' +
                             'OI: %{y:,.0f}<br>' +
                             'Category: %{text}<br>' +
                             'Strength: ' + call_writers['WriterStrength'].astype(str) + '<extra></extra>',
                customdata=call_writers['WriterStrength']
            ), row=1, col=1
        )

    # Put writers
    if not put_writers.empty:
        # Color by writer strength
        colors = ['#006400' if strength >= 70 else '#00C851' if strength >= 50 else '#90EE90' 
                 for strength in put_writers['WriterStrength']]

        fig.add_trace(
            go.Bar(
                x=put_writers['Strike'],
                y=put_writers['OI'],
                name='Put Writers OI',
                marker_color=colors,
                text=put_writers['WriterCategory'],
                textposition='auto',
                hovertemplate='<b>Put Writer</b><br>' +
                             'Strike: %{x}<br>' +
                             'OI: %{y:,.0f}<br>' +
                             'Category: %{text}<br>' +
                             'Strength: ' + put_writers['WriterStrength'].astype(str) + '<extra></extra>',
                customdata=put_writers['WriterStrength']
            ), row=1, col=2
        )

    # Add ATM lines
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="blue", opacity=0.8, row=1, col=1)
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="blue", opacity=0.8, row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"🎯 WRITERS BY STRIKE (ATM: {underlying_price:.0f})",
            x=0.5,
            font=dict(size=18, color='#2C3E50')
        ),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )

    fig.update_xaxes(title_text="Strike Price", showgrid=False)
    fig.update_yaxes(title_text="Open Interest", showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    st.plotly_chart(fig, use_container_width=True)

def render_writer_summary_table(writers_df):
    """Show detailed writer summary table"""
    if writers_df.empty:
        st.info("No writers detected")
        return

    st.subheader("📋 DETAILED WRITERS ANALYSIS")

    # Sort by writer strength
    display_df = writers_df.nlargest(20, 'WriterStrength')[
        ['Strike', 'Type', 'WriterType', 'WriterCategory', 'WriterStrength', 
         'OI', 'Volume', 'LTP', 'IV', 'NotionalValue']
    ].copy()

    # Format for display
    display_df['Strength'] = display_df['WriterStrength'].round(0).astype(int)
    display_df['OI(K)'] = (display_df['OI'] / 1000).round(1)
    display_df['Notional(₹Cr)'] = (display_df['NotionalValue'] / 10000000).round(1)
    display_df['LTP'] = display_df['LTP'].round(2)
    display_df['IV%'] = display_df['IV'].round(1)

    # Select final columns
    final_cols = ['Strike', 'Type', 'WriterType', 'WriterCategory', 
                 'Strength', 'OI(K)', 'Volume', 'LTP', 'IV%', 'Notional(₹Cr)']

    # Style the dataframe
    def highlight_strong_writers(row):
        if row['Strength'] >= 70:
            return ['background-color: #ffcccc; font-weight: bold'] * len(row)
        elif row['Strength'] >= 50:
            return ['background-color: #fff2cc'] * len(row)
        return [''] * len(row)

    styled_df = display_df[final_cols].style.apply(highlight_strong_writers, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

def main():
    """Main application focused on call/put writers only"""
    st.title("✍️ CALL & PUT WRITERS TRACKER")
    st.markdown("**Focused on Writer Detection • Real-time Analysis • Institutional Intelligence**")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval", 20, 120, 45, help="Seconds")

        st.markdown("---")
        st.header("📊 Writer Stats")

        if st.session_state.call_writers_timeline:
            latest_call = st.session_state.call_writers_timeline[-1]
            latest_put = st.session_state.put_writers_timeline[-1]

            st.metric("📉 Call Writers", latest_call['TotalCallWriters'])
            st.metric("📈 Put Writers", latest_put['TotalPutWriters'])

            # Writer sentiment
            call_strength = latest_call['AvgCallWriterStrength']
            put_strength = latest_put['AvgPutWriterStrength']

            if call_strength > put_strength + 10:
                st.error("🐻 BEARISH - Strong Call Writing")
            elif put_strength > call_strength + 10:
                st.success("🐂 BULLISH - Strong Put Writing")
            else:
                st.info("⚖️ NEUTRAL - Balanced Writing")

        st.markdown("---")
        if st.button("🔄 Manual Refresh", type="primary"):
            st.rerun()

        if st.button("🗑️ Clear History"):
            for key in ['writer_history', 'call_writers_timeline', 'put_writers_timeline']:
                st.session_state[key] = []
            st.success("History cleared!")
            time.sleep(1)
            st.rerun()

    # Get data
    expiry_dates = get_expiry_dates()
    if not expiry_dates:
        st.error("❌ Unable to fetch expiry dates")
        return

    selected_expiry = expiry_dates[0]
    st.info(f"📅 **Expiry:** {selected_expiry}")

    # Fetch and detect writers
    with st.spinner("🔍 Scanning for Call & Put Writers..."):
        option_chain = fetch_option_chain(selected_expiry)

        if not option_chain:
            st.error("❌ Failed to fetch option chain data")
            return

        writers_df, underlying_price = detect_writers(option_chain)

        if writers_df.empty:
            st.warning("⚠️ No significant writers detected in current scan")
            return

        # Update timeline
        update_writer_timeline(writers_df)

        # Success metrics
        call_writers_count = len(writers_df[writers_df['Type'] == 'CE'])
        put_writers_count = len(writers_df[writers_df['Type'] == 'PE'])
        institutional_writers = len(writers_df[writers_df['WriterCategory'] == 'Institutional'])

        st.success(f"✅ **Nifty: ₹{underlying_price:,.2f}** • **{call_writers_count} Call Writers** • **{put_writers_count} Put Writers** • **{institutional_writers} Institutional**")

    # Main charts layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Primary writer strength timeline
        render_writer_strength_chart()

        # Writer notional comparison
        render_writer_notional_chart()

    with col2:
        st.subheader("🎯 Key Insights")

        # Latest writer stats
        if writers_df is not None and not writers_df.empty:
            strong_call_writers = len(writers_df[
                (writers_df['Type'] == 'CE') & (writers_df['WriterStrength'] >= 70)
            ])
            strong_put_writers = len(writers_df[
                (writers_df['Type'] == 'PE') & (writers_df['WriterStrength'] >= 70)
            ])

            st.metric("💪 Strong Call Writers", strong_call_writers)
            st.metric("💪 Strong Put Writers", strong_put_writers)

            # Directional bias
            if strong_call_writers > strong_put_writers + 2:
                st.error("📉 **BEARISH BIAS**\nMore call writing detected")
            elif strong_put_writers > strong_call_writers + 2:
                st.success("📈 **BULLISH BIAS**\nMore put writing detected")
            else:
                st.info("⚖️ **NEUTRAL**\nBalanced writer activity")

            # Top strikes
            st.markdown("**🔥 Active Writer Strikes:**")
            top_strikes = writers_df.nlargest(5, 'WriterStrength')[
                ['Strike', 'Type', 'WriterStrength']
            ]
            for _, row in top_strikes.iterrows():
                strength_emoji = "🔥" if row['WriterStrength'] >= 70 else "⚡" if row['WriterStrength'] >= 50 else "💫"
                color_emoji = "🔴" if row['Type'] == 'CE' else "🟢"
                st.write(f"{strength_emoji} {color_emoji} {row['Strike']:.0f} {row['Type']} ({row['WriterStrength']:.0f})")

    # Strike-wise analysis
    st.markdown("---")
    render_strike_wise_writers(writers_df, underlying_price)

    # Detailed table
    st.markdown("---")
    render_writer_summary_table(writers_df)

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
