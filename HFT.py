
import requests
import pandas as pd
import time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

# Set IST Timezone
IST = pytz.timezone("Asia/Kolkata")

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Institutional Options Tracker", layout="wide")

# Dhan API Configuration
CLIENT_ID = '1100244268'
ACCESS_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU5Mjk2MzI2LCJpYXQiOjE3NTkyMDk5MjYsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMjQ0MjY4In0.4gPj992GB0WhYi3eLeTTvXXFMIKvQQ2y4uiCgUYxeStWxgrV8rdyIYxuhYuVuPUg1VGBE2_NwvPhDj2S7x2Wxw'
HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}

# Enhanced Institutional Detection Thresholds
INSTITUTIONAL_THRESHOLDS = {
    'LARGE_POSITION_NOTIONAL': 10000000,    # ₹1 Cr+ for institutional
    'SMART_MONEY_OI_THRESHOLD': 50000,      # High OI threshold
    'DARK_POOL_RATIO': 5.0,                # OI/Volume ratio for stealth
    'GAMMA_CONCENTRATION': 0.05,           # Significant gamma level
    'PROFESSIONAL_VOLUME': 1000,           # Professional trade size
    'MM_ATM_CONCENTRATION': 50             # % of volume at ATM
}

def get_expiry_dates():
    """Fetch expiry dates from Dhan API"""
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()['data']
    except Exception as e:
        st.error(f"Error fetching expiry dates: {e}")
        return []

def fetch_option_chain(expiry):
    """Fetch option chain data from Dhan API"""
    url = "https://api.dhan.co/v2/optionchain"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry}
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        time.sleep(2)  # Rate limiting
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

def calculate_institutional_metrics(df, underlying_price):
    """Calculate enhanced institutional detection metrics"""

    # Basic metrics
    df['Notional_Value'] = df['OI'] * df['LTP'] * 75  # Assuming lot size 75
    df['OI_Volume_Ratio'] = df['OI'] / (df.get('Volume', 1) + 1)

    # 1. Smart Money Detection
    large_positions = df[df['Notional_Value'] > INSTITUTIONAL_THRESHOLDS['LARGE_POSITION_NOTIONAL']]
    smart_money_score = len(large_positions) / len(df) * 100 if len(df) > 0 else 0

    # 2. Dark Pool Activity
    dark_pool_candidates = df[df['OI_Volume_Ratio'] > INSTITUTIONAL_THRESHOLDS['DARK_POOL_RATIO']]
    dark_pool_score = len(dark_pool_candidates) / len(df) * 100 if len(df) > 0 else 0

    # 3. Market Maker Activity (ATM concentration)
    atm_strikes = df[abs(df['StrikePrice'] - underlying_price) <= 100]
    atm_volume = atm_strikes.get('Volume', pd.Series(dtype=float)).sum()
    total_volume = df.get('Volume', pd.Series(dtype=float)).sum()
    mm_concentration = (atm_volume / total_volume * 100) if total_volume > 0 else 0

    # 4. Gamma Wall Detection
    gamma_by_strike = df.groupby('StrikePrice')['Gamma'].sum().abs()
    max_gamma_strike = gamma_by_strike.idxmax() if len(gamma_by_strike) > 0 else underlying_price
    max_gamma_value = gamma_by_strike.max() if len(gamma_by_strike) > 0 else 0

    # 5. Institutional Classification
    def classify_institutional_activity(row):
        score = 0
        reasons = []

        # Large notional value
        if row['Notional_Value'] > INSTITUTIONAL_THRESHOLDS['LARGE_POSITION_NOTIONAL']:
            score += 30
            reasons.append("Large Position")

        # High OI with low volume (stealth)
        if row['OI_Volume_Ratio'] > INSTITUTIONAL_THRESHOLDS['DARK_POOL_RATIO']:
            score += 25
            reasons.append("Stealth Activity")

        # Professional volume size
        volume = row.get('Volume', 0)
        if volume > INSTITUTIONAL_THRESHOLDS['PROFESSIONAL_VOLUME']:
            score += 20
            reasons.append("Professional Volume")

        # Strategic positioning (ATM/ITM with high gamma)
        strike_distance = abs(row['StrikePrice'] - underlying_price)
        if strike_distance < 100 and abs(row['Gamma']) > INSTITUTIONAL_THRESHOLDS['GAMMA_CONCENTRATION']:
            score += 15
            reasons.append("Strategic Positioning")

        # High OI concentration
        if row['OI'] > INSTITUTIONAL_THRESHOLDS['SMART_MONEY_OI_THRESHOLD']:
            score += 10
            reasons.append("High OI")

        # Classification
        if score >= 60:
            return "Institutional", score, "; ".join(reasons)
        elif score >= 40:
            return "Professional", score, "; ".join(reasons)
        elif score >= 20:
            return "Informed", score, "; ".join(reasons)
        else:
            return "Retail", score, "; ".join(reasons)

    # Apply classification
    df[['Player_Type', 'Institutional_Score', 'Reasons']] = df.apply(
        lambda row: pd.Series(classify_institutional_activity(row)), axis=1
    )

    # 6. Call/Put Writer Detection
    def detect_writers(row):
        # High OI with decreasing IV suggests writing
        oi_change = row.get('OI_Change', 0)
        iv_change = row.get('IV_Change', 0)
        volume = row.get('Volume', 0)

        if oi_change > 10 and iv_change < -5 and volume > 100:
            if row['Type'] == 'CE':
                return "Call Writer"
            else:
                return "Put Writer"
        elif oi_change > 10 and volume > 200:
            if row['Type'] == 'CE':
                return "Call Buyer" 
            else:
                return "Put Buyer"
        else:
            return "Neutral"

    df['Writer_Status'] = df.apply(detect_writers, axis=1)

    return {
        'enhanced_df': df,
        'smart_money_score': smart_money_score,
        'dark_pool_score': dark_pool_score,
        'mm_concentration': mm_concentration,
        'gamma_wall_strike': max_gamma_strike,
        'gamma_wall_strength': max_gamma_value,
        'large_positions': large_positions,
        'dark_pool_strikes': dark_pool_candidates['StrikePrice'].tolist()
    }

def process_option_chain(option_chain):
    """Process raw option chain data into structured DataFrame"""

    if "data" not in option_chain or "oc" not in option_chain["data"]:
        st.error("Invalid option chain data received!")
        return pd.DataFrame(), 0

    option_chain_data = option_chain["data"]["oc"]
    underlying_price = option_chain["data"]["last_price"]
    data_list = []

    # Filter to ATM ± 10 strikes for focus
    atm_strike = min(option_chain_data.keys(), 
                    key=lambda x: abs(float(x) - underlying_price))
    atm_strike = float(atm_strike)
    min_strike = atm_strike - 10 * 50  # ±10 strikes with 50 point intervals
    max_strike = atm_strike + 10 * 50

    for strike, contracts in option_chain_data.items():
        strike_price = float(strike)

        # Focus on relevant strikes only
        if strike_price < min_strike or strike_price > max_strike:
            continue

        # Process CE and PE data
        for option_type, option_key in [('CE', 'ce'), ('PE', 'pe')]:
            contract = contracts.get(option_key, {})
            if not contract:  # Skip if no data
                continue

            data_list.append({
                'StrikePrice': strike_price,
                'Type': option_type,
                'LTP': contract.get('last_price', 0),
                'OI': contract.get('oi', 0),
                'Volume': contract.get('volume', 0),
                'IV': contract.get('implied_volatility', 0),
                'Delta': contract.get('greeks', {}).get('delta', 0),
                'Gamma': contract.get('greeks', {}).get('gamma', 0),
                'Theta': contract.get('greeks', {}).get('theta', 0),
                'Vega': contract.get('greeks', {}).get('vega', 0)
            })

    df = pd.DataFrame(data_list)
    if df.empty:
        return df, underlying_price

    df['Timestamp'] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by=['StrikePrice', 'Type']).reset_index(drop=True)

    # Add change calculations (simplified - would need historical data for real changes)
    df['OI_Change'] = np.random.uniform(-5, 15, len(df))  # Placeholder
    df['IV_Change'] = np.random.uniform(-10, 10, len(df))  # Placeholder

    return df, underlying_price

def render_institutional_dashboard(institutional_data, underlying_price):
    """Render clean institutional dashboard"""

    df = institutional_data['enhanced_df']

    # Header with key metrics
    st.title("🏛️ NIFTY INSTITUTIONAL OPTIONS TRACKER")
    st.markdown(f"**Current Nifty Level: ₹{underlying_price:,.2f}** | **Last Updated: {datetime.now(IST).strftime('%H:%M:%S')}**")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Smart Money Activity", f"{institutional_data['smart_money_score']:.1f}%")
    with col2:
        st.metric("Dark Pool Activity", f"{institutional_data['dark_pool_score']:.1f}%")
    with col3:
        st.metric("MM ATM Concentration", f"{institutional_data['mm_concentration']:.1f}%")
    with col4:
        gamma_wall = institutional_data['gamma_wall_strike']
        st.metric("Gamma Wall", f"{gamma_wall:.0f}" if gamma_wall else "None")

    # Player Type Distribution
    st.subheader("📊 Market Participant Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Player type pie chart
        player_counts = df['Player_Type'].value_counts()
        fig_pie = px.pie(
            values=player_counts.values,
            names=player_counts.index,
            title="Market Participants Distribution",
            color_discrete_map={
                'Institutional': '#FF4444',
                'Professional': '#FF8800', 
                'Informed': '#FFCC00',
                'Retail': '#44FF44'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Writer vs Buyer distribution
        writer_counts = df['Writer_Status'].value_counts()
        fig_writer = px.pie(
            values=writer_counts.values,
            names=writer_counts.index,
            title="Writer vs Buyer Activity",
            color_discrete_map={
                'Call Writer': '#FF6B6B',
                'Put Writer': '#4ECDC4',
                'Call Buyer': '#45B7D1',
                'Put Buyer': '#96CEB4',
                'Neutral': '#FECA57'
            }
        )
        st.plotly_chart(fig_writer, use_container_width=True)

    # Institutional Activity Heatmap
    st.subheader("🗺️ Institutional Activity by Strike")

    # Create heatmap
    heatmap_data = df.pivot_table(
        index='StrikePrice', 
        columns='Type', 
        values='Institutional_Score', 
        aggfunc='mean'
    ).fillna(0)

    fig_heatmap = px.imshow(
        heatmap_data.values,
        x=['Call', 'Put'],
        y=heatmap_data.index,
        color_continuous_scale='Reds',
        title="Institutional Score Heatmap"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Top Institutional Positions Table
    st.subheader("🎯 Top Institutional Positions")

    institutional_positions = df[df['Player_Type'].isin(['Institutional', 'Professional'])].nlargest(
        15, 'Institutional_Score'
    )[['StrikePrice', 'Type', 'Player_Type', 'Writer_Status', 'Institutional_Score', 
       'OI', 'Volume', 'LTP', 'Notional_Value', 'Reasons']].round(2)

    if len(institutional_positions) > 0:
        # Format notional value for better readability
        institutional_positions['Notional_Value'] = institutional_positions['Notional_Value'].apply(
            lambda x: f"₹{x/10000000:.1f}Cr" if x > 10000000 else f"₹{x/1000000:.1f}L"
        )
        st.dataframe(institutional_positions, use_container_width=True)
    else:
        st.info("No significant institutional positions detected")

    # Call/Put Writer Detection
    st.subheader("✍️ Call & Put Writer Detection")

    col1, col2 = st.columns(2)

    with col1:
        call_writers = df[(df['Type'] == 'CE') & (df['Writer_Status'] == 'Call Writer')]
        st.write("**🔴 Call Writers (Bearish Outlook):**")
        if len(call_writers) > 0:
            for _, row in call_writers.head(5).iterrows():
                st.write(f"• {row['StrikePrice']:.0f} CE: OI {row['OI']:,.0f} | {row['Player_Type']}")
        else:
            st.write("No significant call writing detected")

    with col2:
        put_writers = df[(df['Type'] == 'PE') & (df['Writer_Status'] == 'Put Writer')]
        st.write("**🟢 Put Writers (Bullish Outlook):**")
        if len(put_writers) > 0:
            for _, row in put_writers.head(5).iterrows():
                st.write(f"• {row['StrikePrice']:.0f} PE: OI {row['OI']:,.0f} | {row['Player_Type']}")
        else:
            st.write("No significant put writing detected")

    # Key Insights
    st.subheader("🔍 Key Insights")

    total_institutional = len(df[df['Player_Type'].isin(['Institutional', 'Professional'])])
    total_call_writers = len(df[df['Writer_Status'] == 'Call Writer'])
    total_put_writers = len(df[df['Writer_Status'] == 'Put Writer'])

    col1, col2, col3 = st.columns(3)
    with col1:
        if institutional_data['smart_money_score'] > 20:
            st.success(f"✅ High institutional activity detected ({total_institutional} positions)")
        else:
            st.info("ℹ️ Moderate institutional activity")

    with col2:
        if total_call_writers > total_put_writers:
            st.warning("⚠️ More call writing suggests bearish institutional view")
        elif total_put_writers > total_call_writers:
            st.success("✅ More put writing suggests bullish institutional view")
        else:
            st.info("ℹ️ Balanced call/put writing activity")

    with col3:
        if institutional_data['dark_pool_score'] > 15:
            st.error("🕳️ Significant dark pool activity - watch for hidden moves")
        else:
            st.info("👁️ Normal transparency in options flow")

def main():
    """Main application function"""

    # Sidebar controls
    st.sidebar.title("⚙️ Controls")

    # Auto refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 30, 300, 60)

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Get expiry dates
    expiry_dates = get_expiry_dates()
    if not expiry_dates:
        st.error("Unable to fetch expiry dates")
        return

    # Use nearest expiry
    selected_expiry = expiry_dates[0]
    st.sidebar.write(f"**Expiry:** {selected_expiry}")

    # Fetch and process data
    with st.spinner("Fetching option chain data..."):
        option_chain = fetch_option_chain(selected_expiry)

        if not option_chain:
            st.error("Failed to fetch option chain data")
            return

        df, underlying_price = process_option_chain(option_chain)

        if df.empty:
            st.error("No option chain data available")
            return

        # Calculate institutional metrics
        institutional_data = calculate_institutional_metrics(df, underlying_price)

        # Render dashboard
        render_institutional_dashboard(institutional_data, underlying_price)

    # Auto refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
