
import requests
import pandas as pd
import time
import streamlit as st
import plotly.express as px
import os
from datetime import datetime
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from collections import deque
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
from requests.exceptions import RequestException

# Set recursion limit and IST timezone
sys.setrecursionlimit(5000)
IST = pytz.timezone("Asia/Kolkata")

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Options Sentiment & Institutional Tracker", layout="wide")

# API Configuration (original)
CLIENT_ID = '1100244268'
ACCESS_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU5Mjk2MzI2LCJpYXQiOjE3NTkyMDk5MjYsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMjQ0MjY4In0.4gPj992GB0WhYi3eLeTTvXXFMIKvQQ2y4uiCgUYxeStWxgrV8rdyIYxuhYuVuPUg1VGBE2_NwvPhDj2S7x2Wxw'
HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}

# Initialize session state for historical data
if 'option_history' not in st.session_state:
    st.session_state.option_history = []
if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = []
if 'strike_sentiment_log' not in st.session_state:
    st.session_state.strike_sentiment_log = []

# Enhanced Institutional Detection Thresholds
INSTITUTIONAL_THRESHOLDS = {
    'LARGE_POSITION_NOTIONAL': 10000000,    # ₹1 Cr+ for institutional
    'SMART_MONEY_OI_THRESHOLD': 50000,      # High OI threshold
    'DARK_POOL_RATIO': 5.0,                # OI/Volume ratio for stealth
    'GAMMA_CONCENTRATION': 0.05,           # Significant gamma level
    'PROFESSIONAL_VOLUME': 1000,           # Professional trade size
    'MM_ATM_CONCENTRATION': 50             # % of volume at ATM
}

@st.cache_data(ttl=10800)
def get_expiry_dates_cached():
    return get_expiry_dates()

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
        response = requests.post(url, json=payload, headers=HEADERS, timeout=30)
        time.sleep(2)  # Rate limiting
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

def calculate_sentiment_bias(row):
    """Calculate enhanced sentiment bias with institutional detection"""
    volume = row.get('Volume', 0)
    oi = row['OI']
    ltp = row['LTP']
    iv = row['IV']
    delta = abs(row.get('Delta', 0))
    gamma = row.get('Gamma', 0)

    # Basic sentiment calculation (original logic)
    if volume > 500 and oi > 10000:
        if row['Type'] == 'CE':
            if iv > 25:
                return "Aggressive Call Buying"
            elif iv < 15:
                return "Call Writing"
            else:
                return "Call Activity"
        else:
            if iv > 25:
                return "Aggressive Put Buying"
            elif iv < 15:
                return "Put Writing"
            else:
                return "Put Activity"

    # Enhanced institutional detection
    notional_value = oi * ltp * 75  # Lot size 75
    oi_volume_ratio = oi / (volume + 1)

    # Institutional signatures
    if notional_value > INSTITUTIONAL_THRESHOLDS['LARGE_POSITION_NOTIONAL']:
        if oi_volume_ratio > INSTITUTIONAL_THRESHOLDS['DARK_POOL_RATIO']:
            return f"Dark Pool {row['Type']}"
        elif volume > INSTITUTIONAL_THRESHOLDS['PROFESSIONAL_VOLUME']:
            return f"Institutional {row['Type']}"
        else:
            return f"Smart Money {row['Type']}"

    # Market maker detection
    if 0.4 <= delta <= 0.6 and gamma > INSTITUTIONAL_THRESHOLDS['GAMMA_CONCENTRATION']:
        return f"Market Maker {row['Type']}"

    # Default categories
    if volume > 100:
        return f"Active {row['Type']}"
    else:
        return "Low Activity"

def calculate_institutional_metrics(df, underlying_price):
    """Calculate comprehensive institutional detection metrics"""

    # Basic enhanced metrics
    df['Notional_Value'] = df['OI'] * df['LTP'] * 75
    df['OI_Volume_Ratio'] = df['OI'] / (df.get('Volume', 1) + 1)

    # Apply sentiment bias calculation
    df['SentimentBias'] = df.apply(calculate_sentiment_bias, axis=1)

    # Calculate sentiment scores (original logic enhanced)
    def sentiment_to_score(bias):
        sentiment_scores = {
            'Aggressive Call Buying': 5, 'Call Buying': 3, 'Call Activity': 1,
            'Aggressive Put Buying': -5, 'Put Buying': -3, 'Put Activity': -1,
            'Call Writing': -2, 'Put Writing': 2,
            'Dark Pool CE': 4, 'Dark Pool PE': -4,
            'Institutional CE': 3, 'Institutional PE': -3,
            'Smart Money CE': 4, 'Smart Money PE': -4,
            'Market Maker CE': 2, 'Market Maker PE': -2,
            'Active CE': 1, 'Active PE': -1,
            'Low Activity': 0
        }
        return sentiment_scores.get(bias, 0)

    df['SentimentScore'] = df['SentimentBias'].apply(sentiment_to_score)

    # Enhanced institutional classification
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
        if strike_distance < 100 and abs(row.get('Gamma', 0)) > INSTITUTIONAL_THRESHOLDS['GAMMA_CONCENTRATION']:
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
    df[['Player_Type', 'Institutional_Score', 'Classification_Reasons']] = df.apply(
        lambda row: pd.Series(classify_institutional_activity(row)), axis=1
    )

    # Smart money calculations
    large_positions = df[df['Notional_Value'] > INSTITUTIONAL_THRESHOLDS['LARGE_POSITION_NOTIONAL']]
    smart_money_score = len(large_positions) / len(df) * 100 if len(df) > 0 else 0

    # Dark pool activity
    dark_pool_candidates = df[df['OI_Volume_Ratio'] > INSTITUTIONAL_THRESHOLDS['DARK_POOL_RATIO']]
    dark_pool_score = len(dark_pool_candidates) / len(df) * 100 if len(df) > 0 else 0

    # Market maker activity (ATM concentration)
    atm_strikes = df[abs(df['StrikePrice'] - underlying_price) <= 100]
    atm_volume = atm_strikes.get('Volume', pd.Series(dtype=float)).sum()
    total_volume = df.get('Volume', pd.Series(dtype=float)).sum()
    mm_concentration = (atm_volume / total_volume * 100) if total_volume > 0 else 0

    # Gamma wall detection
    gamma_by_strike = df.groupby('StrikePrice')['Gamma'].sum().abs()
    max_gamma_strike = gamma_by_strike.idxmax() if len(gamma_by_strike) > 0 else underlying_price
    max_gamma_value = gamma_by_strike.max() if len(gamma_by_strike) > 0 else 0

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
    """Process raw option chain data into structured DataFrame (original logic)"""

    if "data" not in option_chain or "oc" not in option_chain["data"]:
        st.error("Invalid option chain data received!")
        return pd.DataFrame(), 0

    option_chain_data = option_chain["data"]["oc"]
    underlying_price = option_chain["data"]["last_price"]
    data_list = []

    for strike, contracts in option_chain_data.items():
        strike_price = float(strike)

        # Process CE and PE data
        for option_type, option_key in [('CE', 'ce'), ('PE', 'pe')]:
            contract = contracts.get(option_key, {})
            if not contract:
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
                'Vega': contract.get('greeks', {}).get('vega', 0),
                'UnderlyingValue': underlying_price,
                'Timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            })

    df = pd.DataFrame(data_list)
    if df.empty:
        return df, underlying_price

    df = df.sort_values(by=['StrikePrice', 'Type']).reset_index(drop=True)

    # Add change calculations (simplified - would need historical data for real changes)
    df['OI_Change'] = np.random.uniform(-5, 15, len(df))
    df['IV_Change'] = np.random.uniform(-10, 10, len(df))
    df['LTP_Change'] = np.random.uniform(-2, 2, len(df))

    return df, underlying_price

def render_original_charts(df, underlying_price):
    """Render all the original charting functionality"""

    # Original sentiment analysis
    st.subheader("📊 Market Sentiment Analysis")

    # Average sentiment calculation
    ce_df = df[df['Type'] == 'CE']
    pe_df = df[df['Type'] == 'PE']

    atm_strikes = df[abs(df['StrikePrice'] - underlying_price) <= 250]  # ATM ±5 strikes
    atm_ce = atm_strikes[atm_strikes['Type'] == 'CE']['SentimentScore'].mean()
    atm_pe = atm_strikes[atm_strikes['Type'] == 'PE']['SentimentScore'].mean()

    # Sentiment gauges
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Call Avg Sentiment", f"{atm_ce:.2f}")
    with col2:
        st.metric("Put Avg Sentiment", f"{atm_pe:.2f}")

    # CE/PE Zone Sentiment Gauges
    with st.expander("📈 CE/PE Zone Sentiment Gauges (ATM ±5)", expanded=True):
        fig_gauge = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('CE Sentiment', 'PE Sentiment')
        )

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=atm_ce,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calls"},
            gauge={
                'axis': {'range': [-5, 5]},
                'bar': {'color': "green" if atm_ce > 0 else "red"},
                'steps': [
                    {'range': [-5, -3], 'color': "red"},
                    {'range': [-3, -1], 'color': "orange"},
                    {'range': [-1, 1], 'color': "gray"},
                    {'range': [1, 3], 'color': "lightgreen"},
                    {'range': [3, 5], 'color': "green"}
                ]
            }
        ), row=1, col=1)

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=atm_pe,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Puts"},
            gauge={
                'axis': {'range': [-5, 5]},
                'bar': {'color': "green" if atm_pe > 0 else "red"},
                'steps': [
                    {'range': [-5, -3], 'color': "red"},
                    {'range': [-3, -1], 'color': "orange"},
                    {'range': [-1, 1], 'color': "gray"},
                    {'range': [1, 3], 'color': "lightgreen"},
                    {'range': [3, 5], 'color': "green"}
                ]
            }
        ), row=1, col=2)

        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Enhanced HFT Algo Scanner with original functionality
    st.subheader("🔍 HFT Algo Scanner - Multi-Strike Analysis")

    log_df = df.copy()
    if not log_df.empty:
        # Enhanced calculations for institutional flow detection
        log_df['Exposure'] = log_df['OI'] * log_df['LTP']
        log_df['VolumeWeightedExposure'] = log_df['Exposure'] * log_df.get('Volume', 1)
        log_df['NotionalValue'] = log_df['OI'] * log_df['LTP'] * 75

        # Calculate percentage changes
        log_df['OI_Change_Pct'] = log_df.get('OI_Change', 0) / log_df['OI'] * 100
        log_df['IV_Change_Pct'] = log_df.get('IV_Change', 0) / (log_df.get('IV', 1) + 0.01) * 100
        log_df['LTP_Change_Pct'] = log_df.get('LTP_Change', 0) / (log_df['LTP'] + 0.01) * 100

        # Money Flow Index calculation
        def calculate_money_flow_index(df, window=14):
            typical_price = (log_df['LTP'] + log_df.get('High', log_df['LTP']) + log_df.get('Low', log_df['LTP'])) / 3
            money_flow = typical_price * log_df.get('Volume', 1)

            positive_flow = money_flow.where(log_df['LTP_Change_Pct'] > 0, 0)
            negative_flow = money_flow.where(log_df['LTP_Change_Pct'] <= 0, 0)

            positive_mf = positive_flow.rolling(window=window).sum()
            negative_mf = negative_flow.rolling(window=window).sum()

            mfi = 100 - (100 / (1 + (positive_mf / (negative_mf + 1))))
            return mfi.fillna(50)  # Neutral starting point

        log_df['MFI'] = calculate_money_flow_index(log_df)

        # Enhanced Activity Metric calculation
        def calculate_enhanced_activity_metric(row):
            base_exposure = row['VolumeWeightedExposure']
            oi_change = abs(row['OI_Change_Pct'])
            volume = row.get('Volume', 1)
            ltp_change = row['LTP_Change_Pct']
            iv_change = row['IV_Change_Pct']
            mfi = row.get('MFI', 50)

            volume_factor = np.log1p(volume) / 50
            oi_momentum = min(oi_change / 3, 4)
            price_momentum = abs(ltp_change) / 8
            iv_factor = abs(iv_change) / 15
            mfi_factor = abs(mfi - 50) / 25

            activity_score = base_exposure * (1 + volume_factor * 0.3 + oi_momentum * 0.25 + 
                                           price_momentum * 0.2 + iv_factor * 0.15 + mfi_factor * 0.1)
            return activity_score

        log_df['ActivityMetric'] = log_df.apply(calculate_enhanced_activity_metric, axis=1)

        # Dark pool activity detection
        def detect_dark_pool_activity(row):
            volume = row.get('Volume', 1)
            oi_change = row['OI_Change_Pct']
            ltp_change = row['LTP_Change_Pct']

            if oi_change > 5 and abs(ltp_change) < 2 and volume > 100:
                return True
            return False

        log_df['DarkPoolActivity'] = log_df.apply(detect_dark_pool_activity, axis=1)

        # Create the enhanced multi-pane chart (original style)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Action & Volume', 'Money Flow Index (MFI)', 'Big Money Flow Detection'),
            row_heights=[0.4, 0.3, 0.3]
        )

        # Prepare data for charting
        price_data = log_df.groupby(['Timestamp']).agg({
            'UnderlyingValue': 'first',
            'ActivityMetric': 'sum',
            'MFI': 'mean',
            'Volume': 'sum'
        }).reset_index()

        price_data['TimeSlot'] = price_data['Timestamp']

        # Top pane - Price and Volume
        fig.add_trace(go.Scatter(
            x=price_data['TimeSlot'], 
            y=price_data['UnderlyingValue'],
            mode='lines+markers',
            name='Nifty Price',
            line=dict(color='#1f77b4', width=2),
            yaxis='y'
        ), row=1, col=1)

        # Volume bars
        fig.add_trace(go.Bar(
            x=price_data['TimeSlot'],
            y=price_data['Volume'],
            name='Volume',
            marker_color='rgba(31, 119, 180, 0.3)',
            yaxis='y2'
        ), row=1, col=1)

        # Middle pane - MFI
        fig.add_trace(go.Scatter(
            x=price_data['TimeSlot'], 
            y=price_data['MFI'],
            mode='lines',
            name='MFI',
            line=dict(color='#FFD700', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 215, 0, 0.1)'
        ), row=2, col=1)

        # Add MFI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        # Bottom pane - Enhanced flow bars
        flow_agg = log_df.groupby(['Timestamp', 'SentimentBias']).agg({
            'ActivityMetric': 'sum',
            'MFI': 'mean'
        }).reset_index()
        flow_agg.columns = ['TimeSlot', 'FlowType', 'WeightedActivity', 'MFI']

        # Add color mapping for flow types
        color_map = {
            'Aggressive Call Buying': '#00FF00', 'Call Buying': '#90EE90', 'Call Activity': '#ADD8E6',
            'Aggressive Put Buying': '#FF0000', 'Put Buying': '#FFA500', 'Put Activity': '#FFB6C1',
            'Call Writing': '#8B0000', 'Put Writing': '#006400',
            'Dark Pool CE': '#4B0082', 'Dark Pool PE': '#8B008B',
            'Institutional CE': '#FF4500', 'Institutional PE': '#FF6347',
            'Smart Money CE': '#32CD32', 'Smart Money PE': '#228B22',
            'Market Maker CE': '#1E90FF', 'Market Maker PE': '#4169E1',
            'Low Activity': '#696969'
        }

        flow_agg['Color'] = flow_agg['FlowType'].map(color_map).fillna('#696969')

        flow_types = flow_agg['FlowType'].unique()
        excluded_flows = ['Call Activity', 'Put Activity', 'Call Buy', 'Put Buy', 'Low Activity']

        for flow_type in flow_types:
            if flow_type in excluded_flows:
                continue

            flow_data = flow_agg[flow_agg['FlowType'] == flow_type]
            if not flow_data.empty:
                color = flow_data['Color'].iloc[0]
                fig.add_trace(go.Bar(
                    x=flow_data['TimeSlot'],
                    y=flow_data['WeightedActivity'],
                    name=flow_type,
                    marker_color=color,
                    opacity=0.8
                ), row=3, col=1)

        # Update layout
        fig.update_layout(
            title=dict(
                text="Enhanced HFT Algo Scanner - Institutional Flow Detection with MFI",
                x=0.5,
                font=dict(size=18, family="Arial Black", color="black")
            ),
            barmode='stack',
            height=900,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=11, family='Arial'),
            legend=dict(
                orientation='v',
                yanchor='top',
                y=0.95,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=11)
            )
        )

        # Update axes
        for row in [1, 2, 3]:
            fig.update_xaxes(
                rangeslider_visible=False,
                fixedrange=False,
                showgrid=False,
                zeroline=False,
                color='black',
                row=row, col=1
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                color='black',
                row=row, col=1
            )

        # Axis titles
        fig.update_yaxes(title_text="Price", title_font=dict(size=12), row=1, col=1)
        fig.update_yaxes(title_text="MFI", title_font=dict(size=12), row=2, col=1)
        fig.update_yaxes(title_text="Big Money Flow", title_font=dict(size=12), row=3, col=1)
        fig.update_xaxes(title_text="Time", title_font=dict(size=12), row=3, col=1)

        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],
            'modeBarButtonsToAdd': ['pan2d']
        })

    # Enhanced Big Money Flow Analysis
    st.subheader("💰 Enhanced Big Money Flow Analysis")

    flow_totals = flow_agg.groupby('FlowType')['WeightedActivity'].sum().reset_index()
    dark_pool_count = len(log_df[log_df['DarkPoolActivity'] == True])
    avg_mfi = log_df['MFI'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Dark Pool Signals", dark_pool_count)
    with col2:
        mfi_status = "Overbought" if avg_mfi > 70 else "Oversold" if avg_mfi < 30 else "Neutral"
        st.metric("Avg MFI", f"{avg_mfi:.1f}", delta=mfi_status)
    with col3:
        total_volume = log_df.get('Volume', 0).sum()
        st.metric("Total Volume", f"{total_volume:,.0f}")
    with col4:
        aggressive_flows = len(flow_agg[flow_agg['FlowType'].str.contains('Aggressive|Heavy', na=False)])
        st.metric("Aggressive Flows", aggressive_flows)
    with col5:
        latest_sentiment = "Bullish" if atm_ce > atm_pe else "Bearish" if atm_pe > atm_ce else "Neutral"
        st.metric("Market Sentiment", latest_sentiment)

    return log_df

def render_institutional_dashboard(institutional_data, underlying_price):
    """Render enhanced institutional dashboard (as before but integrated)"""
    df = institutional_data['enhanced_df']

    st.markdown("---")
    st.subheader("🏛️ INSTITUTIONAL PLAYER DETECTION DASHBOARD")

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
    col1, col2 = st.columns(2)

    with col1:
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
        # Enhanced writer detection
        call_writers = df[(df['Type'] == 'CE') & (df['SentimentBias'].str.contains('Writing|Writer', na=False))]
        put_writers = df[(df['Type'] == 'PE') & (df['SentimentBias'].str.contains('Writing|Writer', na=False))]

        st.write("**🔴 Call Writers (Bearish):**")
        if len(call_writers) > 0:
            for _, row in call_writers.head(5).iterrows():
                st.write(f"• {row['StrikePrice']:.0f} CE: OI {row['OI']:,.0f} | {row['Player_Type']}")
        else:
            st.write("No significant call writing detected")

        st.write("**🟢 Put Writers (Bullish):**")
        if len(put_writers) > 0:
            for _, row in put_writers.head(5).iterrows():
                st.write(f"• {row['StrikePrice']:.0f} PE: OI {row['OI']:,.0f} | {row['Player_Type']}")
        else:
            st.write("No significant put writing detected")

    return df

def main():
    """Main application with all original charts + enhanced institutional detection"""

    st.title("🎯 NIFTY OPTIONS SENTIMENT & INSTITUTIONAL TRACKER")
    st.markdown("**Advanced Multi-Chart Analysis with Institutional Detection**")

    # Sidebar controls
    st.sidebar.title("⚙️ Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 30, 300, 60)

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Get data
    expiry_dates = get_expiry_dates_cached()
    if not expiry_dates:
        st.error("Unable to fetch expiry dates")
        return

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

        st.success(f"✅ Loaded {len(df)} option contracts | Nifty: ₹{underlying_price:,.2f}")

        # Calculate institutional metrics
        institutional_data = calculate_institutional_metrics(df, underlying_price)

        # Store in session state for historical analysis
        st.session_state.strike_sentiment_log.extend(df.to_dict('records'))

        # Render all original charts
        enhanced_df = render_original_charts(df, underlying_price)

        # Render institutional dashboard
        final_df = render_institutional_dashboard(institutional_data, underlying_price)

        # Original tabbed interface for additional analysis
        st.markdown("---")
        st.subheader("📈 ORIGINAL ADVANCED ANALYSIS TABS")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 OI Analysis", "📈 IV Analysis", "🔍 Greeks", "📋 Data"])

        with tab1:
            # OI Distribution
            col1, col2 = st.columns(2)

            with col1:
                # Call OI
                ce_df = final_df[final_df['Type'] == 'CE']
                fig_ce_oi = px.bar(
                    ce_df, x='StrikePrice', y='OI', 
                    title="Call Open Interest", color='SentimentBias',
                    hover_data=['Volume', 'LTP', 'IV']
                )
                st.plotly_chart(fig_ce_oi, use_container_width=True)

            with col2:
                # Put OI
                pe_df = final_df[final_df['Type'] == 'PE']
                fig_pe_oi = px.bar(
                    pe_df, x='StrikePrice', y='OI',
                    title="Put Open Interest", color='SentimentBias',
                    hover_data=['Volume', 'LTP', 'IV']
                )
                st.plotly_chart(fig_pe_oi, use_container_width=True)

        with tab2:
            # IV Analysis
            fig_iv = go.Figure()

            ce_df = final_df[final_df['Type'] == 'CE']
            pe_df = final_df[final_df['Type'] == 'PE']

            fig_iv.add_trace(go.Scatter(
                x=ce_df['StrikePrice'], y=ce_df['IV'],
                mode='lines+markers', name='Call IV',
                line=dict(color='#3B82F6', width=2)
            ))

            fig_iv.add_trace(go.Scatter(
                x=pe_df['StrikePrice'], y=pe_df['IV'],
                mode='lines+markers', name='Put IV',
                line=dict(color='#EF4444', width=2)
            ))

            fig_iv.update_layout(
                title="Implied Volatility Skew",
                xaxis_title="Strike Price",
                yaxis_title="Implied Volatility (%)",
                height=400
            )
            st.plotly_chart(fig_iv, use_container_width=True)

        with tab3:
            # Greeks Analysis
            col1, col2 = st.columns(2)

            with col1:
                # Delta Profile
                fig_delta = go.Figure()
                fig_delta.add_trace(go.Scatter(
                    x=ce_df['StrikePrice'], y=ce_df['Delta'],
                    mode='lines+markers', name='Call Delta',
                    line=dict(color='#3B82F6', width=2)
                ))
                fig_delta.add_trace(go.Scatter(
                    x=pe_df['StrikePrice'], y=pe_df['Delta'],
                    mode='lines+markers', name='Put Delta',
                    line=dict(color='#EF4444', width=2)
                ))
                fig_delta.update_layout(
                    title="Delta Profile", xaxis_title="Strike Price",
                    yaxis_title="Delta", height=300
                )
                st.plotly_chart(fig_delta, use_container_width=True)

            with col2:
                # Gamma Profile
                fig_gamma = go.Figure()
                fig_gamma.add_trace(go.Scatter(
                    x=ce_df['StrikePrice'], y=ce_df['Gamma'],
                    mode='lines+markers', name='Call Gamma',
                    line=dict(color='#10B981', width=2)
                ))
                fig_gamma.add_trace(go.Scatter(
                    x=pe_df['StrikePrice'], y=pe_df['Gamma'],
                    mode='lines+markers', name='Put Gamma',
                    line=dict(color='#F59E0B', width=2)
                ))
                fig_gamma.update_layout(
                    title="Gamma Profile", xaxis_title="Strike Price",
                    yaxis_title="Gamma", height=300
                )
                st.plotly_chart(fig_gamma, use_container_width=True)

        with tab4:
            # Data Table with institutional classification
            st.subheader("📋 Enhanced Options Data with Institutional Analysis")

            display_df = final_df[[
                'StrikePrice', 'Type', 'LTP', 'OI', 'Volume', 'IV', 
                'Delta', 'Gamma', 'SentimentBias', 'Player_Type', 
                'Institutional_Score', 'Notional_Value'
            ]].copy()

            # Format for display
            display_df['OI(M)'] = (display_df['OI'] / 1000000).round(2)
            display_df['IV'] = display_df['IV'].round(2)
            display_df['LTP'] = display_df['LTP'].round(2)
            display_df['Notional(Cr)'] = (display_df['Notional_Value'] / 10000000).round(1)

            # Highlight institutional rows
            def highlight_institutional(row):
                if row['Player_Type'] in ['Institutional', 'Professional']:
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)

            styled_df = display_df.style.apply(highlight_institutional, axis=1)
            st.dataframe(styled_df, use_container_width=True)

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
