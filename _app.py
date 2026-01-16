
import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. 설정 및 데이터 처리 클래스
# ==========================================
class TechnicalAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self._load_data()

    def _load_data(self):
        try:
            # 최근 10년 데이터 로드
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*10)

            df = fdr.DataReader(self.ticker, start=start_date.strftime('%Y-%m-%d'))

            if df.empty: return df

            if df.index.name != 'Date':
                df.index.name = 'Date'

            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].replace(0, np.nan).fillna(method='ffill')
            else:
                df['Volume'] = 1000000

            # 오늘 데이터 제외 로직
            if not df.empty:
                last_date = df.index[-1].date()
                today_date = datetime.now().date()
                if last_date == today_date:
                    df = df.iloc[:-1]

            return df
        except Exception:
            return pd.DataFrame()

    def add_indicators(self):
        df = self.df.copy()
        if df.empty: return df

        required = {'Close', 'High', 'Low', 'Volume'}
        if not required.issubset(df.columns): return pd.DataFrame()

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # --- [Group 1] 추세 ---
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        alpha = 1/14
        plus_dm = high.diff()
        minus_dm = low.diff()
        _plus = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        _minus = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        smooth_plus = pd.Series(_plus, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        smooth_minus = pd.Series(_minus, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (smooth_plus / atr)
        minus_di = 100 * (smooth_minus / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()

        # --- [Group 2] 모멘텀 ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        lowest_low = low.rolling(9).min()
        highest_high = high.rolling(9).max()
        df['Stoch_K'] = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(6).mean()

        df['ROC'] = ((close - close.shift(12)) / close.shift(12)) * 100

        # --- [Group 3] 심리 ---
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(14).mean()
        mad = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)

        ema13 = close.ewm(span=13, adjust=False).mean()
        df['Bull_Power'] = high - ema13
        df['Bear_Power'] = low - ema13

        df['MA20_Gap'] = (close / close.rolling(20).mean()) - 1

        # --- [Group 4] 거래량/변동성 ---
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        pos_sum = pd.Series(positive_flow, index=df.index).rolling(14).sum()
        neg_sum = pd.Series(negative_flow, index=df.index).rolling(14).sum()
        mfi_ratio = pos_sum / neg_sum
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))

        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        df['Upper_Band'] = upper_band 
        df['Lower_Band'] = lower_band 
        df['MA20'] = ma20             

        df['Band_Width'] = (upper_band - lower_band) / ma20
        df['Band_Rank'] = df['Band_Width'].rolling(window=120).rank(pct=True) * 100

        return df.dropna()

    def interpret_indicators(self, row):
        signals = {}
        signals['MACD_Sig'] = "매수" if row['MACD'] > row['MACD_Signal'] else "매도"
        signals['ADX_Sig'] = "추세 강함" if row['ADX'] > 25 else "추세 약함"

        if row['RSI'] > 70: signals['RSI_Sig'] = "과매수"
        elif row['RSI'] < 30: signals['RSI_Sig'] = "과매도"
        else: signals['RSI_Sig'] = "중립"

        if row['Stoch_K'] > 80: signals['Stoch_Sig'] = "과매수"
        elif row['Stoch_K'] < 20: signals['Stoch_Sig'] = "과매도"
        else: signals['Stoch_Sig'] = "중립"

        signals['ROC_Sig'] = "매수" if row['ROC'] > 0 else "매도"

        if row['CCI'] > 100: signals['CCI_Sig'] = "매수"
        elif row['CCI'] < -100: signals['CCI_Sig'] = "매도"
        else: signals['CCI_Sig'] = "중립"

        if row['Bull_Power'] > 0 and row['Bear_Power'] > 0: signals['Power_Sig'] = "매수(강세)"
        elif row['Bull_Power'] < 0 and row['Bear_Power'] < 0: signals['Power_Sig'] = "매도(약세)"
        else: signals['Power_Sig'] = "중립"

        if row['MA20_Gap'] > 0.05: signals['MA_Gap_Sig'] = "과열"
        elif row['MA20_Gap'] < -0.05: signals['MA_Gap_Sig'] = "침체"
        else: signals['MA_Gap_Sig'] = "중립"

        if row['MFI'] > 80: signals['MFI_Sig'] = "과열(유입)"
        elif row['MFI'] < 20: signals['MFI_Sig'] = "침체(유출)"
        else: signals['MFI_Sig'] = "중립"

        if row['Band_Rank'] < 20: signals['Band_Width_Sig'] = "스퀴즈(응축)"
        elif row['Band_Rank'] > 80: signals['Band_Width_Sig'] = "변동성 폭발"
        else: signals['Band_Width_Sig'] = "보통"

        return pd.Series(signals)

# ==========================================
# 2. 메인 앱 로직
# ==========================================
st.set_page_config(page_title="AI Market Similarity", layout="wide")

# CSS 스타일 적용
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }

    /* 차트 컨테이너 및 3번 섹션 박스 스타일 */
    /* [수정] text-align: center를 추가하여 내부 텍스트 및 요소 자동 가운데 정렬 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 15px !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        background-color: white;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center !important; 
    }

    .chart-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 5px;
        margin-top: 5px;
        padding-left: 5px;
        text-align: left; /* 4번 차트 제목은 왼쪽 정렬 유지 */
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 기술적 분석 (Prototype)")

threshold = 10
holding_days = 5

@st.cache_data
def get_stock_list():
    df_krx = fdr.StockListing('KOSPI')
    if 'Marcap' in df_krx.columns:
        df_krx = df_krx.sort_values(by='Marcap', ascending=False)
    return dict(zip(df_krx['Name'], df_krx['Code']))

@st.cache_data(ttl=3600)
def run_analysis(ticker):
    analyzer = TechnicalAnalyzer(ticker)
    if analyzer.df.empty: return None

    df = analyzer.add_indicators()
    if df is None or df.empty: return None

    signal_df = df.apply(analyzer.interpret_indicators, axis=1)
    full_df = pd.concat([df, signal_df], axis=1)
    return full_df

try:
    stock_map = get_stock_list()
    stock_names = list(stock_map.keys()) 

    # 1. 종목 선택
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_name = st.selectbox(
            "종목 선택 (시가총액 내림차순)", 
            options=stock_names, 
            index=stock_names.index('삼성전자') if '삼성전자' in stock_names else 0
        )

    ticker_code = stock_map[selected_name]

    # 2. 데이터 분석 실행
    with st.spinner(f"'{selected_name}' 정밀 금융 분석 중..."):
        full_df = run_analysis(ticker_code)

        if full_df is None or full_df.empty:
            st.error("데이터 로드 실패.")
            st.stop()

        today_row = full_df.iloc[-1]

        # [순서 및 용어 통일]
        sig_cols = [
            'MACD_Sig',         # 1. MACD
            'CCI_Sig',          # 2. CCI
            'ADX_Sig',          # 3. ADX
            'Power_Sig',        # 4. Bull/Bear Power
            'RSI_Sig',          # 5. RSI
            'MA_Gap_Sig',       # 6. MA Gap
            'Stoch_Sig',        # 7. Stochastic
            'MFI_Sig',          # 8. MFI
            'ROC_Sig',          # 9. ROC
            'Band_Width_Sig'    # 10. Bollinger Band Width
        ]

        today_signals = today_row[sig_cols]
        last_date = pd.to_datetime(today_row.name).strftime('%Y-%m-%d')

        # ---------------------------------------------------------
        # [통계 미리 계산]
        # ---------------------------------------------------------
        past_signals = full_df[sig_cols].iloc[:-1]
        matches = (past_signals == today_signals).sum(axis=1)
        similar_days_raw = full_df.iloc[:-1].loc[matches >= threshold].copy()

        selected_indices = []
        last_selected_date = None
        similar_days_raw = similar_days_raw.sort_index()

        for idx in similar_days_raw.index:
            current_date = idx
            if last_selected_date is None or (current_date - last_selected_date).days >= holding_days:
                selected_indices.append(idx)
                last_selected_date = current_date

        similar_days = similar_days_raw.loc[selected_indices]

        calc_win_rate = 0.0
        calc_avg_return = 0.0
        calc_count = len(similar_days)

        if calc_count > 0:
            wins = 0
            total_return = 0.0
            for idx in similar_days.index:
                loc_idx = full_df.index.get_loc(idx)
                future_loc = loc_idx + holding_days
                if future_loc < len(full_df):
                    fut_price = full_df.iloc[future_loc]['Close']
                    cur_price = full_df.iloc[loc_idx]['Close']
                    ret = (fut_price - cur_price) / cur_price
                    total_return += ret
                    if fut_price > cur_price:
                        wins += 1
            calc_win_rate = (wins / calc_count) * 100
            calc_avg_return = (total_return / calc_count) * 100

    # 3. [요약 멘트]
    summary_bg = "#e8f0fe"
    summary_border = "#d2e3fc"
    win_color = "#d62728" if calc_win_rate >= 50 else "#1f77b4"
    ret_color = "#d62728" if calc_avg_return > 0 else "#1f77b4"

    st.markdown(f"""
    <div style="background-color: {summary_bg}; padding: 15px; border-radius: 10px; border: 1px solid {summary_border}; margin-top: 10px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 1.2rem; color: #222222; line-height: 1.6;"> 
            ✨ <b>과거 패턴 분석 요약</b><br>
            최근 10년 기술적 지표가 유사했던 날은 총 <b>{calc_count}일</b> 포착되었습니다.<br>
            해당 시점들의 {holding_days}일 후 평균적으로 상승했던 비율은 <b style="color: {win_color};">{calc_win_rate:.1f}%</b>, 
            수익률은 <b style="color: {ret_color};">{calc_avg_return:+.1f}%</b>입니다.<br>
            <span style="font-size: 0.85rem; color: #555555;">(※ 이는 과거 통계일 뿐이며, 미래 수익을 보장하지 않습니다.)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # SECTION 1: 과거 유사 패턴 백테스팅
    # ==========================================
    # st.markdown("---")
    # st.subheader(f"최근 10년 분석")

    if calc_count > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("기술적 지표 유사했던 날", f"{calc_count}건")
        c2.metric(f"{holding_days}일 후 상승한 비율", f"{calc_win_rate:.1f}%")
        c3.markdown(f"""
        <div style="text-align: center;">
            <p style="margin-bottom: 0px; font-size: 0.8rem;">{holding_days}일 후 평균 수익률</p>
            <p style="font-size: 2rem; font-weight: bold; color: {ret_color}; margin: 0;">{calc_avg_return:+.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"현재 10개 지표가 모두 일치하는 과거 사례가 없습니다. (Threshold: 10/10)")

    # # ==========================================
    # # SECTION 2: 전체 기간 차트
    # # ==========================================
    # st.markdown("---")
    # st.subheader("2. 전체 기간 차트 (Full History)")

    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(x=full_df.index, y=full_df['Close'], mode='lines', name='주가', line=dict(color='#cccccc', width=1.5)))
    if not similar_days.empty:
        fig_full.add_trace(go.Scatter(x=similar_days.index, y=similar_days['Close'], mode='markers', name='유사패턴 발생일', marker=dict(color='#d62728', size=6, symbol='circle', opacity=0.8)))

    fig_full.update_layout(
        title=dict(text=f"{selected_name} 유사패턴 발생일", font=dict(size=15)),
        template="plotly_white", height=400,
        showlegend=True,
        xaxis=dict(fixedrange=True, title=None, tickformat="%Y"),
        yaxis=dict(fixedrange=True, tickformat=","),
        dragmode=False
    )
    with st.container(border=True):
        st.plotly_chart(fig_full, use_container_width=True, config={'staticPlot': True})

    # ==========================================
    # SECTION 3: 오늘의 10대 지표 정밀 진단
    # ==========================================
    st.markdown("---")
    st.markdown(f"### 기술적 분석 지표 진단 (기준일: {last_date} | 주가: {today_row['Close']:,.0f}원)")

    cols = st.columns(5)

    # [설정] 통합 지표 정의 (이름 + 툴팁)
    indicator_defs = {
        'MACD_Sig': {
            'name': 'MACD', 
            'tip': "MACD선(파랑)이 시그널선(주황)을 상향 돌파하면 매수(골든크로스), 하향 돌파하면 매도(데드크로스) 신호입니다. 0선 위는 상승 추세 구간입니다."
        },
        'CCI_Sig': {
            'name': 'CCI (Commodity Channel Index)', 
            'tip': "주가 평균과 현재 주가의 편차입니다. +100 이상 과매수, -100 이하 과매도입니다. 0선 돌파를 추세 전환으로 보기도 합니다."
        },
        'ADX_Sig': {
            'name': 'ADX (Trend Strength)', 
            'tip': "현재 추세의 강도를 나타냅니다. 25 이상이면 강한 추세장(상승이든 하락이든), 20 이하면 추세가 없는 횡보장입니다."
        },
        'Power_Sig': {
            'name': 'Bull/Bear Power', 
            'tip': "매수(Bull)와 매도(Bear) 세력의 강도입니다. 0선 위면 해당 세력이 우세함을 의미합니다. 둘 다 양수면 강한 상승장입니다."
        },
        'RSI_Sig': {
            'name': 'RSI (Relative Strength)', 
            'tip': "상대강도지수입니다. 70 이상은 과매수(매도 검토), 30 이하는 과매도(매수 검토) 구간입니다. 50을 기준으로 추세 힘을 가늠합니다."
        },
        'MA_Gap_Sig': {
            'name': 'MA Gap (이격도)', 
            'tip': "주가와 20일 이동평균선 간의 차이입니다. +0.05(+5%) 이상이면 단기 과열, -0.05(-5%) 이하면 단기 낙폭 과대로 반등 가능성이 있습니다."
        },
        'Stoch_Sig': {
            'name': 'Stochastic Oscillator', 
            'tip': "주가의 상대적 위치를 나타냅니다. 80 이상 과매수, 20 이하 과매도입니다. %K(파랑)가 %D(주황)를 상향 돌파하면 매수 신호입니다."
        },
        'MFI_Sig': {
            'name': 'MFI (Money Flow Index)', 
            'tip': "거래량을 고려한 RSI입니다. 자금의 유입/유출을 봅니다. 80 이상 과열(매도), 20 이하 침체(매수) 구간입니다."
        },
        'ROC_Sig': {
            'name': 'ROC (Rate of Change)', 
            'tip': "가격 변화율입니다. 0선 위면 상승 모멘텀, 아래면 하락 모멘텀을 의미합니다. 0선을 상향 돌파할 때가 매수 포인트입니다."
        },
        'Band_Width_Sig': {
            'name': 'Bollinger Band Width', 
            'tip': "볼린저 밴드의 폭(너비)입니다. 수치가 낮아지면(Squeeze) 에너지가 응축된 상태로, 곧 위든 아래든 큰 변동성이 터질 것임을 예고합니다."
        }
    }

    for i, col in enumerate(sig_cols):
        status = today_signals[col]
        info = indicator_defs.get(col, {'name': col, 'tip': ''})

        if status in ['매수', '매수(강세)', '과매수', '과열', '추세 강함', '변동성 폭발', '과열(유입)']: color = "#d62728"
        elif status in ['매도', '매도(약세)', '과매도', '침체', '추세 약함', '스퀴즈(응축)', '침체(유출)']: color = "#1f77b4"
        else: color = "#666666"

        with cols[i % 5]:
            # [수정] st.container() + st.markdown(help=...) 방식으로 변경하여 물음표 아이콘 자동 생성
            with st.container(border=True):
                st.markdown(f"**{info['name']}**", help=info['tip'])
                st.markdown(f"<div style='color:{color}; font-weight:bold; font-size:15px; margin-top:5px;'>{status}</div>", unsafe_allow_html=True)

    # ==========================================
    # SECTION 4: 최근 6개월 기술적 지표 정밀 분석
    # # ==========================================
    # st.subheader("4. 최근 6개월 기술적 지표 정밀 분석 (10대 지표)")
    # st.caption("각 지표 제목에 마우스를 올리면 상세 해석 방법(Tooltip)이 표시됩니다.")

    six_months_ago = full_df.index[-1] - timedelta(days=180)
    df_recent = full_df[full_df.index >= six_months_ago]

    def create_chart(height=250):
        fig = go.Figure()
        fig.update_layout(
            height=height,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(fixedrange=True, showticklabels=True, showgrid=False, tickformat="%y-%m-%d", title=None),
            yaxis=dict(fixedrange=True, showgrid=True, gridcolor='#f5f5f5', tickformat=","),
            dragmode=False
        )
        return fig

    # [Chart 0] Price & Bollinger (Visual Reference) - Extra Chart
    with st.container(border=True):
        st.markdown("#### Price & Bollinger", help="주가가 밴드 상단 돌파 시 과매수, 하단 이탈 시 과매도로 간주합니다. 밴드 폭이 좁아지면 조만간 큰 변동성이 올 수 있습니다.")
        fig1 = create_chart(height=350)
        fig1.add_trace(go.Candlestick(x=df_recent.index, open=df_recent['Open'], high=df_recent['High'], low=df_recent['Low'], close=df_recent['Close'], name='주가'))
        fig1.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Upper_Band'], line=dict(color='gray', width=1), showlegend=False))
        fig1.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Lower_Band'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(200,200,200,0.1)', showlegend=False))
        st.plotly_chart(fig1, use_container_width=True, config={'staticPlot': True})

    # [Chart 1] MACD
    col_key = 'MACD_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig2 = create_chart()
        fig2.add_trace(go.Bar(x=df_recent.index, y=df_recent['MACD_Hist'], marker_color='gray', name='MACD Hist'))
        fig2.add_trace(go.Scatter(x=df_recent.index, y=df_recent['MACD'], line=dict(color='blue', width=1), name='MACD'))
        fig2.add_trace(go.Scatter(x=df_recent.index, y=df_recent['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'))
        st.plotly_chart(fig2, use_container_width=True, config={'staticPlot': True})

    # [Chart 2] CCI
    col_key = 'CCI_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig7 = create_chart()
        fig7.add_trace(go.Scatter(x=df_recent.index, y=df_recent['CCI'], line=dict(color='brown', width=1), name='CCI'))
        fig7.add_hline(y=100, line_dash="dash", line_color="red"); fig7.add_hline(y=-100, line_dash="dash", line_color="green")
        st.plotly_chart(fig7, use_container_width=True, config={'staticPlot': True})

    # [Chart 3] ADX
    col_key = 'ADX_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig3 = create_chart()
        fig3.add_trace(go.Scatter(x=df_recent.index, y=df_recent['ADX'], line=dict(color='black', width=1), name='ADX'))
        fig3.add_hline(y=25, line_dash="dot", line_color="red")
        st.plotly_chart(fig3, use_container_width=True, config={'staticPlot': True})

    # [Chart 4] Power
    col_key = 'Power_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig8 = create_chart()
        fig8.add_trace(go.Bar(x=df_recent.index, y=df_recent['Bull_Power'], marker_color='green', name='Bull'))
        fig8.add_trace(go.Bar(x=df_recent.index, y=df_recent['Bear_Power'], marker_color='red', name='Bear'))
        st.plotly_chart(fig8, use_container_width=True, config={'staticPlot': True})

    # [Chart 5] RSI
    col_key = 'RSI_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig4 = create_chart()
        fig4.add_trace(go.Scatter(x=df_recent.index, y=df_recent['RSI'], line=dict(color='purple', width=1), name='RSI'))
        fig4.add_hline(y=70, line_dash="dash", line_color="red"); fig4.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig4, use_container_width=True, config={'staticPlot': True})

    # [Chart 6] MA Gap
    col_key = 'MA_Gap_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig9 = create_chart()
        fig9.add_trace(go.Scatter(x=df_recent.index, y=df_recent['MA20_Gap'], line=dict(color='navy', width=1), name='MA Gap'))
        fig9.add_hline(y=0.05, line_dash="dot", line_color="red"); fig9.add_hline(y=-0.05, line_dash="dot", line_color="green")
        st.plotly_chart(fig9, use_container_width=True, config={'staticPlot': True})

    # [Chart 7] Stochastic
    col_key = 'Stoch_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig5 = create_chart()
        fig5.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Stoch_K'], line=dict(color='blue', width=1), name='K'))
        fig5.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Stoch_D'], line=dict(color='orange', width=1), name='D'))
        fig5.add_hline(y=80, line_dash="dash", line_color="red"); fig5.add_hline(y=20, line_dash="dash", line_color="green")
        st.plotly_chart(fig5, use_container_width=True, config={'staticPlot': True})

    # [Chart 8] MFI
    col_key = 'MFI_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig10 = create_chart()
        fig10.add_trace(go.Scatter(x=df_recent.index, y=df_recent['MFI'], line=dict(color='green', width=1), name='MFI'))
        fig10.add_hline(y=80, line_dash="dash", line_color="red"); fig10.add_hline(y=20, line_dash="dash", line_color="green")
        st.plotly_chart(fig10, use_container_width=True, config={'staticPlot': True})

    # [Chart 9] ROC
    col_key = 'ROC_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig6 = create_chart()
        fig6.add_trace(go.Scatter(x=df_recent.index, y=df_recent['ROC'], line=dict(color='teal', width=1), name='ROC'))
        fig6.add_hline(y=0, line_color="black")
        st.plotly_chart(fig6, use_container_width=True, config={'staticPlot': True})

    # [Chart 10] Band Width
    col_key = 'Band_Width_Sig'
    with st.container(border=True):
        st.markdown(f"#### {indicator_defs[col_key]['name']}", help=indicator_defs[col_key]['tip'])
        fig11 = create_chart() 
        fig11.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Band_Width'], line=dict(color='magenta', width=1), name='Band Width'))
        st.plotly_chart(fig11, use_container_width=True, config={'staticPlot': True})

except Exception as e:
    st.error(f"Error: {e}")
