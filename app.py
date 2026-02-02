import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from google import genai 
import truststore, certifi, requests, time
from google.genai import types 
import FinanceDataReader as fdr

truststore.inject_into_ssl()

# ==========================================
# 1. 설정 및 데이터 처리 클래스 (변경됨)
# ==========================================
class TechnicalAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self._load_data()

    def _load_data(self):
        try:
            # 10년 전 날짜 계산
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            # fdr을 사용하여 데이터 로드
            df = fdr.DataReader(self.ticker, start=start_date)
            
            if df.empty: return df

            # 컬럼 이름 표준화
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 0인 거래량 처리
            df['Volume'] = df['Volume'].replace(0, np.nan).ffill()
            df['Volume'] = df['Volume'].fillna(1000000)

            # 오늘 데이터 제외 로직
            if not df.empty:
                last_date = df.index[-1].date()
                today_date = datetime.now().date()
                if last_date == today_date:
                    df = df.iloc[:-1]
            return df
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()

    def add_indicators(self):
        df = self.df.copy()
        if df.empty or len(df) < 60: return pd.DataFrame()

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # --- [Group 1] 추세 및 모멘텀 (8대 이진 지표) ---
        
        # 1. Sig_MA: 종가 > 20일 이평선
        ma20 = close.rolling(window=20).mean()
        df['Sig_MA'] = np.where(close > ma20, 1, 0)
        df['MA20'] = ma20 # 시각화용

        # 2. Sig_DMI: PDI > MDI
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr14)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr14)
        df['Sig_DMI'] = np.where(plus_di > minus_di, 1, 0)
        df['Plus_DI'] = plus_di # 시각화용
        df['Minus_DI'] = minus_di # 시각화용

        # 3. Sig_RSI: RSI > 50
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        df['Sig_RSI'] = np.where(rsi > 50, 1, 0)
        df['RSI'] = rsi # 시각화용

        # 4. Sig_CCI: CCI > 0
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mad)
        df['Sig_CCI'] = np.where(cci > 0, 1, 0)
        df['CCI'] = cci # 시각화용

        # 5. Sig_BB: 볼린저밴드 폭 확대 (현재폭 > 20일 평균폭)
        std20 = close.rolling(20).std()
        bb_width = (std20 * 4) / ma20
        df['Sig_BB'] = np.where(bb_width > bb_width.rolling(20).mean(), 1, 0)
        df['Upper_Band'] = ma20 + (std20 * 2) # 시각화용
        df['Lower_Band'] = ma20 - (std20 * 2) # 시각화용
        df['Band_Width'] = bb_width # 시각화용

        # 6. Sig_ATR: 변동성(에너지) 확대 (현재 ATR > 20일 평균 ATR)
        df['Sig_ATR'] = np.where(atr14 > atr14.rolling(20).mean(), 1, 0)
        df['ATR'] = atr14 # 시각화용

        # 7. Sig_OBV: OBV > OBV_MA20
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['Sig_OBV'] = np.where(obv > obv.rolling(20).mean(), 1, 0)
        df['OBV'] = obv # 시각화용

        # 8. Sig_MFI: MFI > 50
        money_flow = tp * volume
        pos_mf = pd.Series(np.where(tp > tp.shift(1), money_flow, 0), index=df.index).rolling(14).sum()
        neg_mf = pd.Series(np.where(tp < tp.shift(1), money_flow, 0), index=df.index).rolling(14).sum()
        mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
        df['Sig_MFI'] = np.where(mfi > 50, 1, 0)
        df['MFI'] = mfi # 시각화용

        return df.dropna()

    def interpret_indicators(self, row):
        # 1: 긍정(강세/확대), 0: 부정(약세/축소)
        signals = {}
        mapping = {
            'Sig_MA': ('상향', '하향'),
            'Sig_DMI': ('매수우위', '매도우위'),
            'Sig_RSI': ('강세', '약세'),
            'Sig_CCI': ('상승추세', '하락추세'),
            'Sig_BB': ('변동성확대', '변동성축소'),
            'Sig_ATR': ('에너지강화', '에너지약화'),
            'Sig_OBV': ('수급개선', '수급악화'),
            'Sig_MFI': ('자금유입', '자금유출')
        }
        for col, (pos, neg) in mapping.items():
            signals[col] = pos if row[col] == 1 else neg
        return pd.Series(signals)

# ==========================================
# 2. Gemini AI 생성 함수
# ==========================================
def get_ai_diagnosis(api_key, stock_name, current_price, indicators, signals):
    try:
        client = genai.Client(api_key=api_key,
        http_options=types.HttpOptions(
        client_args={
            "verify": certifi.where(),
            "trust_env": True,
        })
        )

        prompt = f"""
        당신은 전문 주식 기술적 분석가입니다. 아래 데이터를 바탕으로 '{stock_name}' 종목에 대한 기술적 분석 요약을 작성해주세요.

        [기본 정보]
        - 현재가: {current_price:,.0f}원

        [8대 이진 보조지표 상태]
        1. 이동평균선(20일): {signals['Sig_MA']}
        2. DMI(추세): {signals['Sig_DMI']}
        3. RSI(모멘텀): {signals['Sig_RSI']}
        4. CCI(방향성): {signals['Sig_CCI']}
        5. 볼린저밴드폭: {signals['Sig_BB']}
        6. ATR(변동성 에너지): {signals['Sig_ATR']}
        7. OBV(수급): {signals['Sig_OBV']}
        8. MFI(자금유입): {signals['Sig_MFI']}

        [요청사항]
        - 현재 기술적 지표들이 가리키는 전반적인 추세와 매수/매도 관점의 통찰을 제공하세요.
        - 최대 2문장의 자연스러운 한국어 평문으로 간결하게 출력하세요. 
        - 서두 인사말은 생략하고 바로 본론만 말하세요.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash", # 최신 모델로 유지
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI 분석 생성 실패: {str(e)}"

# ==========================================
# 3. 메인 앱 로직
# ==========================================
st.set_page_config(page_title="AI Market Similarity", layout="wide")

gemini_api_key = st.secrets["GEMINI_API_KEY"]

# CSS 스타일 적용 (기존 유지)
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
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
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 기술적 분석 (Binary Prototype)")

threshold = 8 # 8개 지표 일치 여부 확인
holding_days = 5

@st.cache_data
def get_stock_list():
    try:
        df_krx = pd.read_csv('KOSPI_filtered.csv')
        if 'Name' in df_krx.columns and 'Code' in df_krx.columns:
            df_krx['Code'] = df_krx['Code'].astype(str).str.zfill(6)
            if 'Marcap' in df_krx.columns:
                df_krx = df_krx.sort_values(by='Marcap', ascending=False)
            return dict(zip(df_krx['Name'], df_krx['Code']))
        return {}
    except FileNotFoundError:
        st.error("'KOSPI_filtered.csv' 파일을 찾을 수 없습니다.")
        return {}

@st.cache_data(ttl=3600)
def run_analysis(ticker):
    analyzer = TechnicalAnalyzer(ticker)
    if analyzer.df.empty: return None
    df = analyzer.add_indicators()
    if df.empty: return None
    signal_df = df.apply(analyzer.interpret_indicators, axis=1)
    full_df = pd.concat([df, signal_df], axis=1)
    return full_df

stock_map = get_stock_list()

if stock_map:
    stock_names = list(stock_map.keys()) 
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_name = st.selectbox(
            "종목 선택 (시가총액 내림차순)", 
            options=stock_names, 
            index=stock_names.index('삼성전자') if '삼성전자' in stock_names else 0
        )
    ticker_code = stock_map[selected_name]

    with st.spinner(f"'{selected_name}' 정밀 금융 분석 중..."):
        full_df = run_analysis(ticker_code)
        if full_df is None or full_df.empty:
            st.error("데이터 로드 실패.")
            st.stop()

        today_row = full_df.iloc[-1]
        # 지표 컬럼 리스트 (8개)
        sig_cols = ['Sig_MA', 'Sig_DMI', 'Sig_RSI', 'Sig_CCI', 'Sig_BB', 'Sig_ATR', 'Sig_OBV', 'Sig_MFI']
        
        today_signals = today_row[sig_cols]
        last_date = pd.to_datetime(today_row.name).strftime('%Y-%m-%d')

        # [통계 계산]
        past_signals = full_df[sig_cols].iloc[:-1]
        matches = (past_signals == today_signals).sum(axis=1)
        similar_days_raw = full_df.iloc[:-1].loc[matches >= threshold].copy()

        selected_indices = []
        last_selected_date = None
        similar_days_raw = similar_days_raw.sort_index()

        for idx in similar_days_raw.index:
            if last_selected_date is None or (idx - last_selected_date).days >= holding_days:
                selected_indices.append(idx)
                last_selected_date = idx

        similar_days = similar_days_raw.loc[selected_indices]
        calc_win_rate = 0.0; calc_avg_return = 0.0; calc_count = len(similar_days)

        if calc_count > 0:
            wins = 0; total_return = 0.0
            for idx in similar_days.index:
                loc_idx = full_df.index.get_loc(idx)
                future_loc = loc_idx + holding_days
                if future_loc < len(full_df):
                    ret = (full_df.iloc[future_loc]['Close'] - full_df.iloc[loc_idx]['Close']) / full_df.iloc[loc_idx]['Close']
                    total_return += ret
                    if ret > 0: wins += 1
            calc_win_rate = (wins / calc_count) * 100
            calc_avg_return = (total_return / calc_count) * 100

    # 3. [요약 멘트] (기존 디자인 유지)
    summary_bg = "#e8f0fe"; summary_border = "#d2e3fc"
    win_color = "#d62728" if calc_win_rate >= 50 else "#1f77b4"
    ret_color = "#d62728" if calc_avg_return > 0 else "#1f77b4"

    st.markdown(f"""
    <div style="background-color: {summary_bg}; padding: 15px; border-radius: 10px; border: 1px solid {summary_border}; margin-top: 10px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 1.2rem; color: #222222; line-height: 1.6;"> 
             <b>📈 과거 패턴 분석 요약</b><br>
            최근 10년 기술적 지표가 <b>{threshold}개 모두 동일했던 날</b>은 총 <b>{calc_count}일</b> 포착되었습니다.<br>
            해당 시점들의 {holding_days}일 후 평균 상승 확률은 <b style="color: {win_color};">{calc_win_rate:.1f}%</b>, 
            평균 수익률은 <b style="color: {ret_color};">{calc_avg_return:+.1f}%</b>입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 1: 과거 유사 패턴 백테스팅
    c1, c2, c3 = st.columns(3)
    c1.metric("유사 패턴 발생 건수", f"{calc_count}건")
    c2.metric(f"{holding_days}일 후 상승 확률", f"{calc_win_rate:.1f}%")
    c3.markdown(f"""<div style="text-align: center;"><p style="margin-bottom: 0px; font-size: 0.8rem;">{holding_days}일 후 평균 수익률</p>
    <p style="font-size: 2rem; font-weight: bold; color: {ret_color}; margin: 0;">{calc_avg_return:+.2f}%</p></div>""", unsafe_allow_html=True)

    # SECTION 2: 차트 및 테이블
    col_chart, col_table = st.columns([3, 1])
    
    with col_chart:
        lookback_days = 10  
        df_recent_history = full_df.iloc[-lookback_days:]
        fig_projection = go.Figure()
        current_close = today_row['Close']
        future_movements = [] 
        last_date_obj = pd.to_datetime(today_row.name)
        future_dates = [last_date_obj + timedelta(days=i) for i in range(0, holding_days + 1)]

        if not similar_days.empty:
            for idx in similar_days.index:
                loc_idx = full_df.index.get_loc(idx)
                if loc_idx + holding_days < len(full_df):
                    past_segment = full_df.iloc[loc_idx : loc_idx + holding_days + 1]['Close']
                    base_price_past = full_df.iloc[loc_idx]['Close']
                    rebased_segment = (past_segment.values / base_price_past) * current_close
                    future_movements.append(rebased_segment)
                    fig_projection.add_trace(go.Scatter(x=future_dates, y=rebased_segment, mode='lines', 
                        line=dict(color='rgba(200, 200, 200, 0.4)', width=1), showlegend=False, hoverinfo='skip'))

        fig_projection.add_trace(go.Scatter(x=df_recent_history.index, y=df_recent_history['Close'], mode='lines', name='최근 주가', line=dict(color='black', width=2)))

        if future_movements:
            avg_path = np.round(np.mean(future_movements, axis=0), 0)
            avg_color = '#d62728' if calc_avg_return > 0 else '#1f77b4'
            fig_projection.add_trace(go.Scatter(x=future_dates, y=avg_path, mode='lines+markers', name='예상 평균', line=dict(color=avg_color, width=3, dash='dot')))
            fig_projection.add_annotation(x=future_dates[-1], y=avg_path[-1], text=f"{calc_avg_return:+.2f}%", showarrow=True, arrowhead=1, ax=35, ay=-30, font=dict(color=avg_color, size=13, weight='bold'))

        fig_projection.update_layout(title=dict(text=f"<b>과거 유사 패턴 매칭 ({threshold}개 지표 일치)</b>", font=dict(size=18)), template="plotly_white", height=400, margin=dict(l=10, r=10, t=40, b=10), showlegend=True, xaxis=dict(tickformat="%m-%d"), hovermode="x unified")
        st.plotly_chart(fig_projection, use_container_width=True)

    with col_table:
        st.markdown(f"<div style='margin-top: 10px; font-weight:bold; font-size:1.05rem;'>유사 시점 목록</div>", unsafe_allow_html=True)
        if not similar_days.empty:
            records = []
            for idx in similar_days.index:
                loc_idx = full_df.index.get_loc(idx)
                if loc_idx + holding_days < len(full_df):
                    ret = (full_df.iloc[loc_idx + holding_days]['Close'] - full_df.iloc[loc_idx]['Close']) / full_df.iloc[loc_idx]['Close']
                    records.append({"발생일": idx.strftime("%Y-%m-%d"), "비고": f"{ret:+.2%}"})
            df_table = pd.DataFrame(records).sort_values(by="발생일", ascending=False)
            st.dataframe(df_table, use_container_width=True, height=350, hide_index=True)

    # SECTION 3: 오늘의 지표 정밀 진단
    st.markdown("---")
    st.markdown(f"### 기술적 분석 지표 진단 (기준일: {last_date})")

    if gemini_api_key:
        with st.spinner("🤖 AI 진단 생성 중..."):
            ai_comment = get_ai_diagnosis(gemini_api_key, selected_name, today_row['Close'], today_row, today_row)
            st.markdown(f"""<div style="background-color: #f1f8e9; padding: 15px; border-radius: 10px; border: 1px solid #c5e1a5; margin-bottom: 20px;">
                <h4 style="margin-top:0; color: #33691e;">✨ AI 기술적 진단</h4>
                <p style="margin: 0; font-size: 1.2rem; color: #333333; line-height: 1.6;">{ai_comment}</p></div>""", unsafe_allow_html=True)

    # 지표 카드 레이아웃 (새로운 8개 지표 반영)
    indicator_defs = {
        'Sig_MA': {'name': '20일 이평선', 'tip': '주가가 20일 이동평균선 위에 있는지 여부'},
        'Sig_DMI': {'name': 'DMI (추세)', 'tip': 'PDI가 MDI보다 커서 상승 에너지가 우위에 있는지 여부'},
        'Sig_RSI': {'name': 'RSI (강도)', 'tip': 'RSI 지수가 50을 상회하여 매수세가 강한지 여부'},
        'Sig_CCI': {'name': 'CCI (방향)', 'tip': 'CCI가 0을 상회하여 주가 평균 대비 강세인지 여부'},
        'Sig_BB': {'name': 'BB폭 (변동성)', 'tip': '볼린저 밴드 너비가 평균보다 넓어져 변동성이 확대 중인지 여부'},
        'Sig_ATR': {'name': 'ATR (에너지)', 'tip': '변동성 수치가 평균보다 높아 주가 움직임이 활발한지 여부'},
        'Sig_OBV': {'name': 'OBV (수급)', 'tip': '거래량 기반 OBV가 평균 위에서 수급이 개선 중인지 여부'},
        'Sig_MFI': {'name': 'MFI (자금흐름)', 'tip': '거래량을 고려한 RSI인 MFI가 50을 상회하는지 여부'}
    }

    cols_card = st.columns(4)
    for i, col in enumerate(sig_cols):
        status_text = today_row[col] # '상향', '하향' 등 문자열
        is_positive = today_row[col.replace('Sig_', 'Sig_')] == 1 # 이진값 확인
        color = "#d62728" if is_positive else "#1f77b4"
        info = indicator_defs.get(col, {'name': col, 'tip': ''})
        with cols_card[i % 4]:
            with st.container(border=True):
                st.markdown(f"**{info['name']}**", help=info['tip'])
                st.markdown(f"<div style='color:{color}; font-weight:bold; font-size:15px; margin-top:5px;'>{status_text}</div>", unsafe_allow_html=True)

    # SECTION 4: 차트 시각화
    st.markdown("---")
    six_months_ago = full_df.index[-1] - timedelta(days=180)
    df_recent = full_df[full_df.index >= six_months_ago]

    def create_chart(height=250):
        fig = go.Figure()
        fig.update_layout(height=height, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickformat="%y-%m-%d"), yaxis=dict(tickformat=","))
        return fig

    # 1. Price & Bollinger
    with st.container(border=True):
        st.markdown("#### Price & Bollinger")
        fig1 = create_chart(height=350)
        fig1.add_trace(go.Candlestick(x=df_recent.index, open=df_recent['Open'], high=df_recent['High'], low=df_recent['Low'], close=df_recent['Close']))
        fig1.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Upper_Band'], line=dict(color='gray', width=1)))
        fig1.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Lower_Band'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'))
        st.plotly_chart(fig1, use_container_width=True)

    # 2. RSI & MFI (결합 시각화)
    with st.container(border=True):
        st.markdown("#### RSI & MFI (강도 및 자금유입)")
        fig2 = create_chart()
        fig2.add_trace(go.Scatter(x=df_recent.index, y=df_recent['RSI'], name='RSI', line=dict(color='purple')))
        fig2.add_trace(go.Scatter(x=df_recent.index, y=df_recent['MFI'], name='MFI', line=dict(color='green')))
        fig2.add_hline(y=50, line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)

    # 3. OBV (수급)
    with st.container(border=True):
        st.markdown("#### OBV (수급 추이)")
        fig3 = create_chart()
        fig3.add_trace(go.Scatter(x=df_recent.index, y=df_recent['OBV'], line=dict(color='orange')))
        st.plotly_chart(fig3, use_container_width=True)

    # 4. ATR & Band Width (변동성 에너지)
    with st.container(border=True):
        st.markdown("#### Volatility (ATR & Band Width)")
        fig4 = create_chart()
        fig4.add_trace(go.Scatter(x=df_recent.index, y=df_recent['ATR'], name='ATR', line=dict(color='red')))
        st.plotly_chart(fig4, use_container_width=True)
