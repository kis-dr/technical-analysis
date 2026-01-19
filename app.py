
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
# 1. 설정 및 데이터 처리 클래스
# ==========================================
class TechnicalAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self._load_data()

    def _load_data(self):
        try:
            # 1. FinanceDataReader (Naver Finance) 시도
            # 한국 주식 코드는 숫자 6자리로 들어옵니다 (예: '005930')
            # yfinance와 달리 .KS를 붙이지 않습니다.
            
            # 10년 전 날짜 계산
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            
            # fdr을 사용하여 데이터 로드 (한국 주식에 훨씬 안정적)
            df = fdr.DataReader(self.ticker, start=start_date)
            
            if df.empty:
                # 혹시 실패하면 yfinance로 2차 시도 (Backup)
                symbol = f"{self.ticker}.KS"
                stock = yf.Ticker(symbol)
                df = stock.history(period="10y")
            
            if df.empty: return df

            # 컬럼 이름 표준화 (fdr은 이미 Open, High, Low, Close, Volume, Change 등을 반환함)
            # 필요한 컬럼만 선택
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 데이터 타입 변환 (가끔 문자열로 들어오는 경우 방지)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 0인 거래량 처리
            df['Volume'] = df['Volume'].replace(0, np.nan).fillna(method='ffill')
            df['Volume'] = df['Volume'].fillna(1000000) # 여전히 NaN이면 임의값

            # 오늘 데이터 제외 로직 (장 중인 경우)
            if not df.empty:
                last_date = df.index[-1].date()
                today_date = datetime.now().date()
                # 장 마감 전이라도 데이터가 들어올 수 있으므로, 
                # 분석의 정확도를 위해 오늘 날짜 데이터는 제외 (선택사항)
                if last_date == today_date:
                    df = df.iloc[:-1]
            
            return df

        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
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
# 2. Gemini AI 생성 함수 (최신 google-genai 사용)
# ==========================================
def get_ai_diagnosis(api_key, stock_name, current_price, indicators, signals):
    try:
        client = genai.Client(api_key=api_key,
        http_options=types.HttpOptions(
        # httpx.Client(...)로 그대로 전달됨
        client_args={
            "verify": certifi.where(),   # <- 핵심!
            "trust_env": True,   # HTTPS_PROXY/SSL_CERT_FILE 같은 환경변수도 신뢰
            # 필요 시 "proxies": {"https": "http://user:pwd@proxy:port"} 도 가능
        })
        )

        # 프롬프트 구성
        prompt = f"""
        당신은 전문 주식 기술적 분석가입니다. 아래 데이터를 바탕으로 '{stock_name}' 종목에 대한 기술적 분석 요약을 작성해주세요.

        [기본 정보]
        - 현재가: {current_price:,.0f}원

        [보조지표 수치 및 신호]
        1. MACD: {indicators['MACD']:.2f} (Signal: {indicators['MACD_Signal']:.2f}) -> {signals['MACD_Sig']}
        2. RSI: {indicators['RSI']:.2f} -> {signals['RSI_Sig']}
        3. CCI: {indicators['CCI']:.2f} -> {signals['CCI_Sig']}
        4. Stochastic K: {indicators['Stoch_K']:.2f} -> {signals['Stoch_Sig']}
        5. ADX: {indicators['ADX']:.2f} -> {signals['ADX_Sig']}
        6. MFI: {indicators['MFI']:.2f} -> {signals['MFI_Sig']}
        7. Bull/Bear Power: Bull({indicators['Bull_Power']:.0f}), Bear({indicators['Bear_Power']:.0f}) -> {signals['Power_Sig']}
        8. 이격도(MA Gap): {indicators['MA20_Gap']:.2%} -> {signals['MA_Gap_Sig']}
        9. ROC: {indicators['ROC']:.2f} -> {signals['ROC_Sig']}
        10. 볼린저밴드 폭 상태: {signals['Band_Width_Sig']}

        [요청사항]
        - 현재 기술적 지표들이 가리키는 전반적인 추세(상승/하락/횡보)를 진단하세요.
        - 매수 또는 매도 관점에서 주의해야 할 특이사항(과열, 침체, 다이버전스 가능성 등)이 있다면 언급하세요.
        - 최대 2문장의 자연스러운 한국어 평문으로 요약해서 간결하게 출력하세요. 
        - "지표가 ~하므로" 식의 나열보다는 통찰력 있는 분석 멘트를 제공하세요.
        - 서두 인사말(안녕하세요 등)은 생략하고 바로 본론만 말하세요.
        """

        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI 분석 생성 실패: {str(e)}"

# ==========================================
# 3. 메인 앱 로직
# ==========================================
st.set_page_config(page_title="AI Market Similarity", layout="wide")

# 사이드바 API Key 입력
# with st.sidebar:
#     st.header("설정")
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# CSS 스타일 적용
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }

    /* 차트 컨테이너 및 3번 섹션 박스 스타일 */
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

st.title("📊 기술적 분석 (Prototype)")

threshold = 10
holding_days = 5

@st.cache_data
def get_stock_list():
    # 로컬 CSV 파일에서 로드
    try:
        df_krx = pd.read_csv('KOSPI_filtered.csv')
        # Name과 Code 컬럼이 있는지 확인
        if 'Name' in df_krx.columns and 'Code' in df_krx.columns:
            # Code가 숫자형일 경우 6자리 문자열로 변환 (005930 등)
            df_krx['Code'] = df_krx['Code'].astype(str).str.zfill(6)

            if 'Marcap' in df_krx.columns:
                df_krx = df_krx.sort_values(by='Marcap', ascending=False)

            return dict(zip(df_krx['Name'], df_krx['Code']))
        else:
            st.error("CSV 파일에 'Name' 또는 'Code' 컬럼이 없습니다.")
            return {}
    except FileNotFoundError:
        st.error("'KOSPI_filtered.csv' 파일을 찾을 수 없습니다. 같은 폴더에 위치시켜주세요.")
        return {}

@st.cache_data(ttl=3600)
def run_analysis(ticker):
    analyzer = TechnicalAnalyzer(ticker)
    if analyzer.df.empty: return None

    df = analyzer.add_indicators()
    if df is None or df.empty: return None

    signal_df = df.apply(analyzer.interpret_indicators, axis=1)
    full_df = pd.concat([df, signal_df], axis=1)
    return full_df

stock_map = get_stock_list()

if stock_map:
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
             <b>📈 과거 패턴 분석 요약</b><br>
            최근 10년 기술적 지표가 유사했던 날은 총 <b>{calc_count}일</b> 포착되었습니다.<br>
            해당 시점들의 {holding_days}일 후 평균적으로 상승했던 비율은 <b style="color: {win_color};">{calc_win_rate:.1f}%</b>, 
            수익률은 <b style="color: {ret_color};">{calc_avg_return:+.1f}%</b>입니다.<br>
            <span style="font-size: 0.85rem; color: #555555;">(※ 이는 과거 통계일 뿐이며, 미래 수익을 보장하지 않습니다.)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # [NEW] Gemini AI 진단 멘트 출력
    # ---------------------------------------------------------


    # ==========================================
    # SECTION 1: 과거 유사 패턴 백테스팅
    # ==========================================

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

    # ==========================================
    # SECTION 2: 과거 유사 패턴 기반 미래 예측 (스파게티 차트 & 분석 테이블)
    # ==========================================
    
    # 화면을 좌(차트) 우(테이블)로 분할 (비율 2.5 : 1)
    col_chart, col_table = st.columns([3, 1])
    
    # -------------------------------------------------------
    # [좌측] 시나리오 예측 차트 (스파게티 차트)
    # -------------------------------------------------------
    with col_chart:
        # 최근 3개월 데이터 준비
        lookback_days = 10  
        df_recent_history = full_df.iloc[-lookback_days:]
        
        # 차트 캔버스 생성
        fig_projection = go.Figure()

        current_close = today_row['Close']
        future_movements = [] 

        # 미래 날짜 축 생성
        last_date_obj = pd.to_datetime(today_row.name)
        future_dates = [last_date_obj + timedelta(days=i) for i in range(0, holding_days + 1)]

        if not similar_days.empty:
            for idx in similar_days.index:
                loc_idx = full_df.index.get_loc(idx)
                
                # 데이터 슬라이싱 및 정규화
                if loc_idx + holding_days < len(full_df):
                    past_segment = full_df.iloc[loc_idx : loc_idx + holding_days + 1]['Close']
                    base_price_past = full_df.iloc[loc_idx]['Close']
                    rebased_segment = (past_segment.values / base_price_past) * current_close
                    
                    future_movements.append(rebased_segment)
                    
                    # 개별 경로 (연한 회색)
                    fig_projection.add_trace(go.Scatter(
                        x=future_dates, 
                        y=rebased_segment, 
                        mode='lines', 
                        line=dict(color='rgba(200, 200, 200, 0.4)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

        # 메인 1: 최근 3개월 주가 (검정 실선)
        fig_projection.add_trace(go.Scatter(
            x=df_recent_history.index, 
            y=df_recent_history['Close'], 
            mode='lines', 
            name='최근 주가', 
            line=dict(color='black', width=2)
        ))

        # 메인 2: 예상 평균 경로 (점선)
        if future_movements:
            # 시각적으로는 반올림된 가격을 보여주더라도
            avg_path = np.round(np.mean(future_movements, axis=0), 0)
            avg_color = '#d62728' if calc_avg_return > 0 else '#1f77b4' # 색상도 수익률 기준
            
            fig_projection.add_trace(go.Scatter(
                x=future_dates,
                y=avg_path,
                mode='lines+markers',
                name=f'예상 평균',
                line=dict(color=avg_color, width=3, dash='dot'),
                marker=dict(size=5)
            ))
            
            # [수정 핵심] 여기서 직접 계산하지 않고, Section 1에서 구한 'calc_avg_return' 변수를 사용
            fig_projection.add_annotation(
                x=future_dates[-1], y=avg_path[-1],
                text=f"{calc_avg_return:+.2f}%", # <-- Section 1 값과 일치
                showarrow=True, arrowhead=1, ax=35, ay=-30,
                font=dict(color=avg_color, size=13, weight='bold')
            )

        # 기준선 (0%)
        combined_x_range = list(df_recent_history.index) + future_dates[1:]
        fig_projection.add_shape(
            type="line",
            x0=combined_x_range[0], y0=current_close,
            x1=combined_x_range[-1], y1=current_close,
            line=dict(color="gray", width=1, dash="dash"),
        )

        fig_projection.update_layout(
            title=dict(
                text=f"<b>과거 유사 패턴 매칭 </b>", 
                font=dict(size=18),
                x=0, y=0.95
            ),
            template="plotly_white", 
            height=400, # 높이 조정
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1
            ),
            xaxis=dict(title=None, tickformat="%m-%d", showgrid=False),
            yaxis=dict(tickformat=",", showgrid=True, gridcolor='#f0f0f0'),
            hovermode="x unified"
        )
        st.plotly_chart(fig_projection, use_container_width=True)

    # -------------------------------------------------------
    # [우측] 상세 데이터 테이블 (수정됨: 모든 건수 표시)
    # -------------------------------------------------------
    with col_table:
        st.markdown(f"<div style='margin-top: 10px; font-weight:bold; font-size:1.05rem;'>유사 시점 목록</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem; color:gray; margin-bottom:10px;'>총 {len(similar_days)}건 (최근순)</div>", unsafe_allow_html=True)

        if not similar_days.empty:
            records = []
            for idx in similar_days.index:
                loc_idx = full_df.index.get_loc(idx)
                
                # 표시할 날짜 문자열
                date_str = idx.strftime("%Y-%m-%d")
                
                # 미래 데이터 확인
                if loc_idx + holding_days < len(full_df):
                    past_price = full_df.iloc[loc_idx]['Close']
                    future_price = full_df.iloc[loc_idx + holding_days]['Close']
                    ret = (future_price - past_price) / past_price
                    
                    # 결과가 있는 경우
                    records.append({
                        "발생일": date_str,
                        "수익률": ret,   # 숫자형 (정렬/색상용)
                        "비고": f"{ret:+.2%}" # 표시용 문자열
                    })
                else:
                    # 결과가 아직 없는 경우 (최근 발생)
                    records.append({
                        "발생일": date_str,
                        "수익률": 0,     # 색상 처리를 위해 0 또는 NaN 처리
                        "비고": "진행중"  # 표시용 텍스트
                    })
            
            # DataFrame 생성 및 정렬
            df_table = pd.DataFrame(records)
            df_table = df_table.sort_values(by="발생일", ascending=False)
            
            # 색상 스타일링 함수
            def style_table(row):
                val = row['비고']
                if val == "진행중":
                    color = "gray"
                elif "+" in val: # 양수
                    color = "#d62728" # 빨강
                elif "-" in val: # 음수
                    color = "#1f77b4" # 파랑
                else:
                    color = "black"
                return [f'color: {color}; font-weight: bold' if col == '비고' else '' for col in row.index]

            # '수익률' 컬럼은 로직용이므로 숨기고 '비고'를 보여줌
            st.dataframe(
                df_table.style.apply(style_table, axis=1),
                use_container_width=True,
                height=350,
                hide_index=True,
                column_order=["발생일", "비고"], # 수익률(숫자) 컬럼 숨김
                column_config={
                    "발생일": st.column_config.TextColumn("발생일", width="medium"),
                    "비고": st.column_config.TextColumn(f"{holding_days}일 후", width="small")
                }
            )
        else:
            st.caption("표시할 데이터가 없습니다.")
    # ==========================================
    # SECTION 3: 오늘의 10대 지표 정밀 진단
    # ==========================================
    st.markdown("---")
    st.markdown(f"### 기술적 분석 지표 진단 (기준일: {last_date} | 주가: {today_row['Close']:,.0f}원)")

    if gemini_api_key:
        with st.spinner("🤖 AI가 보조지표를 정밀 분석 중입니다..."):
            ai_comment = get_ai_diagnosis(
                gemini_api_key, 
                selected_name, 
                today_row['Close'], 
                today_row, 
                today_signals
            )

            st.markdown(f"""
            <div style="background-color: #f1f8e9; padding: 15px; border-radius: 10px; border: 1px solid #c5e1a5; margin-bottom: 20px;">
                <h4 style="margin-top:0; color: #33691e;">✨ AI 기술적 진단</h4>
                <p style="margin: 0; font-size: 1.2rem; color: #333333; line-height: 1.6;">{ai_comment}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        pass

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
            with st.container(border=True):
                st.markdown(f"**{info['name']}**", help=info['tip'])
                st.markdown(f"<div style='color:{color}; font-weight:bold; font-size:15px; margin-top:5px;'>{status}</div>", unsafe_allow_html=True)

    # ==========================================
    # SECTION 4: 최근 6개월 기술적 지표 정밀 분석
    # ==========================================

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



