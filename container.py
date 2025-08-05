import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform

# ✅ 한글 폰트 설정 (OS별로 처리)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')   # MacOS
else:
    plt.rc('font', family='NanumGothic')   # 리눅스 (추가 설치 필요 가능)

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ✅ 인증 ID 목록
ALLOWED_IDS = ['hansehyuk']

# 🔐 사용자 인증
if 'authorized' not in st.session_state:
    st.session_state.authorized = False

if not st.session_state.authorized:
    # 라이트 모드 스타일 적용
    # apply_light_theme()  # 기존에 있으면 활성화
    st.header("DATA & AI 활용 국내 수출 컨테이너 고객 분석")
    
    user_id = st.text_input("🔐 허가된 사람만 입장 가능합니다. 아이디를 입력하세요.")
   
    if st.button("입장"):
        if user_id in ALLOWED_IDS:
            st.session_state.authorized = True 
            st.rerun()         
        else:
            st.warning("등록된 아이디가 아닙니다. 관리자에게 문의하세요")
    
    st.image("pepe7.png", width=1600)
    st.stop()

# 📁 파일 경로 설정
PREDEFINED_FILE_PATH = 'combined4.xlsx'

# 📄 데이터 로드
@st.cache_data
def load_data():
    try:
        df = pd.read_excel(PREDEFINED_FILE_PATH, parse_dates=['선적일'], engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {e}")
        return None
    


# 📊 데이터 개요 대시보드 함수
def show_data_overview(df, start_date=None, end_date=None):
    
    
    # 날짜가 없으면 데이터 기준 min/max로 설정
    if start_date is None:
        start_date = df['선적일'].min()
    if end_date is None:
        end_date = df['선적일'].max()
    
    # 날짜 문자열 포맷팅 (datetime -> yyyy-mm-dd)
    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
    end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)

    st.markdown(f"✅ **분석 데이터 개요 ({start_str} ~ {end_str})**")

    total_records = len(df)
    total_exporters = df['수출자'].nunique()
    total_loading_ports = df['선적항'].nunique()
    total_countries = df['도착지국가'].nunique()
    total_arrival_ports = df['도착항'].nunique()
    total_containers = df['컨테이너수'].sum()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.markdown("""
    <div style='text-align: center;'>
        📄 <b>선적 건</b><br>
        <span style='font-size: 20px;'>{:,}</span>
    </div>
    """.format(total_records), unsafe_allow_html=True)

    col2.markdown("""
    <div style='text-align: center;'>
        👤 <b>수출자</b><br>
        <span style='font-size: 20px;'>{:,}</span>
    </div>
    """.format(total_exporters), unsafe_allow_html=True)

    col3.markdown("""
    <div style='text-align: center;'>
        📦 <b>컨테이너</b><br>
        <span style='font-size: 20px;'>{:,}</span>
    </div>
    """.format(total_containers), unsafe_allow_html=True)

    col4.markdown("""
    <div style='text-align: center;'>
        ⚓ <b>선적항</b><br>
        <span style='font-size: 20px;'>{}</span>
    </div>
    """.format(total_loading_ports), unsafe_allow_html=True)

    col5.markdown("""
    <div style='text-align: center;'>
        🌍 <b>도착지국가</b><br>
        <span style='font-size: 20px;'>{}</span>
    </div>
    """.format(total_countries), unsafe_allow_html=True)

    col6.markdown("""
    <div style='text-align: center;'>
        ⚓ <b>도착항</b><br>
        <span style='font-size: 20px;'>{}</span>
    </div>
    """.format(total_arrival_ports), unsafe_allow_html=True)

    st.write("")
    st.image("pepe5.png", width=700)

# 🔍 조건 기반 필터링 함수
def filter_data(df, start_date, end_date, loading_port, arrival_port, arrival_country, min_containers):
    df = df[(df['선적일'] >= pd.to_datetime(start_date)) & (df['선적일'] <= pd.to_datetime(end_date))]

    if loading_port != 'All':
        df = df[df['선적항'] == loading_port]
    if arrival_country != 'All':
        df = df[df['도착지국가'] == arrival_country]
    if arrival_port != 'All':
        df = df[df['도착항'] == arrival_port]

    grouped = df.groupby('수출자').agg({'컨테이너수': 'sum'}).reset_index()
    grouped = grouped[grouped['컨테이너수'] >= min_containers]
    filtered_df = df[df['수출자'].isin(grouped['수출자'])]

    return filtered_df

def generate_exporter_report(수출자, df):
    exporter_data = df[df['수출자'] == 수출자]

    if exporter_data.empty:
        return "해당 수출자에 대한 데이터가 없습니다."

    total_containers = exporter_data['컨테이너수'].sum()
    main_routes = exporter_data.groupby('도착항')['컨테이너수'].sum().sort_values(ascending=False).head(5)
    main_country = exporter_data.groupby('도착지국가')['컨테이너수'].sum().sort_values(ascending=False).head(5)

    prompt = f"""
    다음 데이터를 기반으로 '{수출자}'에 대한 컨테이너 수출 분석 보고서를 작성해 주세요:
    국제물류 포워더로서 해당 수출자에게 컨테이너 물류 영업을 해야 합니다.
    보고서 내용은 기업 개요, 컨테이너 수출 현황, 물류 영업 전략, 컨테이너 선사 협력 전략 네 부분으로 나눠서 작성해야 합니다.
    기업 개요는 주요 사업이나 제품에 대해서 간단하게 설명해주세요.

    - 컨테이너 선적 기간: {exporter_data['선적일'].min().date()} ~ {exporter_data['선적일'].max().date()} 
    - 총 수출한 컨테이너 수: {total_containers}
    - 선적항별 컨테이너 수: {exporter_data.groupby('선적항')['컨테이너수'].sum().to_dict()} 
    - 컨테이너 수출 국가: {exporter_data['도착지국가'].unique().tolist()}
    - 컨테이너 수출 상위 5개 도착지국가: {main_country}
    - 컨테이너 수출 상위 5개 도착항: {main_routes}
    - 컨테이너 부킹 상위 5개 컨테이너 선사: {exporter_data.groupby('컨테이너선사')['컨테이너수'].sum().sort_values(ascending=False).head(5).to_dict()}

    컨테이너 대수는 TEU나 개수로 표현하지 말고, '대수'로 표현해 주세요.
    선적 기간을 반드시 명시하세요.
    컨테이너 수출 국가는 아시아, 유럽, 아프리카, 북미, 남미, 오세아니아 등으로 구분해 주세요.
    물류 영업 전략은 수출자의 주요한 도착지국가와 도착항을 바탕으로 타겟국가, 타겟항구 대상 영업을 확대 제안해주세요 
    컨테이너 선사 협력 전략은 어떤 컨테이너 선사와 협력하는 것이 좋을지 컨테이너 부킹 선사를 바탕으로 제안해주세요.
    """

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an assistant that generates export container analysis reports."},
        {"role": "user", "content": prompt}
    ]
    )

    content = response.choices[0].message.content
    return content
    
# 🔍 실화주 분류 함수
def classify_actual_shippers(exporter_list):
    prompt = f"""
    다음은 대한민국 수출자 리스트입니다. 각 수출자가 물류 회사인지, 아니면 실제 화주인지 분류해 주세요.
    실제 화주는 제품을 직접 생산하거나 수출하는 기업입니다.
    물류회사는 Freight Forwarder, Shipping Company, Logistics, Sea & Air 등입니다.

    수출자 리스트:
    {exporter_list}

    물류회사가 아닌 실제 화주만 JSON 리스트로 알려 주세요.
    형식: ["화주A", "화주B", ...]
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for analyzing export companies."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = response.choices[0].message.content
    return content
    
    # JSON 추출 처리 (간단한 문자열 파싱 기반)
    try:
        actual_exporters = eval(result_text.strip())  # 예: ["삼성전자", "LG화학"]
    except Exception:
        actual_exporters = []
    return actual_exporters



# 🏠 홈으로 돌아가기 버튼 클릭 시 세션 상태 초기화
def reset_to_home():

    if 'authorized' not in st.session_state:
        st.session_state.authorized = False  # 비정상 접근 방지

    # 검색/분석 결과 초기화
    st.session_state.has_search_results = False
    st.session_state.has_analysis_results = False
    st.session_state.analysis_data = None
    st.session_state.show_similar_customers = False
    st.session_state.exporters = []
    st.session_state.start_date = None
    st.session_state.end_date = None
    st.session_state.loading_port = 'All'
    st.session_state.arrival_country = 'All'
    st.session_state.arrival_port = 'All'
    st.session_state.min_containers = 0
    st.session_state.home_clicked = True
    
    # 사이드바 조건들 초기화 (데이터 로드 후 기본값으로 설정됨)
    if 'start_date' in st.session_state:
        del st.session_state.start_date
    if 'end_date' in st.session_state:
        del st.session_state.end_date
    if 'loading_port' in st.session_state:
        del st.session_state.loading_port
    if 'arrival_country' in st.session_state:
        del st.session_state.arrival_country
    if 'arrival_port' in st.session_state:
        del st.session_state.arrival_port
    if 'min_containers' in st.session_state:
        del st.session_state.min_containers
    if 'exporters' in st.session_state:
        del st.session_state.exporters

# 🧭 Streamlit 앱 UI

def app():
    # ✅ 1. 홈 버튼 클릭 감지 및 처리
    if 'home_clicked' not in st.session_state:
        st.session_state.home_clicked = False

    if "home" in st.query_params:        
        reset_to_home()

    # 🔐 인증 확인
    if 'authorized' not in st.session_state:
        st.session_state.authorized = False

    # 홈 버튼을 누른 경우 인증 상태는 유지하고 초기화만 수행
    if st.session_state.home_clicked:
        if st.session_state.get("authorized", False):
            st.session_state.home_clicked = False
            st.rerun()
        else:
            # 비인증 상태일 경우, 로그인으로 이동
            st.session_state.home_clicked = False
            st.session_state.authorized = False
            st.rerun()

    if not st.session_state.authorized:
        st.title("\ud83d\udce6 \uad6d\ub0b4 \ucee8\ud14c\uc774\ub108 \uc218\ucd9c \uace0\uac1d \uac80\uc0c9\uae30")
        user_id = st.text_input("\ud83d\udd10 \ud5c8\uac00\ub41c \uc0ac\ub78c\ub9cc \uc785\uc7a5 \uac00\ub2a5\ud569\ub2c8\ub2e4. \uc544\uc774\ub514\ub97c \uc785\ub825\ud558\uc138\uc694.")

        if st.button("\uc785\uc7a5"):
            if user_id in ALLOWED_IDS:
                st.session_state.authorized = True
                st.rerun()
            else:
                st.warning("\ub4f1\ub85d\ub41c \uc544\uc774\ub514\uac00 \uc544\ub2c8\ub2e4\ub124\uc694. \uad00\ub9ac\uc790\uc5d0\uac8c \ubb38\uc758\ud574\uc8fc\uc138\uc694")

        st.image("pepe7.png", width=1600)
        st.stop()

 


    # 검색 결과나 분석 결과가 있을 때는 타이틀 숨김
    if not st.session_state.get('has_search_results', False) and not st.session_state.get('has_analysis_results', False):
        st.header("Data & AI 활용 국내 수출 컨테이너 고객 분석")
        st.markdown(
          "<hr style='margin-top: 10px; margin-bottom: 10px;'>",
               unsafe_allow_html=True
                    )
        
    with st.spinner("⏳ 조금만 기다려주세요. 데이터 로딩 중입니다. (1분 정도 소요됩니다)"):
        df = load_data()
    if df is None:
        return

    # 세션 상태 초기화
    if 'has_search_results' not in st.session_state:
        st.session_state.has_search_results = False
    if 'has_analysis_results' not in st.session_state:
        st.session_state.has_analysis_results = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None

    

    min_date = df['선적일'].min()
    max_date = df['선적일'].max()

    default_keys = {
        'start_date': min_date,
        'end_date': max_date,
        'loading_port': 'All',
        'arrival_country': 'All',
        'arrival_port': 'All',
        'min_containers': 0,
        'exporters': []
    }
    for key, val in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = val

    with st.sidebar:
            # ✅ 2. 사이드바: 제목 + 홈 버튼을 한 줄에 배치
    
        col1, col2 = st.columns([2.8, 1])  # 비율 조정 가능
        with col1:
            st.subheader("🚩 고객 조건 검색")            
        with col2:
            if st.button("🏠", key="home_button"):
                reset_to_home()
                st.rerun()
        st.markdown(
                  "<p style='font-size:14px; color: black; margin-top: 0px; margin-bottom: 6px;'>📅 기간</p>",
                    unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_date = st.date_input("시작", min_value=min_date, max_value=max_date, value=st.session_state.start_date, label_visibility="collapsed")
        with col2:
            st.session_state.end_date = st.date_input("종료", min_value=min_date, max_value=max_date, value=st.session_state.end_date, label_visibility="collapsed")
        


    loading_port_options = ['All'] + sorted(df['선적항'].dropna().astype(str).unique().tolist())
    loading_port_index = loading_port_options.index(st.session_state.loading_port) if st.session_state.loading_port in loading_port_options else 0
    st.session_state.loading_port = st.sidebar.selectbox("⚓ 선적항", loading_port_options, index=loading_port_index)

    arrival_country_options = ['All'] + sorted(df['도착지국가'].dropna().astype(str).unique().tolist())
    arrival_country_index = arrival_country_options.index(st.session_state.arrival_country) if st.session_state.arrival_country in arrival_country_options else 0
    st.session_state.arrival_country = st.sidebar.selectbox("🌎 도착지국가", arrival_country_options, index=arrival_country_index)

    arrival_port_raw = df[df['도착지국가'] == st.session_state.arrival_country]['도착항'].dropna() if st.session_state.arrival_country != 'All' else df['도착항'].dropna()
    arrival_port_options = ['All'] + sorted(arrival_port_raw.astype(str).unique().tolist())
    arrival_port_index = arrival_port_options.index(st.session_state.arrival_port) if st.session_state.arrival_port in arrival_port_options else 0
    st.session_state.arrival_port = st.sidebar.selectbox("⚓ 도착항", arrival_port_options, index=arrival_port_index)

    container_values = [0, 10, 50, 100, 500, 1000, 5000, 10000]
    container_index = container_values.index(st.session_state.min_containers) if st.session_state.min_containers in container_values else 0
    st.session_state.min_containers = st.sidebar.selectbox("📦 최소 컨테이너 수", container_values, index=container_index)

    if st.sidebar.button("고객 검색"):
        st.session_state.has_search_results = True
        st.session_state.has_analysis_results = False  # 검색할 때는 분석 결과 숨김3
        st.session_state.analysis_data = None
        st.session_state.show_similar_customers = False
        st.rerun()  # 즉시 페이지 새로고침하여 헤더 숨김
        
    # 고객 검색 결과 표시
    if st.session_state.has_search_results:

        # ✅ 검색 조건 요약 표시
        start_str = st.session_state.start_date.strftime('%Y-%m-%d')
        end_str = st.session_state.end_date.strftime('%Y-%m-%d')
        st.subheader(f"📊 조건 검색 결과 ({start_str} ~ {end_str})")
        st.markdown(
          "<hr style='margin-top: 10px; margin-bottom: 10px;'>",
               unsafe_allow_html=True
                    )
        st.markdown("🚩 **조건 정보**")
        st.markdown("""
        <div style='display: flex; justify-content: space-around; text-align: center;'>
            <div>
                <strong>⚓ 선적항</strong><br>
                <span style='font-size:16px;'>{loading_port}</span>
            </div>
            <div>
                <strong>🌎 도착지국가</strong><br>
                <span style='font-size:16px;'>{arrival_country}</span>
            </div>
            <div>
                <strong>⚓ 도착항</strong><br>
                <span style='font-size:16px;'>{arrival_port}</span>
            </div>
            <div>
                <strong>📦 최소 컨테이너 수</strong><br>
                <span style='font-size:16px;'>{min_containers:,}</span>
            </div>
        </div>
        """.format(
            loading_port=st.session_state.loading_port,
            arrival_country=st.session_state.arrival_country,
            arrival_port=st.session_state.arrival_port,
            min_containers=st.session_state.min_containers
        ), unsafe_allow_html=True)

        st.markdown(
          "<hr style='margin-top: 10px; margin-bottom: 10px;'>",
               unsafe_allow_html=True
                    )
        
        with st.spinner("⌛ 조건 기반 데이터를 조회 중입니다..."):
            result_df = filter_data(
                df,
                st.session_state.start_date,
                st.session_state.end_date,
                st.session_state.loading_port,
                st.session_state.arrival_port,
                st.session_state.arrival_country,
                st.session_state.min_containers
            )
            if not result_df.empty:
                grouped = result_df.groupby('수출자').agg({'컨테이너수': 'sum'}).reset_index()
                grouped = grouped.sort_values(by='컨테이너수', ascending=False)
                grouped['순위'] = grouped['컨테이너수'].rank(ascending=False, method='min')
                grouped = grouped[['순위', '수출자', '컨테이너수']].reset_index(drop=True)
                total_customers = len(grouped)  # 총 고객 수 계산
                
                if 'show_actual_shippers' not in st.session_state:
                   st.session_state.show_actual_shippers = False
                
                st.write("✅ **고객 리스트**")
                with st.expander(f"🔍 총 **{total_customers}**개 고객 확인", expanded=False):
                     st.write("", grouped)
                        # ▶ 버튼 클릭 시 GPT 요청하도록 구성
                     if st.button("✨ AI 실화주 확인", key="check_actual_shippers"):
                        if st.session_state.show_actual_shippers:
                            with st.spinner("AI를 통해 실화주 분류 중입니다. "):
                                exporters_list = grouped['수출자'].tolist()
                                actual_shippers = classify_actual_shippers(exporters_list)

                            if actual_shippers:
                                    actual_df = grouped[grouped['수출자'].isin(actual_shippers)].copy()
                                    st.success("AI를 통해 분류된 실화주 고객입니다.")
                                    st.dataframe(actual_df)

                            else:
                                    st.warning("다시 한 번 시도해주세요.")
                            st.session_state.show_actual_shippers = True

                port_grouped = result_df.groupby('컨테이너선사').agg({'컨테이너수': 'sum'}).reset_index()                
                port_grouped = port_grouped.sort_values(by='컨테이너수', ascending=False)
                port_grouped['순위'] = port_grouped['컨테이너수'].rank(ascending=False, method='min')
                port_grouped = port_grouped[['순위', '컨테이너선사', '컨테이너수']].reset_index(drop=True)
                total_lines = len(port_grouped)
                st.write("✅ **컨테이너선사 정보**")
                with st.expander(f"🔍 총 **{total_lines}**개 선사 확인", expanded=False):
                    st.write("", port_grouped)

            else:
                st.warning("조건에 맞는 데이터가 없습니다.")

    

    # 수출자 목록 불러오기 + placeholder 추가
    all_exporters = sorted(df['수출자'].dropna().astype(str).unique().tolist())
    exporter_options = ["Company Name"] + all_exporters

    # 이전 선택 상태 불러오기 (있다면 유지)
    default_exporter = st.session_state.exporters[0] if st.session_state.get("exporters") else exporter_options[0]

    # selectbox 표시
    selected_exporter = st.sidebar.selectbox("📌 **고객 상세 검색**", exporter_options, index=exporter_options.index(default_exporter))

    # 선택된 값이 유효할 때만 session_state에 저장
    if selected_exporter != "Company Name":
        st.session_state.exporters = [selected_exporter]
    else:
        st.session_state.exporters = []
    
    if st.sidebar.button("고객 분석"):
     if st.session_state.exporters:
        st.session_state.has_analysis_results = True
        st.session_state.has_search_results = False

        # 👉 분석 데이터 준비
        date_filtered_df = df[(df['선적일'] >= pd.to_datetime(st.session_state.start_date)) &
                              (df['선적일'] <= pd.to_datetime(st.session_state.end_date))]
        filtered = date_filtered_df[date_filtered_df['수출자'].isin(st.session_state.exporters)]

        if not filtered.empty:
            st.session_state.analysis_data = {
                'filtered': filtered,
                'exporters': st.session_state.exporters.copy()
            }
        else:
            st.session_state.analysis_data = None
            st.warning("선택한 수출자에 해당하는 데이터가 없습니다.")
            return

        st.rerun()
     else:
        st.warning("수출자를 한 명 이상 선택해 주세요.")
    
    # 분석 결과 표시 (세션 상태 기반)
    if st.session_state.has_analysis_results and st.session_state.analysis_data:
        filtered = st.session_state.analysis_data['filtered']
        selected_exporters = st.session_state.analysis_data['exporters']

        start_str = st.session_state.start_date.strftime("%Y-%m-%d")
        end_str = st.session_state.end_date.strftime("%Y-%m-%d")
        selected_exporter_str = ", ".join(selected_exporters)

        st.subheader(f"📈 {selected_exporter_str} 결과 ({start_str} ~ {end_str}) ")
        st.markdown(
          "<hr style='margin-top: 5px; margin-bottom: 10px;'>",
               unsafe_allow_html=True
                    )
        st.markdown("✅ **요약 정보**")

        total_records = len(filtered)
        total_loading_ports = filtered['선적항'].nunique()
        total_countries = filtered['도착지국가'].nunique()
        total_arrival_ports = filtered['도착항'].nunique()
        total_containers = filtered['컨테이너수'].sum()
        total_container_lines = filtered['컨테이너선사'].nunique()

        col1, col2, col3, col4, col5, col6 = st.columns(6)        
        col1.markdown("""
        <div style='text-align: center;'>
            📄 <b>선적 건</b><br>
            <span style='font-size: 20px;'>{:,}</span>
        </div>
        """.format(total_records), unsafe_allow_html=True)

        col2.markdown("""
        <div style='text-align: center;'>
            📦 <b>컨테이너</b><br>
            <span style='font-size: 20px;'>{:,}</span>
        </div>
        """.format(total_containers), unsafe_allow_html=True)

        col3.markdown("""
        <div style='text-align: center;'>
            🚢 <b>부킹 선사</b><br>
            <span style='font-size: 20px;'>{}</span>
        </div>
        """.format(total_container_lines), unsafe_allow_html=True)

        col4.markdown("""
        <div style='text-align: center;'>
            ⚓ <b>선적항</b><br>
            <span style='font-size: 20px;'>{}</span>
        </div>
        """.format(total_loading_ports), unsafe_allow_html=True)

        col5.markdown("""
        <div style='text-align: center;'>
            🌍 <b>도착지국가</b><br>
            <span style='font-size: 20px;'>{}</span>
        </div>
        """.format(total_countries), unsafe_allow_html=True)

        col6.markdown("""
        <div style='text-align: center;'>
            ⚓ <b>도착항</b><br>
            <span style='font-size: 20px;'>{}</span>
        </div>
        """.format(total_arrival_ports), unsafe_allow_html=True)
        st.markdown("")
        
        # [1] 도착지국가별 컨테이너 수 합계
        arrival_country_sum = filtered.groupby('도착지국가').agg({'컨테이너수': 'sum'}).reset_index()
        arrival_country_sum = arrival_country_sum.sort_values(by='컨테이너수', ascending=False).reset_index(drop=True)

        # ▶ 비중(%) 계산 추가
        total_containers = arrival_country_sum['컨테이너수'].sum()
        arrival_country_sum['비중(%)'] = (arrival_country_sum['컨테이너수'] / total_containers * 100).round(1)

        st.markdown("✅ **상세 정보**")

        with st.expander("🔍 **상세 정보 확인**", expanded=False):
            st.markdown("🌍 **도착지국가**")
            st.dataframe(arrival_country_sum)


            grouped_exporter = filtered.groupby(['수출자', '선적항', '도착지국가', '도착항']).agg({'컨테이너수': 'sum'}).reset_index()
            grouped_exporter = grouped_exporter.sort_values(by='컨테이너수', ascending=False).reset_index(drop=True)

            total_sum = grouped_exporter['컨테이너수'].sum()
            total_row = pd.DataFrame([{
                '수출자': '총합계',
                '선적항': '',
                '도착지국가': '',
                '도착항': '',
                '컨테이너수': total_sum
            }])
            grouped_exporter = pd.concat([grouped_exporter, total_row], ignore_index=True)
            
            st.markdown("⚓ **선적항-도착지국가-도착항**")
            st.dataframe(grouped_exporter)

            # [2] 도착지국가별 컨테이너선사별 컨테이너 수 및 비중
            grouped_by_country_line = filtered.groupby(['도착지국가', '컨테이너선사']).agg({'컨테이너수': 'sum'}).reset_index()
            total_per_country = grouped_by_country_line.groupby('도착지국가')['컨테이너수'].transform('sum')
            grouped_by_country_line['비중(%)'] = (grouped_by_country_line['컨테이너수'] / total_per_country * 100).round(1)
            grouped_by_country_line = grouped_by_country_line.sort_values(by=['도착지국가', '컨테이너수'], ascending=[True, False]).reset_index(drop=True)
            
            
            st.markdown("🚢 **도착지국가-컨테이너선사**")
            st.dataframe(grouped_by_country_line)



            arrival_importer_df = (
                filtered.groupby(['도착지국가', '수입자'])
                .agg({'컨테이너수': 'sum'})
                .reset_index()
                .sort_values(['도착지국가', '컨테이너수'], ascending=[True, False])
            )

            st.markdown("🧑 **도착지국가-수입자**")
            st.dataframe(arrival_importer_df)

        st.markdown("✅ **컨테이너 물동량**")
        with st.expander("🔍 **월별 추세 확인**", expanded=False):
            
            # 월별 컨테이너 수 집계
            monthly_container_count = filtered.copy()
            monthly_container_count['월'] = monthly_container_count['선적일'].dt.to_period('M').astype(str)
            monthly_summary = monthly_container_count.groupby('월')['컨테이너수'].sum().reset_index()
            monthly_summary = monthly_summary.sort_values(by='월')

            # ✅ 꺾은선 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(
                monthly_summary['월'],
                monthly_summary['컨테이너수'],
                marker='o',
                linestyle='-',
                color="#1A34AC"
            )

            ax.set_xlabel("", fontsize=12)
            ax.set_ylabel("", fontsize=12)
            ax.set_title("", fontsize=12)
            ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

        with st.expander("🔍 **도착지국가 월별 추세 확인**", expanded=False):
            filtered['선적월'] = filtered['선적일'].dt.to_period('M').astype(str)

            # 월별, 도착지국가별 집계
            monthly_by_country = filtered.groupby(['선적월', '도착지국가'])['컨테이너수'].sum().reset_index()

            # [3] 전체 기간 동안 상위 10개 도착지국가 추출
            top_10_countries = (
                filtered.groupby('도착지국가')['컨테이너수']
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
            )

            # [4] 상위 10개 국가만 필터링
            monthly_top10 = monthly_by_country[monthly_by_country['도착지국가'].isin(top_10_countries)]

            # [5] 피벗 테이블 생성
            pivot_df = monthly_top10.pivot(index='선적월', columns='도착지국가', values='컨테이너수').fillna(0)

            pivot_df = pivot_df[top_10_countries]

            # [6] 그래프 그리기
            fig, ax = plt.subplots(figsize=(10, 4))
            pivot_df.plot(ax=ax, marker='o')

            plt.title("")
            plt.xlabel("")
            plt.ylabel("")
            plt.xticks(rotation=45)
            plt.legend(
                title='Top 10',
                title_fontsize=14,
                fontsize=13.2,
                loc='center left',
                bbox_to_anchor=(1.0, 0.5)  # ▶ 오른쪽 바깥쪽 (x=1.0, y=0.5)
            )
            plt.tight_layout()

            # [7] Streamlit에 표시
            st.pyplot(fig)

        st.markdown(
          "<hr style='margin-top: 10px; margin-bottom: 10px;'>",
               unsafe_allow_html=True
                    )
        st.markdown("✨ **AI 고객 분석 보고서**")

        with st.expander("🔍 **AI 고객 분석 보고서 확인**", expanded=False):     
            with st.spinner("AI가 보고서를 생성하고 있습니다. 잠시만 기다려 주세요."):
                report = generate_exporter_report(selected_exporters[0], df)
                st.success("고객 분석 보고서가 생성되었습니다.")
                st.markdown(report)
        if 'generate_exporter_report' in st.session_state:
            st.markdown(st.session_state.generate_exporter_report)

                   
    if not st.session_state.has_search_results and not st.session_state.has_analysis_results:
        show_data_overview(df)

if __name__ == "__main__":
    app()








