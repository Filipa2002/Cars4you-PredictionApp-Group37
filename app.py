import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Cars 4 You - Group 37",
    page_icon='./images/4.jpg',
    layout="wide",
)
try:
    st.logo("./images/4.jpg")
except:
    pass

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}  
    footer {visibility: hidden;}
    
    /* Import FontAwesome for Icons */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');

    /* Style for the Main Banner */
    .hero-container {
        position: relative;
        width: 100%;
        height: 350px;
        overflow: hidden;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .hero-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: brightness(0.5) contrast(1.1);
    }      
    .hero-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        width: 80%;
    }
    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
        background: -webkit-linear-gradient(#fff, #ccc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.8rem;
        margin-top: 15px;
        font-weight: 400;
        color: #f2e209;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
        letter-spacing: 1px;
    }

    /* Style for the Brand Carousel (Swiper CSS) */
    .brand-slider {
        background: #1a1a1a;
        padding: 25px 0;
        overflow: hidden;
        white-space: nowrap;
        position: relative;
        margin-bottom: 40px;
        border-radius: 12px;
        box-shadow: inset 0 0 10px #000;
    }
    .brand-slider:before, .brand-slider:after {
        content: "";
        position: absolute;
        top: 0;
        width: 100px;
        height: 100%;
        z-index: 2;
    }
    .brand-slider:before { left: 0; background: linear-gradient(to right, #1a1a1a, transparent); }
    .brand-slider:after { right: 0; background: linear-gradient(to left, #1a1a1a, transparent); }
    .brand-track { display: inline-block; animation: slide 30s linear infinite; }          
    .brand-item {
        display: inline-block; 
        padding: 0 50px; 
        font-size: 1.8rem; 
        color: #666;
        font-weight: 800; 
        text-transform: uppercase; 
        vertical-align: middle; 
        transition: 0.4s;
    }
    .brand-item:hover { 
        color: #e5c120; 
        transform: scale(1.15); 
        text-shadow: 0 0 10px rgba(229, 193, 32, 0.4); 
        cursor: pointer; 
    }

    @keyframes slide { 
            0% { transform: translateX(0); } 
            100% { transform: translateX(-50%); } 
    }

    /* About Us & Notice Box */
    .about-us-box { background-color: rgba(0, 0, 0, 0.1); padding: 15px; border-radius: 5px; color: #e0e0e0; margin-bottom: 20px; }
    .notice-box {
        background-color: rgba(234, 142, 17, 0.1); padding: 20px; border-radius: 8px;
        color: #f0f0f0; font-size: 0.95rem; margin-top: 25px; margin-bottom: 25px;
        line-height: 1.5; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }   
       
    /* Style for Recent Purchases Cards */
    .car-card { background-color: #262730; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; transition: transform 0.2s; }
    .car-card:hover { transform: translateY(-5px); }
    .car-card img { width: 100%; height: 180px; object-fit: cover; }
    .card-content { padding: 15px; }
    .card-title { font-size: 1.2rem; font-weight: bold; margin-bottom: 5px; color: #fff; }
    .card-meta { font-size: 0.9rem; color: #aaa; display: flex; justify-content: space-between; margin-bottom: 10px; }
    .card-price { font-size: 1.4rem; font-weight: 900; color: #e32222; text-align: right; }

    /* Footer */
    .corporate-footer {
        display: flex; justify-content: center; align-items: center;
        padding: 20px 30px; margin-top: 50px; border-top: 1px solid #333;
        color: #666; font-size: 0.8rem; font-family: monospace; gap: 15px;
    }
    .footer-logo { height: 45px; opacity: 0.9; transition: opacity 0.3s; }
    .footer-logo:hover { opacity: 1; }

    /* Custom Elements */
    .stTabs [data-baseweb="tab-highlight"] { background-color: #e5c120; }
    div.stSlider > div[data-baseweb="slider"] > div > div { background: #e5c120; }
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] { background-color: #e5c120; }          
    div[data-testid="stThumbValue"], div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
        background-color: transparent !important;
        color: #31333F !important;
        font-family: "Source Sans Pro", sans-serif;
    }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#5C4212', '#a92f02', '#b08972', '#e3a76c', '#e5c120', '#f39c06', '#f2e209']


# load assets including model, artifacts and history
@st.cache_resource
def load_assets():
    model, artifacts, history_df = None, None, None
    load_status = {"model": False, "artifacts": False, "history": False}
    
    # I. Load model
    model_path_gz = 'linear_regression_model.sav.gz'    #APAGAR: MUDAR PARA O MODELO FINAL
    
    if os.path.exists(model_path_gz):
        try:
            model = joblib.load(model_path_gz)
            load_status["model"] = True
        except Exception as e:
            st.error(f"Fatal Model Error: {e}")

    # II. Load artifacts
    if os.path.exists('pipeline_artifacts.joblib'):
        try:
            artifacts = joblib.load('pipeline_artifacts.joblib')
            load_status["artifacts"] = True
        except Exception as e:
            st.error(f"Artifact Load Error: {e}")
            
    # III. Load history
    if os.path.exists('history_data.parquet'):
        try:
            history_df = pd.read_parquet('history_data.parquet', engine='fastparquet') # Using FastParquet
            load_status["history"] = True
        except Exception as e:
            st.error(f"History Load Error: {e}")
    return model, artifacts, history_df, load_status


model, artifacts, history_df, load_status = load_assets()

# Brand to Models dictionary from history data
if load_status["history"] and history_df is not None:
    cols_lower = {c.lower(): c for c in history_df.columns}
    brand_c = cols_lower.get('brand', 'Brand')
    model_c = cols_lower.get('model', 'model')
    
    if brand_c in history_df.columns and model_c in history_df.columns:
        BRAND_MODEL_MAP = history_df.groupby(brand_c)[model_c].unique().apply(list).to_dict()
    else:
        BRAND_MODEL_MAP = {}
        st.error(f"Data Error: Columns 'Brand'/'model' were not found in the history file. Available columns: {history_df.columns.tolist()}")
else:
    BRAND_MODEL_MAP = {}
    st.error("Critical Error: Failed to load the Brand/Model mapping.")


# Function to Create Custom KPI Cards
def create_kpi_card(col, icon_class, title, value, unit="", color_border="#e5c120"):
    html_content = f"""
    <div style="background-color: #262730; border-left: 5px solid {color_border}; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); margin-bottom: 10px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <p style="color: #bbb; font-size: 0.9rem; margin: 0; text-transform: uppercase;">{title}</p>
                <h3 style="color: white; margin: 5px 0 0 0; font-weight: 700;">{value} <span style="font-size: 0.8rem; color: #888;">{unit}</span></h3>
            </div>
            <div style="background-color: rgba(255,255,255,0.1); width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                <i class="{icon_class}" style="color: {color_border}; font-size: 1.2rem;"></i>
            </div>
        </div>
    </div>
    """
    col.markdown(html_content, unsafe_allow_html=True)


# Sidebar: About Us & Feedback
with st.sidebar:
    st.header("Cars 4 You")
    st.markdown("### About Us")
    st.markdown("""<div class="about-us-box"><strong>Cars 4 You</strong> is a leading company in the automotive market worldwide, specializing in the sale of semi-new and used vehicles.<br><br>
                With a strong commitment to offering quality, trust, and excellent service, Cars 4 You positions itself as a leader in its field.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Your Opinion Matters")
    sentiment =st.feedback("stars")
    if sentiment is not None:
        st.success("Thank you for your feedback!")


# Main Banner (Hero Section)
st.markdown("""
<div class="hero-container">
    <img src="https://i.redd.it/bmw-m8-no-rules-edition-but-it-s-orange-v0-o1kbj97il0da1.jpg?width=3840&format=pjpg&auto=webp&s=a00eab14e9584d4e258390f17c720add8b990bd9" class="hero-image">
    <div class="hero-text"><h1 class="hero-title">CARS 4 YOU</h1><p class="hero-subtitle">We Buy Your Car</p></div>
</div>
""", unsafe_allow_html=True)


# Brand Carousel
brands_list = " • ".join(list(BRAND_MODEL_MAP.keys())) if BRAND_MODEL_MAP else "LOADING DATA..."
st.markdown(f"""
<div style="text-align: center; margin-bottom: 10px; font-weight: bold; color: #777; font-size: 0.9rem; letter-spacing: 1px;">BRANDS WE DEAL WITH</div>
<div class="brand-slider"><div class="brand-track"><span class="brand-item">{brands_list} • {brands_list}</span></div></div>
""", unsafe_allow_html=True)


# Recent Acquisitions
st.subheader("🏁 Recent Acquisitions")
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
def render_card(img, title, year, fuel, km, transmission, price):
    return f"""
    <div class="car-card"><img src="{img}" onerror="this.src='https://via.placeholder.com/300x180?text=Car+Image'"><div class="card-content">
            <div class="card-title">{title}</div><div class="card-meta"><span>📅 {year}</span><span>⛽ {fuel}</span></div>
            <div class="card-meta"><span>🛣️ {km} km</span><span>🕹️ {transmission}</span></div><div class="card-price">{price}</div></div></div>
    """
with col_c1: st.markdown(render_card("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT8xmA8Uag8EjUh-jwDwQ3dxXCaK9MD9lr2Nw&s", "Audi Q3", "2013", "Diesel", "52 134", "Semi-Auto", "£12 495"), unsafe_allow_html=True)                             #carID 6265
with col_c2: st.markdown(render_card("https://cdn.easysite.pt/pub/14683/1206245/API_NCAUTO_17089482082340-0.webp", "Toyota Aygo", "2017", "Petrol", "11 304", "Automatic", "£8 399"), unsafe_allow_html=True)                                            #carID 54886
with col_c3: st.markdown(render_card("https://vmscdn.porscheinformatik.com/s2/ro/4B36D27CE96149E7B306731E6EE664C1/images/f5bef485-a6f7-49f8-a2be-9f072cc1af93/440", "Audi Q3", "2015", "Diesel", "69 072", "Manual", "£12 990"), unsafe_allow_html=True) #carID 860
with col_c4: st.markdown(render_card("https://ireland.apollo.olxcdn.com/v1/files/2ua4oaje46gp2-PT/image;s=4096x3072", "Ford Fiesta", "2018", "Petrol", "16 709", "Manual", "£10 495"), unsafe_allow_html=True)                                           #carID 15795
st.markdown("<br>", unsafe_allow_html=True)


# Preprocessing using loaded artifacts; used on user input
#Basically it converts form input into numerical matrix compatible with trained pipeline
def preprocess_input(raw_df, artifacts):
    df = raw_df.copy()

    if 'transmission' in df.columns:
        df['transmission'] = df['transmission'].astype(str).str.lower().str.replace('-', ' ').str.strip()
    if 'fuelType' in df.columns:
        df['fuelType'] = df['fuelType'].astype(str).str.lower().str.strip()

    #new variables: age, is_new_car, is_old_car
    current_ref_year = 2020 
    df["age"] = (current_ref_year - pd.to_numeric(df["year"], errors="coerce")).astype(int)
    df['is_new_car']  = (df['age'] <= 1).astype(int)
    df['is_old_car']  = (df['age'] >= 10).astype(int)
    
    # new variables: miles_per_year, high_mileage, low_mileage, age_mileage_interaction
    if 'q75' not in artifacts or 'q25' not in artifacts:
        raise KeyError("Missing 'q75' or 'q25' in artifacts for mileage categorization.")
    q75 = artifacts['q75']
    q25 = artifacts['q25']
    df['miles_per_year'] = df['mileage'] / (df['age'] + 1)
    df['high_mileage'] = (df['mileage'] > q75).astype(int)
    df['low_mileage']  = (df['mileage'] < q25).astype(int)
    df['age_mileage_interaction'] = df['age'] * df['mileage']

    #new variables: brand_segment, brand_mean_price, brand_median_price, brand_price_std
    brand = df['brand'].iloc[0]
    if 'brand_stats' not in artifacts:
        raise KeyError("Missing 'brand_stats' in artifacts for brand statistics.")
    
    if brand not in artifacts['brand_stats']:
        raise KeyError(f"The brand '{brand}' was not found in the training statistics.")
    
    brand_info = artifacts['brand_stats'][brand]

    try:
        # store stats in df for later
        df['brand_mean_price'] = brand_info['brand_mean_price']
        df['brand_median_price'] = brand_info['brand_median_price']
        df['brand_price_std'] = brand_info['brand_price_std']
        df['brand_segment'] = brand_info['brand_segment']
        df['brand_count'] = brand_info['brand_count']
    except KeyError as e:
        raise KeyError(f"Missing specific stat data for brand '{brand}': {e}")
    
    # new variables: premium_brand_engine_size_interaction, tax_per_engine, mpg_per_liter, brand_model, model_popularity
    df['premium_brand_engine_size_interaction'] = (df['brand_segment'] == 'luxury').astype(int) * df['engineSize']
    df['tax_per_engine'] = df['tax'] / (df['engineSize'] + 0.1)
    df['mpg_per_liter'] = df['mpg'] / (df['engineSize'] + 0.1)
    df['brand_model'] = df['brand'] + '_' + df['model']

    model_name = df['model'].iloc[0]
    if 'model_counts' not in artifacts:
        raise KeyError("Missing 'model_counts' in artifacts.")
    
    if model_name not in artifacts['model_counts']:
        raise KeyError(f"The model '{model_name}' was not found in the training statistics.")
    
    df['model_popularity'] = artifacts['model_counts'][model_name]

    if 'target_enc_mappings' not in artifacts:
        raise KeyError("Missing 'target_enc_mappings' in artifacts.")
    # Target Encoding for 'Brand' and 'brand_model'    
    mappings = artifacts['target_enc_mappings']


    for col in ['Brand', 'brand_model']: 
        col_key = 'Brand' if col == 'Brand' else col

        if col_key not in mappings:
            raise KeyError(f"Mapping for '{col_key}' was not found in artifacts.")

        mapping = mappings[col_key]
        col_in_df = col.replace('Brand', 'brand')
        val = df[col_in_df].iloc[0]

        if val not in mapping:
            raise ValueError(f"Value '{val}' not found in the Target Encoding map for '{col_key}'.")
        
        df[f'{col_key}_target_enc'] = mapping[val]                  


    # One-Hot Encoding for low cardinality categorical variables
    low_card_cols = artifacts.get('low_cardinality_cols')
    if not low_card_cols: raise KeyError("Missing 'low_cardinality_cols' in artifacts.")
    
    ohe = artifacts.get('ohe')
    if not ohe: raise KeyError("Missing 'ohe' in artifacts.")
    
    # Check columns exist
    missing_cols = [c for c in low_card_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for OHE: {missing_cols}")

    try:
        ohe_features = ohe.transform(df[low_card_cols])
    except ValueError as e:
        raise ValueError(f"OHE Transformation Error: {e}. Check if input categories match training categories.")

    ohe_df = pd.DataFrame(ohe_features, columns=ohe.get_feature_names_out(), index=df.index)
        
    df_processed = df.drop(columns=['brand', 'model', 'year', 'transmission', 'fuelType', 'brand_model', 'brand_segment', 'model_popularity'], errors='ignore') #APAGAR: talvez tirar mais
    df_final_step = pd.concat([df_processed, ohe_df], axis=1)
        
    final_cols = artifacts.get('final_columns')
    if not final_cols: raise KeyError("Missing 'final_columns' in artifacts.")

    # Add 'paintQuality%' by renaming from 'paintQuality'
    if 'paintQuality' in df_final_step.columns:
        df_final_step['paintQuality%'] = df_final_step['paintQuality']
    
    # missing flags = 0 since we demand all inputs
    df_final_step['mpg_is_missing'] = 0
    df_final_step['tax_is_missing'] = 0
    df_final_step['engineSize_is_missing'] = 0
    df_final_step['year_is_missing'] = 0

    # Ensure all columns exist
    missing_final_cols = [c for c in final_cols if c not in df_final_step.columns]
    if missing_final_cols:
        raise ValueError(f"Missing final columns for the model: {missing_final_cols}")
    
    df_final_step = df_final_step[final_cols]
        
    # Scaling
    scaler = artifacts.get('scaler')
    if not scaler: raise KeyError("Missing 'scaler' in artifacts.")
    
    X_scaled = scaler.transform(df_final_step)
    return X_scaled



# Main Interface
tab1, tab2 = st.tabs(["📝 Vehicle Evaluation", "📊 Purchase History Analysis"])

# TAB 1: evaluation form
with tab1:
    st.markdown("### Car Evaluation Form")
    
    if not load_status["artifacts"] or not load_status["model"]:
        st.error("ERROR: Prediction Engine Offline.")
    
    col1, col2 = st.columns(2)
    with col1: selected_brand = st.selectbox("Brand", options=list(BRAND_MODEL_MAP.keys()))
    with col2: selected_model = st.selectbox("Model", options=BRAND_MODEL_MAP.get(selected_brand, []))

    col3, col4 = st.columns(2)
    with col3: year = st.slider("Registration Year", 1970, 2020, 2015)
    with col4: mileage = st.number_input("Mileage (miles)", 0.0, 323000.0, 40000.0, 1.0)
    
    col5, col6, col7 = st.columns(3)
    with col5: engine_size = st.number_input("Engine Size (L)", 0.0, 6.6, 1.5, 0.1)
    with col6: fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other", "Unknown"])
    with col7: transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto", "Other", "Unknown"])
    
    col8, col9, col10 = st.columns(3)
    with col8: tax = st.number_input("Vehicle Tax (£)", 0.0, 580.0, 150.0)
    with col9: mpg = st.number_input("Miles per Gallon", 0.0, 470.8, 50.0)
    with col10: paint_quality = st.slider("Paint Quality (%)", 0, 100, 80)

    col11, col12 = st.columns(2)
    with col11: prev_owners = st.number_input("Previous Owners", 0, 6, 1)
    with col12: 
        st.write("")
        st.write("")
        has_damage = st.checkbox("Damaged Vehicle?")                         
    
    st.markdown("""<div class="notice-box"><strong> Important Notice:</strong><br>
                Dear Customer, should your vehicle's specifications fall outside the ranges supported by our automated system,
                we strongly recommend contacting our team directly. This ensures you receive a specialized evaluation tailored
                to your vehicle's unique characteristics, as our digital model is optimized for standard market parameters.</div>""", unsafe_allow_html=True)

    if st.button("Calculate Price Estimate", type="primary", use_container_width=True):
        if not load_status["artifacts"] or not load_status["model"]:
            st.error("Cannot predict: Missing model files.")
        else:
            raw_data = pd.DataFrame({
                'brand': [selected_brand], 'model': [selected_model], 'year': [year], 'mileage': [mileage],
                'tax': [tax], 'fuelType': [fuel_type], 'mpg': [mpg], 'engineSize': [engine_size],
                'paintQuality': [paint_quality], 'previousOwners': [prev_owners], 'hasDamage': [has_damage],
                'transmission': [transmission]
            })
            
            try:
                processed_input = preprocess_input(raw_data, artifacts)
                predicted_price = model.predict(processed_input)[0]

                st.markdown("---")
                st.subheader("Estimated Purchase Price")
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = predicted_price, number = {'prefix': "£ "},
                        domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Estimated Buying Price"},
                        gauge = {'axis': {'range': [450, 159999]}, 'bar': {'color': PALETTE[5]}}
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    if has_damage: st.warning("Damage penalty applied.")       #APAGAR
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# TAB 2: analytics dashboard
with tab2:
    st.header("Market Dashboard")
    
    if load_status["history"] and history_df is not None:
        df_analytics = history_df.copy()
        
        # Capitalization normalization - Case insensitive lookup
        cols_lower = {c.lower(): c for c in df_analytics.columns}
        
        # Helper to get col name safely
        def get_col(name, fallback):
            return cols_lower.get(name.lower(), fallback)

        price_col = get_col('price', 'price')
        mileage_col = get_col('mileage', 'mileage')
        fuel_col = get_col('fueltype', 'fuelType')
        brand_col = get_col('brand', 'Brand')
        year_col = get_col('year', 'year')
        model_col = get_col('model', 'model')
        trans_col = get_col('transmission', 'transmission')
        eng_col = get_col('engineSize', 'engineSize')
        age_col = get_col('age', 'age')
        miles_per_year_col = get_col('miles_per_year', 'miles_per_year')

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1: create_kpi_card(kpi1, "fas fa-tag", "Avg Market Price", f"£ {df_analytics[price_col].mean():,.0f}")
        with kpi2: create_kpi_card(kpi2, "fas fa-car", "Total Vehicles", f"{len(df_analytics)}")
        with kpi3: create_kpi_card(kpi3, "fas fa-road", "Avg Mileage", f"{df_analytics[mileage_col].mean():,.0f}", "mi")
        with kpi4:
            elec_count = len(df_analytics[df_analytics[fuel_col].astype(str).str.lower() == 'electric'])
            create_kpi_card(kpi4, "fas fa-bolt", "Electric Share", f"{(elec_count/len(df_analytics)*100):.1f}", "%")

        st.markdown("---")
        
        with st.expander("🔎 Advanced Filters", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            brands_f = c1.multiselect("Brands", df_analytics[brand_col].unique(), default=df_analytics[brand_col].unique()[:4])
            fuel_f = c2.multiselect("Fuel", df_analytics[fuel_col].unique(), default=df_analytics[fuel_col].unique())
            trans_f = c3.multiselect("Transmission", df_analytics[trans_col].unique(), default=df_analytics[trans_col].unique())
            years_f = c4.multiselect("Year", sorted(df_analytics[year_col].unique()), default=sorted(df_analytics[year_col].unique())[-5:])
        
        # Apply filters
        if brands_f: df_analytics = df_analytics[df_analytics[brand_col].isin(brands_f)]
        if fuel_f: df_analytics = df_analytics[df_analytics[fuel_col].isin(fuel_f)]
        if trans_f: df_analytics = df_analytics[df_analytics[trans_col].isin(trans_f)]
        if years_f: df_analytics = df_analytics[df_analytics[year_col].isin(years_f)]

       # Row 1
        st.subheader("Hierarchy Analysis")
        fig_sun = px.sunburst(df_analytics, path=[brand_col, model_col, fuel_col, trans_col], 
                              values=price_col, color=price_col, color_continuous_scale='Oranges',
                              height=600)
        st.plotly_chart(fig_sun, use_container_width=True)

        st.markdown("---")
        # Row 2
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Brand Radar")
            mpg_c = get_col('mpg', 'mpg')
            cols_to_use = [price_col, mileage_col, year_col]
            if eng_col in df_analytics.columns: cols_to_use.append(eng_col)
            if mpg_c in df_analytics.columns: cols_to_use.append(mpg_c)
            
            df_rad = df_analytics.groupby(brand_col)[cols_to_use].mean().reset_index()
            scaler = MinMaxScaler()
            df_rad[cols_to_use] = scaler.fit_transform(df_rad[cols_to_use])
            
            fig = go.Figure()
            all_brands = df_rad[brand_col].unique()
            
            for i, b in enumerate(all_brands):
                val = df_rad[df_rad[brand_col]==b][cols_to_use].values.flatten().tolist()
                is_visible = True if i < 3 else 'legendonly'
                fig.add_trace(go.Scatterpolar(r=val+[val[0]], theta=cols_to_use+[cols_to_use[0]], fill='toself', name=b, visible=is_visible))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Price vs Mileage Density")
            custom_colors = [
                [0.0, "white"],
                [0.5, "#ffb347"],
                [1.0, "#a92f02"]
            ]
            fig = px.density_heatmap(df_analytics, x=mileage_col, y=price_col, 
                                     marginal_x="histogram", marginal_y="histogram",
                                     color_continuous_scale=custom_colors)
            fig.update_xaxes(range=[0, 80000])
            fig.update_yaxes(range=[5000, 60000])
            st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("---")
        # Row 3
        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Transmission Price Impact")
            fig = px.box(df_analytics, x=trans_col, y=price_col, color=trans_col, color_discrete_sequence=PALETTE, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with d2:
            st.subheader("Engine Size vs Price Correlation")
            fig = px.scatter(df_analytics, x=eng_col, y=price_col, color=brand_col, size=price_col, hover_data=[model_col], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # Row 4
        st.subheader("Feature Distributions Explorer")
        e1, e2 = st.columns(2)
        
        with e1:
            # Numeric Vars Selector
            # Check availability
            avail_nums = [c for c in [price_col, mileage_col, 'age', 'miles_per_year', 'tax', 'mpg', eng_col] if c in df_analytics.columns]
            sel_num = st.selectbox("Select Numerical Variable", avail_nums)
            
            fig_num = px.histogram(df_analytics, x=sel_num, nbins=30, color_discrete_sequence=PALETTE)
            st.plotly_chart(fig_num, use_container_width=True)
            
        with e2:
            # Categorical Vars Selector
            # Check availability
            avail_cats = [c for c in [brand_col, fuel_col, trans_col, 'brand_segment'] if c in df_analytics.columns]
            sel_cat = st.selectbox("Select Categorical Variable", avail_cats)
            
            fig_cat = px.bar(df_analytics[sel_cat].value_counts().reset_index(), x=sel_cat, y='count', 
                             color_discrete_sequence=PALETTE)
            st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")
        # Row 5
        st.subheader("Advanced Insights")
        f1, f2 = st.columns(2)
        
        with f1:
            st.markdown("**Depreciation Analysis: Price vs Age**")
            if 'age' in df_analytics.columns:
                fig = px.scatter(df_analytics, x='age', y=price_col, color=fuel_col, 
                                trendline="lowess", trendline_color_override="black",
                                color_discrete_sequence=PALETTE, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        
        with f2:
            st.markdown("**Value for Money: MPG vs Price (Size = Tax)**")
            if 'mpg' in df_analytics.columns and 'tax' in df_analytics.columns:
                fig = px.scatter(df_analytics, x='mpg', y=price_col, size='tax', color=trans_col,
                                color_discrete_sequence=PALETTE, template="plotly_white", opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)


    else:
        st.error("ERROR: Unable to load purchase history data for analytics.")


# Footer
st.markdown("""
<div class="corporate-footer">
    <span>Group Project 37 | Machine Learning 2025/2026</span>
    <img src="https://cityme.novaims.unl.pt/images/footer/novaims.png" class="footer-logo">
</div>
""", unsafe_allow_html=True)
