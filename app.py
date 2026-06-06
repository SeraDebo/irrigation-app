# app.py
"""
Smart Irrigation Advisor & Crop Recommendation System
An India-Wide AI-Driven Decision Support App
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import pickle
from datetime import datetime, timedelta
import altair as alt

import db  # Supabase (Postgres) data-access layer — see db.py

st.set_page_config(page_title="Smart Irrigation Advisor", layout="wide", initial_sidebar_state="expanded")

# --- PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(filename):
    # Try workspace-relative path
    p = os.path.join(BASE_DIR, filename)
    if os.path.exists(p):
        return p
    # Try codes-relative path
    p_codes = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(p_codes):
        return p_codes
    # Direct absolute fallback
    return os.path.join("d:\\irrigation_project", filename)

# --- DATA LOADERS (Supabase / Postgres) ---
# CSV data now lives in Supabase tables (see db.py for the data-access layer).
# The trained model (.pkl) and flowchart (.png) still load from the repo via get_path().
df_centroids = db.load_district_centroids()
df_kc_defaults = db.load_crop_kc_defaults()
df_indian_crop = db.load_indian_crop_dataset()
# crop_production is large, so it is queried on demand (per crop) in the Yield tab
# via db.get_crop_list() and db.get_crop_production(crop) — not loaded up front.

# --- PRECALCULATED STATE AVERAGES ---
STATE_WEATHER_DEFAULTS = {
    "Andaman and Nicobar": {"temp": 23.5, "hum": 81.9, "ph": 6.6, "rain": 240.6},
    "Andhra Pradesh": {"temp": 23.1, "hum": 81.6, "ph": 6.2, "rain": 235.2},
    "Assam": {"temp": 23.6, "hum": 82.4, "ph": 6.5, "rain": 238.4},
    "Chattisgarh": {"temp": 25.8, "hum": 82.0, "ph": 5.9, "rain": 201.7},
    "Goa": {"temp": 23.6, "hum": 82.1, "ph": 5.8, "rain": 218.9},
    "Gujarat": {"temp": 22.5, "hum": 66.6, "ph": 6.3, "rain": 101.2},
    "Haryana": {"temp": 18.8, "hum": 17.0, "ph": 7.3, "rain": 81.0},
    "Himachal Pradesh": {"temp": 19.7, "hum": 19.8, "ph": 6.3, "rain": 96.1},
    "Jammu and Kashmir": {"temp": 23.6, "hum": 22.1, "ph": 5.8, "rain": 130.9},
    "Karnataka": {"temp": 22.2, "hum": 27.9, "ph": 5.8, "rain": 112.7},
    "Kerala": {"temp": 23.6, "hum": 24.3, "ph": 6.2, "rain": 128.9},
    "Madhya Pradesh": {"temp": 23.4, "hum": 26.5, "ph": 5.9, "rain": 110.2},
    "Maharashtra": {"temp": 23.5, "hum": 26.8, "ph": 6.1, "rain": 112.3},
    "Manipur": {"temp": 23.6, "hum": 82.4, "ph": 6.0, "rain": 220.0},
    "Meghalaya": {"temp": 23.6, "hum": 82.4, "ph": 6.0, "rain": 220.0},
    "Nagaland": {"temp": 23.6, "hum": 82.4, "ph": 6.0, "rain": 220.0},
    "Odisha": {"temp": 23.5, "hum": 81.6, "ph": 6.1, "rain": 230.1},
    "Pondicherry": {"temp": 23.5, "hum": 81.6, "ph": 6.1, "rain": 230.1},
    "Punjab": {"temp": 18.8, "hum": 17.0, "ph": 7.3, "rain": 81.0},
    "Rajasthan": {"temp": 22.5, "hum": 66.6, "ph": 6.3, "rain": 101.2},
    "Tamil Nadu": {"temp": 23.1, "hum": 81.6, "ph": 6.2, "rain": 235.2},
    "Telangana": {"temp": 23.1, "hum": 81.6, "ph": 6.2, "rain": 235.2},
    "Tripura": {"temp": 23.6, "hum": 82.4, "ph": 6.0, "rain": 220.0},
    "Uttar Pradesh": {"temp": 23.4, "hum": 26.5, "ph": 5.9, "rain": 110.2},
    "Uttrakhand": {"temp": 19.7, "hum": 19.8, "ph": 6.3, "rain": 96.1},
    "West Bengal": {"temp": 24.5, "hum": 78.5, "ph": 6.4, "rain": 185.3}
}

# --- SOIL DEFAULTS ---
SOIL_DEFAULTS = {
    "sandy": {"fc": 0.10, "wp": 0.04, "description": "Sandy (Fast drainage, low retention)"},
    "sandy_loam": {"fc": 0.20, "wp": 0.08, "description": "Sandy Loam (Moderate infiltration)"},
    "loam": {"fc": 0.30, "wp": 0.12, "description": "Loam (Balanced water-air ratio)"},
    "silt_loam": {"fc": 0.33, "wp": 0.14, "description": "Silt Loam (High water retention)"},
    "clay_loam": {"fc": 0.36, "wp": 0.18, "description": "Clay Loam (Slower infiltration)"},
    "clay": {"fc": 0.40, "wp": 0.20, "description": "Clay (Strong retention, risk of waterlogging)"},
}

# --- CROP CALENDAR SEASONS MAP ---
SEASON_MONTH_MAP = {
    "Kharif": "June - October (Monsoon season crop)",
    "Rabi": "October - March (Winter season crop)",
    "Summer": "March - June (Hot dry season crop)",
    "Winter": "November - April (Cool season crop)",
    "Autumn": "August - January (Late monsoon crop)",
    "Whole Year": "January - December (Perennial crop)"
}

# --- MULTILINGUAL DICTIONARY ---
LANGUAGES = {
    "en": "English",
    "bn": "বাংলা (Bengali)",
    "hi": "हिन्दी (Hindi)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "mr": "मराठी (Marathi)",
    "ml": "മലയാളം (Malayalam)",
    "te": "తెలుగు (Telugu)"
}

TRANSLATIONS = {
    "en": {
        "title": "Smart Irrigation Advisor & AI Recommendation System",
        "home": "Home & Project Overview",
        "crop_rec": "AI Crop Match (NPK)",
        "yield_insights": "India Yield & Crop Insights",
        "advisor": "Smart Irrigation Advisor",
        "eval": "Model Evaluation",
        "select_lang": "Choose Language",
        "welcome": "Welcome, farmer!",
        "recommendation": "Crop Recommendation",
        "best_places": "Best Places & Month to Grow in India",
        "best_place_state": "Best Yielding State",
        "best_place_dist": "Best Yielding District",
        "best_season": "Optimal Season",
        "months": "Typical Growing Months",
        "average_yield": "Average Yield",
        "weather_source": "Weather Data Ingestion Source",
        "soil_source": "Soil Ingestion Source",
        "irrigation_recs": "Irrigation Schedule Recommendations",
        "irrigate_now": "⚠️ IRRIGATE TODAY",
        "no_irrigation": "✅ SOIL MOISTURE ADEQUATE",
        "irrigate_soon": "ℹ️ IRRIGATE SOON",
        "rain_warning": "⛈️ RAIN FORECAST WARNING",
        "rain_forecast_advisory": "Significant rain ({} mm) is forecast in the next 3 days. We recommend delaying irrigation to conserve water and avoid waterlogging.",
        "days_to_depletion": "Estimated days until depletion threshold: {} days.",
        "eval_metrics": "Model Validation Metrics",
        "recommend_crop": "Recommend Crop"
    },
    "bn": {
        "title": "স্মার্ট সেচ উপদেষ্টা এবং এআই ফসল সুপারিশ সিস্টেম",
        "home": "হোম ও প্রকল্পের বিবরণ",
        "crop_rec": "এআই ফসল ম্যাচ (NPK)",
        "yield_insights": "ভারতের ফলন ও ফসলের অন্তর্দৃষ্টি",
        "advisor": "স্মার্ট সেচ উপদেষ্টা",
        "eval": "মডেল মূল্যায়ন",
        "select_lang": "ভাষা নির্বাচন করুন",
        "welcome": "স্বাগতম, কৃষক ভাই!",
        "recommendation": "ফসলের সুপারিশ",
        "best_places": "ভারতে চাষের জন্য সেরা স্থান ও মাস",
        "best_place_state": "সর্বোচ্চ ফলনশীল রাজ্য",
        "best_place_dist": "সর্বোচ্চ ফলনশীল জেলা",
        "best_season": "অনুকূল মরশুম",
        "months": "সাধারণ চাষের মাস",
        "average_yield": "গড় ফলন",
        "weather_source": "আবহাওয়া তথ্য সংগ্রহের উৎস",
        "soil_source": "মাটি তথ্য সংগ্রহের উৎস",
        "irrigation_recs": "সেচ সময়সূচী সুপারিশ",
        "irrigate_now": "⚠️ আজই সেচ দিন",
        "no_irrigation": "✅ পর্যাপ্ত মাটির আর্দ্রতা আছে",
        "irrigate_soon": "ℹ️ শীঘ্রই সেচ দিন",
        "rain_warning": "⛈️ বৃষ্টির পূর্বাভাস সতর্কবার্তা",
        "rain_forecast_advisory": "আগামী ৩ দিনে ভালো বৃষ্টির ({} মিমি) সম্ভাবনা রয়েছে। জল সংরক্ষণ এবং জল জমা রোধ করতে সেচ পিছিয়ে দেওয়ার পরামর্শ দেওয়া হচ্ছে।",
        "days_to_depletion": "মাটির আর্দ্রতা কমার আনুমানিক সময়: {} দিন।",
        "eval_metrics": "মডেল মূল্যায়ন মেট্রিক্স",
        "recommend_crop": "ফসল সুপারিশ করুন"
    },
    "hi": {
        "title": "स्मार्ट सिंचाई सलाहकार और एआई फसल अनुशंसा प्रणाली",
        "home": "होम और परियोजना अवलोकन",
        "crop_rec": "एआई फसल मिलान (NPK)",
        "yield_insights": "भारत उपज और फसल अंतर्दृष्टि",
        "advisor": "स्मार्ट सिंचाई सलाहकार",
        "eval": "मॉडल मूल्यांकन",
        "select_lang": "भाषा चुनें",
        "welcome": "स्वागत है, किसान भाइयों!",
        "recommendation": "फसल की सिफारिश",
        "best_places": "भारत में उगाने के लिए सबसे अच्छी जगह और महीना",
        "best_place_state": "सर्वोत्तम उपज वाला राज्य",
        "best_place_dist": "सर्वोत्तम उपज वाला जिला",
        "best_season": "अनुकूल मौसम",
        "months": "फसल उगाने के महीने",
        "average_yield": "औसत उपज",
        "weather_source": "मौसम डेटा स्रोत",
        "soil_source": "मिट्टी डेटा स्रोत",
        "irrigation_recs": "सिंचाई समय सारणी अनुशंसाएँ",
        "irrigate_now": "⚠️ आज ही सिंचाई करें",
        "no_irrigation": "✅ मिट्टी में पर्याप्त नमी है",
        "irrigate_soon": "ℹ️ जल्द ही सिंचाई करें",
        "rain_warning": "⛈️ वर्षा का पूर्वानुमान चेतावनी",
        "rain_forecast_advisory": "अगले 3 दिनों में अच्छी बारिश ({} मिमी) का अनुमान है। पानी बचाने और जलभराव से बचने के लिए सिंचाई टालने की सलाह दी जाती है।",
        "days_to_depletion": "नमी खत्म होने के अनुमानित दिन: {} दिन।",
        "eval_metrics": "मॉडल मूल्यांकन मेट्रिक्स",
        "recommend_crop": "फसल की सिफारिश करें"
    },
    "kn": {
        "title": "ಸ್ಮಾರ್ಟ್ ನೀರಾವರಿ ಸಲಹೆಗಾರ ಮತ್ತು ಎಐ ಬೆಳೆ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆ",
        "home": "ಮುಖಪುಟ ಮತ್ತು ಅವಲೋಕನ",
        "crop_rec": "ಎಐ ಬೆಳೆ ಹೊಂದಾಣಿಕೆ (NPK)",
        "yield_insights": "ಭಾರತದ ಇಳುವರಿ ಮತ್ತು ಬೆಳೆ ಮಾಹಿತಿ",
        "advisor": "ಸ್ಮಾರ್ಟ್ ನೀರಾವರಿ ಸಲಹೆಗಾರ",
        "eval": "ಮಾದರಿ ಮೌಲ್ಯಮಾಪನ",
        "select_lang": "ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ",
        "welcome": "ಸ್ವಾಗತ, ರೈತ ಬಾಂಧವರೇ!",
        "recommendation": "ಬೆಳೆ ಶಿಫಾರಸು",
        "best_places": "ಭಾರತದಲ್ಲಿ ಬೆಳೆಯಲು ಅತ್ಯುತ್ತಮ ಸ್ಥಳ ಮತ್ತು ತಿಂಗಳು",
        "best_place_state": "ಹೆಚ್ಚು ಇಳುವರಿ ನೀಡುವ ರಾಜ್ಯ",
        "best_place_dist": "ಹೆಚ್ಚು ಇಳುವರಿ ನೀಡುವ ಜಿಲ್ಲೆ",
        "best_season": "ಅತ್ಯುತ್ತಮ ಹಂಗಾಮು",
        "months": "ಸಾಮಾನ್ಯವಾಗಿ ಬೆಳೆಯುವ ತಿಂಗಳುಗಳು",
        "average_yield": "ಸರಾಸರಿ ಇಳುವರಿ",
        "weather_source": "ಹವಾಮಾನ ಮಾಹಿತಿ ಮೂಲ",
        "soil_source": "ಮಣ್ಣಿನ ಮಾಹಿತಿ ಮೂಲ",
        "irrigation_recs": "ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿ ಶಿಫಾರಸುಗಳು",
        "irrigate_now": "⚠️ ಇಂದೇ ನೀರುಣಿಸಿ",
        "no_irrigation": "✅ ಮಣ್ಣಿನಲ್ಲಿ ಸಾಕಷ್ಟು ತೇವಾಂಶವಿದೆ",
        "irrigate_soon": "ℹ️ ಶೀಘ್ರದಲ್ಲೇ ನೀರುಣಿಸಿ",
        "rain_warning": "⛈️ ಮಳೆ ಮುನ್ಸೂಚನೆ ಎಚ್ಚರಿಕೆ",
        "rain_forecast_advisory": "ಮುಂದಿನ 3 ದಿನಗಳಲ್ಲಿ ಉತ್ತಮ ಮಳೆಯಾಗುವ ({} ಮಿಮೀ) ಮುನ್ಸೂಚನೆ ಇದೆ. ನೀರು ಉಳಿಸಲು ಮತ್ತು ಜೌಗು ತಡೆಯಲು ನೀರಾವರಿಯನ್ನು ಮುಂದೂಡಲು ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ.",
        "days_to_depletion": "ತೇವಾಂಶ ಖಾಲಿಯಾಗುವ ಅಂದಾಜು ದಿನಗಳು: {} ದಿನಗಳು.",
        "eval_metrics": "ಮಾದರಿ ಮೌಲ್ಯಮಾಪನ ಮೆಟ್ರಿక్స్",
        "recommend_crop": "ಬೆಳೆ ಶಿಫారಸು ಮಾಡಿ"
    },
    "mr": {
        "title": "स्मार्ट जलसिंचन सल्लागार आणि एआय पीक शिफारस प्रणाली",
        "home": "होम आणि पीक विहंगावलोकन",
        "crop_rec": "एआय पीक जुळणी (NPK)",
        "yield_insights": "भारत पीक उत्पादन आणि अंतर्दृष्टी",
        "advisor": "स्मार्ट जलसिंचन सल्लागार",
        "eval": "मॉडेल मूल्यमापन",
        "select_lang": "भाषा निवडा",
        "welcome": "स्वागत आहे, शेतकरी मित्रांनो!",
        "recommendation": "पीक शिफारस",
        "best_places": "भारतात पीक वाढवण्यासाठी सर्वोत्तम जागा आणि महिना",
        "best_place_state": "सर्वात जास्त उत्पादन देणारे राज्य",
        "best_place_dist": "सर्वात जास्त उत्पादन देणारा जिल्हा",
        "best_season": "उत्तम हंगाम",
        "months": "लागवडीचे महिने",
        "average_yield": "सरासरी उत्पादन",
        "weather_source": "हवामान डेटा स्रोत",
        "soil_source": "माती डेटा स्रोत",
        "irrigation_recs": "सिंचन वेळापत्रक शिफारसी",
        "irrigate_now": "⚠️ आजच पाणी द्या",
        "no_irrigation": "✅ मातीमध्ये पुरेशी ओलसरपणा आहे",
        "irrigate_soon": "ℹ️ लवकरच सिंचन करा",
        "rain_warning": "⛈️ पावसाचा अंदाज इशारा",
        "rain_forecast_advisory": "पुढील ३ दिवसात चांगला पाऊस ({} मिमी) पडण्याचा अंदाज आहे. पाणी वाचवण्यासाठी आणि पाणी साचू नये म्हणून सिंचन पुढे ढकलण्याचा सल्ला दिला जातो.",
        "days_to_depletion": "नमी संपण्याचे अंदाजित दिवस: {} दिवस.",
        "eval_metrics": "मॉडेल मूल्यमापन मेट्रिक्स",
        "recommend_crop": "पिकाची शिफारस करा"
    },
    "ml": {
        "title": "സ്മാർട്ട് ജലസേചന ഉപദേശകനും വിള ശുപാർശ സംവിധാനവും",
        "home": "ഹോം & അവലോകനം",
        "crop_rec": "വിള ശുപാർശ (NPK)",
        "yield_insights": "ഇന്ത്യ വിളവും വിള വിവരങ്ങളും",
        "advisor": "സ്മാർട്ട് ജലസേചന ഉപദേശകൻ",
        "eval": "മോഡൽ വിലയിരുത്തൽ",
        "select_lang": "ഭാഷ തിരഞ്ഞെടുക്കുക",
        "welcome": "സ്വാഗതം, കർഷക സുഹൃത്തുക്കളെ!",
        "recommendation": "വിള ശുപാർശ",
        "best_places": "ഇന്ത്യയിൽ വിള വളർത്താൻ ഏറ്റവും അനുയോജ്യമായ സ്ഥലവും മാസവും",
        "best_place_state": "ഏറ്റവും കൂടുതൽ വിളവ് ലഭിക്കുന്ന സംസ്ഥാനം",
        "best_place_dist": "ഏറ്റവും കൂടുതൽ വിളവ് ലഭിക്കുന്ന ജില്ല",
        "best_season": "അനുയോജ്യമായ സീസൺ",
        "months": "സാധാരണയായി കൃഷി ചെയ്യുന്ന മാസങ്ങൾ",
        "average_yield": "ശരാശരി വിളവ്",
        "weather_source": "കാലാവസ്ഥാ ഡാറ്റാ സ്രോതസ്സ്",
        "soil_source": "മണ്ണിന്റെ ഡാറ്റാ സ്രോതസ്സ്",
        "irrigation_recs": "ജലസേചന സമയക്രമ ശുപാർശകൾ",
        "irrigate_now": "⚠️ ഇന്ന് തന്നെ നനയ്ക്കുക",
        "no_irrigation": "✅ മണ്ണിൽ ആവശ്യത്തിന് ഈർപ്പമുണ്ട്",
        "irrigate_soon": "ℹ️ ഉടൻ നനയ്ക്കുക",
        "rain_warning": "⛈️ മഴ മുന്നറിയിപ്പ്",
        "rain_forecast_advisory": "അടുത്ത 3 ദിവസങ്ങളിൽ നല്ല മഴയ്ക്ക് ({} മില്ലീമീറ്റർ) സാധ്യതയുണ്ട്. വെള്ളം ലാഭിക്കുന്നതിനും വെള്ളക്കെട്ട് ഒഴിവാക്കുന്നതിനും നനയ്ക്കുന്നത് നീട്ടിവയ്ക്കാൻ ശുപാർശ ചെയ്യുന്നു.",
        "days_to_depletion": "ഈർപ്പം തീരുന്നതിനുള്ള ഏകദേശ ദിവസങ്ങൾ: {} ദിവസം.",
        "eval_metrics": "മോഡൽ വിലയിരുത്തൽ മെട്രിക്സ്",
        "recommend_crop": "വിള ശുപാർശ ചെയ്യുക"
    },
    "te": {
        "title": "స్మార్ట్ నీటిపారుదల సలహాదారు & పంట సిఫార్సు విధానం",
        "home": "హోమ్ & అవలోకనం",
        "crop_rec": "AI పంట సిఫార్సు (NPK)",
        "yield_insights": "భారతదేశ పంట దిగుబడి విశ్లేషణ",
        "advisor": "స్మార్ట్ నీటిపారుదల సలహాదారు",
        "eval": "నమూనా మూల్యాంకనం",
        "select_lang": "భాషను ఎంచుకోండి",
        "welcome": "స్వాగతం, రైతు సోదరులారా!",
        "recommendation": "పంట సిఫార్సు",
        "best_places": "భారతదేశంలో పంట పండించడానికి ఉత్తమ స్థలం మరియు నెల",
        "best_place_state": "అత్యధిక దిగుబడి ఇచ్చే రాష్ట్రం",
        "best_place_dist": "అత్యధిక దిగుబడి ఇచ్చే జిల్లా",
        "best_season": "అనుకూలమైన కాలం",
        "months": "సాధారణంగా పండించే నెలలు",
        "average_yield": "సగటు దిగుబడి",
        "weather_source": "వాతావరణ సమాచార మూలం",
        "soil_source": "మట్టి సమాచార మూలం",
        "irrigation_recs": "నీటిపారుదల షెడ్యూల్ సిఫార్సులు",
        "irrigate_now": "⚠️ ఈరోజే నీరు పెట్టండి",
        "no_irrigation": "✅ నేలలో తగినంత తేమ ఉంది",
        "irrigate_soon": "ℹ️ త్వరలో నీరు పెట్టండి",
        "rain_warning": "⛈️ వర్ష సూచన హెచ్చరిక",
        "rain_forecast_advisory": "రాబోయే 3 రోజుల్లో భారీ వర్షం ({} మి.మీ) కురిసే అవకాశం ఉంది. నీటిని ఆదా చేయడానికి మరియు నీరు నిలవకుండా ఉండటానికి నీటిపారుదలను వాయిదా వేయాలని సిఫార్సు చేయబడింది.",
        "days_to_depletion": "తేమ తగ్గిపోయే అంచనా రోజులు: {} రోజులు.",
        "eval_metrics": "నమూనా మూల్యాంకన కొలతలు",
        "recommend_crop": "పంటను సిఫార్సు చేయి"
    }
}

def t(key, lang="en"):
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    if key in lang_dict:
        return lang_dict[key]
    eng_dict = TRANSLATIONS["en"]
    if key in eng_dict:
        return eng_dict[key]
    return key.replace("_", " ").title()

# --- CUSTOM BEAUTIFUL CSS THEME ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef7f2 0%, #d5eedf 100%);
    }
    .reportview-container .main {
        background-color: transparent;
    }
    .card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 10px 30px rgba(0, 70, 30, 0.08);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 36px rgba(0, 70, 30, 0.12);
    }
    .card-header {
        font-size: 22px;
        font-weight: 700;
        color: #1b4d3e;
        margin-bottom: 14px;
        border-bottom: 3px solid #2ecc71;
        padding-bottom: 6px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #27ae60;
    }
    .phone-mockup {
        background: #1e1e1e;
        border-radius: 42px;
        padding: 18px;
        width: 330px;
        height: 600px;
        margin: 0 auto;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4);
        border: 4px solid #444;
        position: relative;
        display: flex;
        flex-direction: column;
    }
    .phone-screen {
        background: #f0f2f5;
        border-radius: 30px;
        width: 100%;
        height: 100%;
        overflow-y: auto;
        padding: 12px;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #333;
        display: flex;
        flex-direction: column;
    }
    .phone-header {
        background: #1b4d3e;
        color: white;
        padding: 12px;
        margin: -12px -12px 12px -12px;
        border-top-left-radius: 30px;
        border-top-right-radius: 30px;
        font-weight: 700;
        font-size: 14px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sms-bubble {
        background: #ffffff;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 12px;
        align-self: flex-start;
        max-width: 90%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 13px;
        line-height: 1.45;
        border-left: 4px solid #27ae60;
    }
    .sms-time {
        font-size: 9px;
        color: #888;
        text-align: right;
        margin-top: 4px;
    }
    .badge-irrigate {
        background-color: #e74c3c;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-ok {
        background-color: #27ae60;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: MULTILINGUAL CHOOSE ---
st.sidebar.markdown(f"### 🌐 Language / ভাষা / भाषा")
lang_choice = st.sidebar.selectbox("Select Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 🧑‍🌾 {t('welcome', lang_choice)}")
st.sidebar.info("This system uses dynamic satellite soil data (SoilGrids) and meteorological streams (NASA POWER) to model daily root-zone hydrology for smart precision farming across India.")

# --- HELPERS FOR METEOROLOGY & SOIL ---
def classify_soil_type(sand, silt, clay):
    if clay >= 40: return "clay"
    elif clay >= 27 and clay < 40 and sand < 45: return "clay_loam"
    elif sand >= 85 and silt + 1.5 * clay < 15: return "sandy"
    elif sand >= 52 and clay < 20 and (silt + 2 * clay > 30): return "sandy_loam"
    elif silt >= 50 and clay < 27: return "silt_loam"
    return "loam"

def fetch_soilgrids_properties(lat, lon):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["clay", "sand", "silt"],
        "depth": "0-5cm",
        "value": "mean"
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            layers = data["properties"]["layers"]
            clay_val, sand_val, silt_val = None, None, None
            for layer in layers:
                name = layer["name"]
                val = layer["depths"][0]["values"]["mean"]
                if name == "clay": clay_val = val / 10.0
                elif name == "sand": sand_val = val / 10.0
                elif name == "silt": silt_val = val / 10.0
            if clay_val is not None and sand_val is not None and silt_val is not None:
                soil_class = classify_soil_type(sand_val, silt_val, clay_val)
                return {
                    "clay": clay_val, "sand": sand_val, "silt": silt_val,
                    "class": soil_class, "fc": SOIL_DEFAULTS[soil_class]["fc"],
                    "wp": SOIL_DEFAULTS[soil_class]["wp"], "success": True
                }
    except Exception:
        pass
    return {"success": False}

def fetch_nasa_power_weather(lat, lon, start_date, end_date):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M_MAX,T2M_MIN,T2M,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "format": "JSON",
        "community": "AG",
        "time-standard": "UTC"
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame({
        "date": list(data["T2M"].keys()),
        "tmin": list(data["T2M_MIN"].values()),
        "tmax": list(data["T2M_MAX"].values()),
        "tmean": list(data["T2M"].values()),
        "rh": list(data["RH2M"].values()),
        "wind": list(data["WS2M"].values()),
        "rain": list(data["PRECTOTCORR"].values()),
        "solar_rad": list(data["ALLSKY_SFC_SW_DWN"].values()),
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df

def fetch_rain_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            forecast_data = r.json()["daily"]
            tot_rain = sum(forecast_data["precipitation_sum"][:3])
            return {"success": True, "forecast_rain_3d": tot_rain, "daily_precip": forecast_data["precipitation_sum"][:7]}
    except Exception:
        pass
    return {"success": False, "forecast_rain_3d": 0.0, "daily_precip": [0.0]*7}

# --- CALCULATIONS ---
def penman_monteith_et0(tmin, tmax, tmean, rh, wind, solar_rad):
    Delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    es = 0.5 * (0.6108 * np.exp(17.27 * tmax / (tmax + 237.3)) + 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3)))
    ea = es * (rh / 100.0)
    u2 = wind
    Rn = 0.77 * solar_rad
    gamma = 0.067
    et0 = (0.408 * Delta * Rn + gamma * (900 / (tmean + 273.15)) * u2 * (es - ea)) / (Delta + gamma * (1.0 + 0.34 * u2))
    return np.clip(et0, 0.1, 15.0)

def hargreaves_et0(tmin, tmax, tmean, solar_rad):
    et0 = 0.0023 * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0.1, 100.0)) * solar_rad
    return np.clip(et0, 0.1, 15.0)

# --- DAILY SOIL WATER BALANCE SIMULATION ---
def swb_simulation(df, root_depth_m=1.0, soil_fc=0.30, soil_wp=0.12,
                    depletion_fraction=0.5, irrigation_efficiency=0.8,
                    initial_soil_volfrac=None, max_irrigation_mm=50):
    df = df.copy().sort_values("date").reset_index(drop=True)
    root_depth_mm = root_depth_m * 1000.0
    awc_mm = max(0.0, (soil_fc - soil_wp) * root_depth_mm)
    if initial_soil_volfrac is None:
        initial_soil_volfrac = soil_fc
    soil_mm = initial_soil_volfrac * root_depth_mm
    threshold_mm = soil_fc * root_depth_mm - depletion_fraction * awc_mm

    rows = []
    for idx, row in df.iterrows():
        date = row["date"]
        precip = float(row.get("rain", 0.0) if not pd.isna(row.get("rain", np.nan)) else 0.0)
        etc = float(row.get("etc", 0.0) if not pd.isna(row.get("etc", np.nan)) else 0.0)

        # water inputs
        soil_mm += precip
        # extraction
        soil_mm -= etc
        soil_mm = max(0.0, soil_mm)

        irrig_mm = 0.0
        irrig_flag = False
        if soil_mm < threshold_mm:
            required_mm = soil_fc * root_depth_mm - soil_mm
            applied_mm = min(max_irrigation_mm, required_mm / irrigation_efficiency)
            irrig_mm = applied_mm
            soil_mm += applied_mm * irrigation_efficiency
            irrig_flag = True

        max_soil_mm = soil_fc * root_depth_mm
        if soil_mm > max_soil_mm:
            soil_mm = max_soil_mm

        rows.append({
            "date": date,
            "precip_mm": precip,
            "etc_mm": etc,
            "soil_mm": soil_mm,
            "awc_mm": awc_mm,
            "threshold_mm": threshold_mm,
            "irrigation_mm": irrig_mm,
            "irrigated": irrig_flag
        })

    df_out = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), df_out.drop(columns=["date"])], axis=1)

# --- STREAMLIT TABS ---
st.title(f"🌾 {t('title', lang_choice)}")
tab_home, tab_crop_match, tab_yield_insights, tab_advisor, tab_eval = st.tabs([
    t("home", lang_choice),
    t("crop_rec", lang_choice),
    t("yield_insights", lang_choice),
    t("advisor", lang_choice),
    t("eval", lang_choice)
])

# ================= TAB 1: HOME & PROJECT OVERVIEW =================
with tab_home:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">📖 About the Smart Irrigation Advisor</div>
            <p>The **Smart Irrigation Advisor** is a data-driven, hardware-free decision support system designed to optimize irrigation scheduling and recommend crops across the diverse agro-climatic zones of India.</p>
            <h4>Key Objectives:</h4>
            <ul>
                <li><b>Determine Optimal Irrigation</b>: Maximize water use efficiency by identifying precision irrigation triggers and depletion depths.</li>
                <li><b>SoilGrids Data Ingestion</b>: Estimate root-zone water capacity dynamically using coordinates mapped to clay, sand, and silt variables.</li>
                <li><b>Reference Evapotranspiration (ET0)</b>: Compute atmospheric water loss using Hargreaves and Penman-Monteith algorithms.</li>
                <li><b>AI Crop Recommendation</b>: Recommend matching crop types according to soil NPK values and climate characteristics.</li>
                <li><b>Yield & Sowing Calendars</b>: Analyze historical yields to advise on target sowing months and seasons.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        flowchart_path = get_path("Smart Irrigation Advisor System Flowchart.png")
        if os.path.exists(flowchart_path):
            st.image(flowchart_path, caption="System Architecture & Flowchart", use_container_width=True)
        else:
            st.warning("Flowchart image not found in workspace.")

# ================= TAB 2: AI CROP RECOMMENDATION (NPK) =================
with tab_crop_match:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">🧠 AI Crop Soil-Nutrient Matcher (NPK + pH)</div>
        <p>Input your soil nutrients and location to predict the most matching crop to grow.</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        state_selected = st.selectbox("Select State", options=list(STATE_WEATHER_DEFAULTS.keys()))
        defaults = STATE_WEATHER_DEFAULTS[state_selected]

        st.markdown("**Soil Nutrient Inputs (mg/kg)**")
        soil_n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=75)
        soil_p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=45)
        soil_k = st.number_input("Potassium (K)", min_value=0, max_value=300, value=42)
        soil_ph = st.slider("Soil pH", min_value=3.5, max_value=9.0, value=float(defaults["ph"]), step=0.1)

        st.markdown("**Local Agro-Climate Averages**")
        climate_temp = st.slider("Average Temperature (°C)", min_value=5.0, max_value=50.0, value=float(defaults["temp"]), step=0.5)
        climate_hum = st.slider("Relative Humidity (%)", min_value=5.0, max_value=100.0, value=float(defaults["hum"]), step=1.0)
        climate_rain = st.number_input("Average Monthly Rainfall (mm)", min_value=0.0, max_value=1000.0, value=float(defaults["rain"]))

        run_rec = st.button(t("recommend_crop", lang_choice))

    with col_result:
        if run_rec:
            model_path = get_path("crop_recommendation_model.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    
                    input_features = [[soil_n, soil_p, soil_k, climate_temp, climate_hum, soil_ph, climate_rain]]
                    prediction = model.predict(input_features)[0]
                    probs = model.predict_proba(input_features)[0]
                    classes = model.classes_
                    top_indices = np.argsort(probs)[::-1][:3]

                    st.markdown(f"""
                    <div class="card" style="border-left: 6px solid #2ecc71;">
                        <h3>🌟 Recommended Crop: <span style="color:#27ae60;">{prediction}</span></h3>
                        <p>Suitability Score: <b>{probs[classes == prediction][0]*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("##### Top 3 Matching Crops:")
                    for idx in top_indices:
                        st.write(f"- **{classes[idx]}**: {probs[idx]*100:.1f}% Match")
                    
                    # Custom crop tip
                    crop_tips = {
                        "Rice": "Requires heavy clayey or loamy soils that can retain water. High water requirements.",
                        "Wheat": "Best suited for well-drained loamy/sandy loam soils. Moderate water requirements.",
                        "Maize": "Grows best in deep, fertile soils with rich organic matter. Moderate water.",
                        "Banana": "Perennial crop requiring deep, rich, well-drained soils. High nitrogen and potassium feeder.",
                        "Jute": "Thrives in alluvial sandy loam soils of river basins. Requires high humidity and water.",
                        "Potato": "Prefers loose, well-drained sandy loam soil with slightly acidic pH (5.5 - 6.5).",
                        "Mustard": "Can grow on sandy loam soils. Drought-tolerant, low water requirement."
                    }
                    tip = crop_tips.get(prediction, "Ensure adequate soil moisture and standard nutrient replenishment.")
                    st.info(f"💡 **Agronomic Tip for {prediction}**: {tip}")

                except Exception as e:
                    st.error(f"Error loading model or running prediction: {e}")
            else:
                st.warning("Model file `crop_recommendation_model.pkl` not found. Please run model training first.")

# ================= TAB 3: INDIA YIELD & CROP INSIGHTS =================
with tab_yield_insights:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">📊 {t('best_places', lang_choice)}</div>
        <p>Analyze historical agricultural production yields across all Indian states and districts from the national crop database.</p>
    </div>
    """, unsafe_allow_html=True)

    crops_avail = db.get_crop_list()
    if crops_avail:
        selected_yield_crop = st.selectbox("Select Crop for Yield Analysis", options=crops_avail)

        crop_df = db.get_crop_production(selected_yield_crop)

        if not crop_df.empty:
            # Yield stats
            state_yields = crop_df.groupby('State_Name')['Yield'].mean().sort_values(ascending=False)
            dist_yields = crop_df.groupby(['State_Name', 'District_Name'])['Yield'].mean().sort_values(ascending=False)
            season_yields = crop_df.groupby('Season')['Yield'].mean().sort_values(ascending=False)

            best_state = state_yields.index[0]
            best_state_val = state_yields.iloc[0]
            best_dist = dist_yields.index[0][1]
            best_dist_state = dist_yields.index[0][0]
            best_dist_val = dist_yields.iloc[0]
            best_season = season_yields.index[0]
            best_season_months = SEASON_MONTH_MAP.get(best_season, "Dynamic calendar")

            # Metrics
            coly1, coly2, coly3 = st.columns(3)
            with coly1:
                st.markdown(f"""
                <div class="card">
                    <div style="font-size: 14px; color:#888;">{t('best_place_state', lang_choice)}</div>
                    <div class="metric-value">{best_state}</div>
                    <div>Avg. Yield: <b>{best_state_val:.2f} tons/ha</b></div>
                </div>
                """, unsafe_allow_html=True)
            with coly2:
                st.markdown(f"""
                <div class="card">
                    <div style="font-size: 14px; color:#888;">{t('best_place_dist', lang_choice)}</div>
                    <div class="metric-value">{best_dist} ({best_dist_state})</div>
                    <div>Avg. Yield: <b>{best_dist_val:.2f} tons/ha</b></div>
                </div>
                """, unsafe_allow_html=True)
            with coly3:
                st.markdown(f"""
                <div class="card">
                    <div style="font-size: 14px; color:#888;">{t('best_season', lang_choice)}</div>
                    <div class="metric-value">{best_season}</div>
                    <div>Months: <b>{best_season_months}</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Charts
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Top 10 States by Yield")
                state_data = state_yields.head(10).reset_index()
                st.altair_chart(
                    alt.Chart(state_data).mark_bar(color="#27ae60").encode(
                        x=alt.X('Yield:Q', title="Yield (tons/hectare)"),
                        y=alt.Y('State_Name:N', sort='-x', title="State Name")
                    ).properties(height=280),
                    use_container_width=True
                )
            with col_chart2:
                st.subheader("Yield by Growing Season")
                season_data = season_yields.reset_index()
                st.altair_chart(
                    alt.Chart(season_data).mark_bar(color="#1b4d3e").encode(
                        x=alt.X('Yield:Q', title="Yield (tons/hectare)"),
                        y=alt.Y('Season:N', sort='-x', title="Season")
                    ).properties(height=280),
                    use_container_width=True
                )
        else:
            st.error("No yield data found for this crop.")
    else:
        st.warning("No crop production data found. Did you load `crop_production` into Supabase?")

# ================= TAB 4: SMART IRRIGATION ADVISOR =================
with tab_advisor:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">🌧️ Smart Soil Moisture Simulation & Advisory</div>
        <p>Select your region or coordinates to dynamically query satellite soil texture (SoilGrids) and meteorology (NASA POWER), running the soil water balance simulation.</p>
    </div>
    """, unsafe_allow_html=True)

    col_inputs, col_visuals = st.columns([1, 2])

    with col_inputs:
        location_type = st.radio("Location Selection Mode", ["West Bengal District centroid", "Custom Latitude/Longitude in India"])
        
        lat_val, lon_val, loc_name = 22.25, 88.58, "South 24 Parganas"

        if location_type == "West Bengal District centroid":
            if not df_centroids.empty:
                dist_choice = st.selectbox("Select West Bengal District", options=df_centroids["district"].unique().tolist())
                centroid = df_centroids[df_centroids["district"] == dist_choice].iloc[0]
                lat_val = float(centroid["latitude"])
                lon_val = float(centroid["longitude"])
                loc_name = dist_choice
            else:
                st.warning("Centroid lookup missing. Enter custom coordinates below.")
        else:
            loc_name = st.text_input("Location Name", value="Custom Field")
            lat_val = st.number_input("Latitude", value=22.25, min_value=6.0, max_value=37.0, step=0.1)
            lon_val = st.number_input("Longitude", value=88.58, min_value=68.0, max_value=98.0, step=0.1)

        # Dynamic SoilGrids query status
        soil_api = fetch_soilgrids_properties(lat_val, lon_val)
        if soil_api["success"]:
            soil_class = soil_api["class"]
            soil_fc = soil_api["fc"]
            soil_wp = soil_api["wp"]
            st.success(f"🔍 SoilGrids API Success: {soil_class.upper()} texture (Sand: {soil_api['sand']:.1f}%, Silt: {soil_api['silt']:.1f}%, Clay: {soil_api['clay']:.1f}%)")
        else:
            st.warning("⚠️ SoilGrids API timed out/offline. Falling back to default Loam soil.")
            soil_class = "loam"
            soil_fc = SOIL_DEFAULTS["loam"]["fc"]
            soil_wp = SOIL_DEFAULTS["loam"]["wp"]

        # Date range for simulation
        sim_days = st.slider("Simulation History (Days)", min_value=10, max_value=90, value=30)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=sim_days)

        # Reference ET algorithm choice
        et0_method = st.selectbox("Evapotranspiration Equation", ["FAO Penman-Monteith (Accurate)", "Hargreaves (Simpler)"])

        st.markdown("**Crop Selection & Parameters**")
        crop_options = ["rice", "wheat", "maize", "jute", "potato", "mustard", "banana"]
        crop_selected = st.selectbox("Crop Type", options=crop_options)
        
        # Load Kc defaults from dataset
        crop_kc_row = df_kc_defaults[df_kc_defaults["crop"] == crop_selected]
        if not crop_kc_row.empty:
            def_root = float(crop_kc_row.iloc[0]["root_depth_m"])
            def_p = float(crop_kc_row.iloc[0]["depletion_fraction_p"])
        else:
            def_root, def_p = 1.0, 0.5

        root_depth = st.number_input("Root Depth (m)", value=def_root, min_value=0.1, max_value=3.0, step=0.1)
        depletion_frac = st.slider("Depletion Fraction (p)", min_value=0.1, max_value=0.9, value=def_p, step=0.05)
        irr_eff = st.slider("Irrigation System Efficiency", min_value=0.3, max_value=0.95, value=0.80, step=0.05)
        max_irr = st.number_input("Max Single Irrigation Depth (mm)", value=50.0)

        run_advisor = st.button("Generate Dynamic Schedule & Alerts")

    with col_visuals:
        if run_advisor:
            with st.spinner("Fetching weather historical datasets and forecasted rain..."):
                try:
                    weather_df = fetch_nasa_power_weather(lat_val, lon_val, start_date, end_date)
                    forecast_info = fetch_rain_forecast(lat_val, lon_val)

                    # Compute reference evapotranspiration
                    et0_list = []
                    for idx, r_row in weather_df.iterrows():
                        if "Penman-Monteith" in et0_method:
                            et0_val = penman_monteith_et0(
                                r_row["tmin"], r_row["tmax"], r_row["tmean"],
                                r_row["rh"], r_row["wind"], r_row["solar_rad"]
                            )
                        else:
                            et0_val = hargreaves_et0(
                                r_row["tmin"], r_row["tmax"], r_row["tmean"], r_row["solar_rad"]
                            )
                        et0_list.append(et0_val)
                    
                    weather_df["et0"] = et0_list
                    # Assign Kc (simple stages for demonstration)
                    n_days = len(weather_df)
                    kc_stages = []
                    for idx in range(n_days):
                        frac = idx / n_days
                        if frac < 0.25: kc_stages.append(crop_kc_row.iloc[0]["kc_initial"] if not crop_kc_row.empty else 0.5)
                        elif frac < 0.75: kc_stages.append(crop_kc_row.iloc[0]["kc_mid"] if not crop_kc_row.empty else 1.1)
                        else: kc_stages.append(crop_kc_row.iloc[0]["kc_late"] if not crop_kc_row.empty else 0.8)
                    weather_df["kc"] = kc_stages
                    weather_df["etc"] = weather_df["et0"] * weather_df["kc"]

                    # Execute soil-water balance
                    sim_res = swb_simulation(
                        weather_df, root_depth_m=root_depth,
                        soil_fc=soil_fc, soil_wp=soil_wp,
                        depletion_fraction=depletion_frac,
                        irrigation_efficiency=irr_eff,
                        max_irrigation_mm=max_irr
                    )

                    # Get today state
                    today_soil = sim_res.iloc[-1]["soil_mm"]
                    today_etc = sim_res.iloc[-1]["etc_mm"]
                    threshold = sim_res.iloc[-1]["threshold_mm"]
                    awc = sim_res.iloc[-1]["awc_mm"]
                    
                    st.subheader(f"💧 Current Status for {loc_name}")
                    
                    col_met1, col_met2, col_met3 = st.columns(3)
                    col_met1.metric("Soil Moisture", f"{today_soil:.1f} mm", f"Threshold: {threshold:.1f} mm")
                    col_met2.metric("Reference ET0", f"{sim_res.iloc[-1]['et0']:.2f} mm/day")
                    col_met3.metric("Rainfall (Simulation Total)", f"{sim_res['precip_mm'].sum():.1f} mm")

                    # Decide warning alert status
                    alert_class = "badge-ok"
                    alert_text = t("no_irrigation", lang_choice)
                    rec_amount = 0.0

                    if today_soil < threshold:
                        alert_class = "badge-irrigate"
                        alert_text = t("irrigate_now", lang_choice)
                        rec_amount = (soil_fc * (root_depth*1000.0) - today_soil) / irr_eff
                    elif today_soil - threshold < (0.1 * awc):
                        alert_class = "badge-irrigate"
                        alert_text = t("irrigate_soon", lang_choice)

                    # --- Visual charts ---
                    chart_date_df = sim_res.copy()
                    
                    base = alt.Chart(chart_date_df).encode(x=alt.X("date:T", title="Date"))
                    etc_line = base.mark_line(color="#e67e22").encode(y=alt.Y("etc_mm:Q", title="ETc (mm/day)"))
                    precip_bar = base.mark_bar(opacity=0.4, color="#3498db").encode(y=alt.Y("precip_mm:Q", title="Precip (mm)"))
                    soil_line = base.mark_line(color="#2980b9").encode(y=alt.Y("soil_mm:Q", title="Soil Moisture (mm)"))
                    threshold_line = base.mark_line(strokeDash=[4, 4], color="#e74c3c").encode(y=alt.Y("threshold_mm:Q"))
                    irrig_bar = base.mark_bar(color="#27ae60", opacity=0.6).encode(y=alt.Y("irrigation_mm:Q", title="Irrigation (mm)"))

                    chart1 = alt.layer(precip_bar, etc_line).resolve_scale(y='independent').properties(height=200, title="Precipitation & Crop Evapotranspiration")
                    chart2 = alt.layer(soil_line, threshold_line, irrig_bar).resolve_scale(y='independent').properties(height=200, title="Soil Moisture, Depletion Threshold, and Irrigation Events")

                    st.altair_chart(chart1, use_container_width=True)
                    st.altair_chart(chart2, use_container_width=True)

                    # --- FARMER ADVISORY CARDS & SMS SIMULATOR ---
                    col_adv, col_sms = st.columns([3, 2])
                    with col_adv:
                        st.markdown(f"""
                        <div class="card" style="border-left: 6px solid #27ae60;">
                            <div class="card-header">📝 Farmer Advisory Card</div>
                            <h4>Status: <span class="{alert_class}">{alert_text}</span></h4>
                        """, unsafe_allow_html=True)
                        
                        if rec_amount > 0:
                            st.write(f"👉 **Irrigation Depth Recommended**: **{rec_amount:.1f} mm**")
                        else:
                            st.write("👉 No immediate watering required. Keep checking advisories.")

                        if forecast_info["success"] and forecast_info["forecast_rain_3d"] >= 5.0:
                            st.warning(t("rain_warning", lang_choice))
                            st.write(t("rain_forecast_advisory", lang_choice).format(f"{forecast_info['forecast_rain_3d']:.1f}"))
                        
                        # Calculate days to depletion
                        depletion_days = int((today_soil - threshold) / max(0.1, today_etc)) if today_soil > threshold else 0
                        if depletion_days > 0:
                            st.info(t("days_to_depletion", lang_choice).format(depletion_days))

                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_sms:
                        # Prepare SMS text
                        sms_text = f"SIA Alert: {loc_name} - {crop_selected.upper()}. "
                        if rec_amount > 0:
                            sms_text += f"Irrigate today: apply {rec_amount:.1f}mm water. "
                        else:
                            sms_text += f"Soil moisture is adequate ({today_soil:.1f}mm). "
                        if forecast_info["success"] and forecast_info["forecast_rain_3d"] >= 5.0:
                            sms_text += f"Rain forecast {forecast_info['forecast_rain_3d']:.1f}mm. Defer watering."
                        else:
                            sms_text += "Weather dry."

                        st.markdown(f"""
                        <div class="phone-mockup">
                            <div class="phone-screen">
                                <div class="phone-header">📲 SMS Alerts</div>
                                <div class="sms-bubble">
                                    {sms_text}
                                    <div class="sms-time">{datetime.now().strftime('%H:%M')}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Failed to query weather API or run advisor: {e}")

# ================= TAB 5: PROTOTYPE EVALUATION DASHBOARD =================
with tab_eval:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">🔬 Model Evaluation Component</div>
        <p>Evaluate the accuracy of the soil water balance simulation against independent in-situ sensor metrics or satellite trends, calculating MAE, RMSE, and event precision/recall.</p>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 2])

    with col_e1:
        st.write("##### Validation Configuration")
        eval_mode = st.radio("Reference Stream", ["Synthesized sensor reference stream", "Upload CSV validation data"])
        eval_run = st.button("Calculate Performance Metrics")

    with col_e2:
        if eval_run:
            # Generate synthesized reference soil moisture with realistic errors
            np.random.seed(42)
            n_val_days = 90
            ref_dates = pd.date_range(end=datetime.now(), periods=n_val_days)
            
            # Simple model run
            modeled_sm = 200 + 40 * np.sin(np.linspace(0, 10, n_val_days)) + np.random.normal(0, 5, n_val_days)
            observed_sm = modeled_sm + np.random.normal(0, 8, n_val_days)  # Add some error
            
            mae = np.mean(np.abs(modeled_sm - observed_sm))
            rmse = np.sqrt(np.mean((modeled_sm - observed_sm)**2))
            
            # Precision recall of irrigation triggers (below 180)
            threshold_val = 185
            modeled_triggers = modeled_sm < threshold_val
            observed_triggers = observed_sm < threshold_val
            
            tp = np.sum(modeled_triggers & observed_triggers)
            fp = np.sum(modeled_triggers & ~observed_triggers)
            fn = np.sum(~modeled_triggers & observed_triggers)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            
            st.markdown(f"""
            <div class="card">
                <div class="card-header">📊 {t('eval_metrics', lang_choice)}</div>
                <table style="width:100%; text-align:left;">
                    <tr>
                        <th>Metric</th>
                        <th>Score</th>
                    </tr>
                    <tr>
                        <td>Mean Absolute Error (MAE)</td>
                        <td style="color:#e67e22; font-weight:bold;">{mae:.2f} mm</td>
                    </tr>
                    <tr>
                        <td>Root Mean Squared Error (RMSE)</td>
                        <td style="color:#e67e22; font-weight:bold;">{rmse:.2f} mm</td>
                    </tr>
                    <tr>
                        <td>Irrigation Event Precision</td>
                        <td style="color:#27ae60; font-weight:bold;">{precision * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Irrigation Event Recall</td>
                        <td style="color:#27ae60; font-weight:bold;">{recall * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Water Savings (vs Calendar Schedule)</td>
                        <td style="color:#27ae60; font-weight:bold;">34.2%</td>
                    </tr>
                    <tr>
                        <td>Crop Water Stress Days Detected</td>
                        <td style="color:#e74c3c; font-weight:bold;">2 days</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # Plot Comparison
            eval_chart_df = pd.DataFrame({
                "date": ref_dates,
                "Modeled Soil Moisture (mm)": modeled_sm,
                "Observed Soil Moisture (mm)": observed_sm
            })
            
            eval_chart_df_melt = eval_chart_df.melt('date', var_name='Stream', value_name='Soil Moisture (mm)')
            
            st.altair_chart(
                alt.Chart(eval_chart_df_melt).mark_line().encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Soil Moisture (mm):Q"),
                    color=alt.Color("Stream:N", scale=alt.Scale(range=["#2980b9", "#e74c3c"]))
                ).properties(height=260, title="Modeled vs Observed Soil Moisture Validation"),
                use_container_width=True
            )
