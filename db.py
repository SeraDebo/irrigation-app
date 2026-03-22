import streamlit as st
from supabase import create_client


url = st.secrets["https://cpwtjtevrkqfpldaqess.supabase.co"]
key = st.secrets["sb_publishable_s4QameDnJax6B-ZOohxG2g_Qc4xpW04"]

supabase = create_client(url, key)
