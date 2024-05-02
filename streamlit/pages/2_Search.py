# Page setting
import streamlit as st
import pandas as pd
import os
import sys
import warnings
import plotly.express as px


# Page setting
st.set_page_config(
    layout="wide", page_title="Movies recommandation", page_icon=":film_projector:"
)

st.markdown(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)