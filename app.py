import streamlit as st
from main import rag_with_fallback, plot_extraction_trend, plot_category_bar, translate_to_en
import re
import os

# Streamlit UI
st.title("INGRES Groundwater Chatbot (07:56 PM IST, Sep 13, 2025)")

lang = st.selectbox("Select Language", ['English', 'Hindi', 'Bengali', 'Odia'])
visualize = st.checkbox("Show Visualization")
show_sources = st.checkbox("Show Sources")

query = st.text_input("Ask about groundwater (e.g., Category for Delhi Civil Lines 2024?):")
if query:
    lang_code = {'English': 'en', 'Hindi': 'hi', 'Bengali': 'bn', 'Odia': 'or'}[lang]
    en_query = translate_to_en(query)
    loc_match = re.search(r'(delhi|west bengal|odisha)?\s*(\w+)?\s*(\d{4})?', en_query, re.I)
    loc_info = f"{loc_match.group(1) or 'General'} {loc_match.group(2) or ''}, {loc_match.group(3) or '2025'}"

    answer, sources = rag_with_fallback(en_query, loc_info, lang_code)
    st.write("**Answer:**", answer)

    if visualize and any(w in en_query.lower() for w in ['trend', 'plot', 'category']):
        state = loc_match.group(1) or "Delhi"
        if 'category' in en_query.lower():
            year_str = loc_match.group(3) + "-25" if loc_match.group(3) else "2025"
            img_path, viz_text = plot_category_bar(state, year_str)
        else:
            loc = loc_match.group(2)
            img_path, viz_text = plot_extraction_trend(state, loc)
        st.image(img_path, caption=viz_text)

    if show_sources and sources:
        st.write("**Sources:**")
        for s in sources[:3]:
            st.write(f"- {s.metadata.get('location', 'Unknown')}, {s.metadata.get('year', 'Unknown')} (SGWD: {s.metadata.get('sgwd', 0):.1f}%)")

if st.button("Clear"):
    st.experimental_rerun()