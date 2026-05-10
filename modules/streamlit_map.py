import streamlit as st


def render_map(folium_map, height: int = 540, key: str | None = None) -> None:
    try:
        from streamlit_folium import st_folium

        st_folium(folium_map, height=height, use_container_width=True, key=key)
    except Exception:
        st.components.v1.html(folium_map._repr_html_(), height=height + 20, scrolling=False)
