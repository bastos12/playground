import streamlit as st


def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )

def header():
    st.title('Bienvenue')

def sidebar():
    st.sidebar.caption('Choisir le dataset')


if __name__ == '__main__':
    intro()
    header()
    sidebar()