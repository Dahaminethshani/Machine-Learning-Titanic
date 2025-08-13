import streamlit as st

st.title("Streamlit Test App")
st.write("🎉 If you can see this, Streamlit is working fine!")

name = st.text_input("Enter your name")
if name:
    st.success(f"Hello, {name}!")
