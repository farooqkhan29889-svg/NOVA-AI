import streamlit as st

st.title("Hallo i am Farooq Khan!")
st.write("I am learning Streamlit today for my new AI Assistant NOVA")

st.divider()

st.header("This is a Header")
st.subheader("This is subheader")

if st.button("click here"):
    st.write("submite your application")
    
st.divider()

name = st.text_input("Enter Your name:")
if name:
    st.write("hallo {name}")