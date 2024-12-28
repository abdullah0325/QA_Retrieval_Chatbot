# frontend.py
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF Q&A Assistant", page_icon=":books:")
st.title("PDF Question Answering Assistant")

def main():
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'csv']
    )
    
    if uploaded_files:
        try:
            with st.spinner("Uploading and processing files..."):
                for uploaded_file in uploaded_files:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }
                    response = requests.post(
                        f"{BACKEND_URL}/upload-pdf/",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"File {uploaded_file.name} processed successfully! "
                            f"Pages: {data['pages']}, Chunks: {data['chunks']}"
                        )
                    else:
                        st.error(
                            f"Error processing {uploaded_file.name}: "
                            f"{response.json().get('error', 'Unknown error')}"
                        )
        except Exception as e:
            st.error(f"Error uploading file: {e}")

    # Question answering
    question = st.text_input("Ask a question about the document:")
    if question:
        try:
            with st.spinner("Generating answer..."):
                response = requests.post(
                    f"{BACKEND_URL}/ask-question/",
                    json={"question": question}  # Properly formatted JSON
                )
                if response.status_code == 200:
                    st.subheader("Answer:")
                    st.write(response.json().get("answer"))
                else:
                    st.error(response.json().get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Error fetching answer: {e}")

if __name__ == "__main__":
    main()