import streamlit as st
import tempfile
import os

from ocr import (
    extract_text_from_image,
    extract_text_from_pdf
)

from summarizer import summarize_text


st.set_page_config(
    page_title="Smart Document Summarizer",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Smart Document OCR & Summarizer")

st.write(
    "Upload a PDF or image. The system will extract "
    "the text and generate a concise summary."
)


uploaded_file = st.file_uploader(
    "Upload your document",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button("🔍 Scan & Summarize"):

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name


        try:

            # =====================================
            # PDF
            # =====================================

            if file_extension == ".pdf":

                with st.spinner(
                    "Processing PDF..."
                ):

                    extracted_text = (
                        extract_text_from_pdf(
                            temp_path
                        )
                    )

            # =====================================
            # IMAGE
            # =====================================

            else:

                with st.spinner(
                    "Running OCR..."
                ):

                    extracted_text = (
                        extract_text_from_image(
                            temp_path
                        )
                    )


            # =====================================
            # DISPLAY OCR
            # =====================================

            st.subheader(
                "📝 Extracted Text"
            )

            if extracted_text:

                st.text_area(
                    "OCR Result",
                    extracted_text,
                    height=400
                )

            else:

                st.warning(
                    "No text could be detected."
                )


            # =====================================
            # SUMMARY
            # =====================================

            if extracted_text:

                with st.spinner(
                    "Generating summary..."
                ):

                    summary = summarize_text(
                        extracted_text
                    )


                st.subheader(
                    "📌 Summary"
                )

                st.success(summary)


        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )


        finally:

            if os.path.exists(temp_path):

                os.remove(temp_path)