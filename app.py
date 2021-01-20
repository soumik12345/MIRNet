import os
import subprocess
from PIL import Image
import streamlit as st
from mirnet.inference import Inferer


def main():
    st.markdown(
        '<h1 align="center">Low-light Image Enhancement using MIRNet</h1><hr>',
        unsafe_allow_html=True
    )
    inferer = Inferer()
    if not os.path.exists('low_light_weights_best.h5'):
        st.sidebar.text('Downloading Model weights...')
        inferer.download_weights('1sUlRD5MTRKKGxtqyYDpTv7T3jOW6aVAL')
        st.sidebar.text('Done')
    st.sidebar.text('Building MIRNet Model...')
    inferer.build_model(
        num_rrg=3, num_mrb=2, channels=64,
        weights_path='low_light_weights_best.h5'
    )
    st.sidebar.text('Done')
    uploaded_files = st.sidebar.file_uploader(
        'Please Upload your Low-light Images',
        accept_multiple_files=True
    )
    col_1, col_2 = st.beta_columns(2)
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            pil_image = Image.open(uploaded_file)
            original_image, output_image = inferer.infer_streamlit(pil_image)
            with col_1:
                st.image(
                    original_image, use_column_width=True,
                    caption='Original Image'
                )
            with col_2:
                st.image(
                    output_image, use_column_width=True,
                    caption='Predicted Image'
                )
            st.markdown('---')
    if not os.path.exists('low_light_weights_best.h5'):
        subprocess.run(['rm', 'low_light_weights_best.h5'])


if __name__ == '__main__':
    main()
