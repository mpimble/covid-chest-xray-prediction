import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='COVID Chest X-rays',
)
st.title('Predicting COVID-19 Pneumonia Severity from Chest X-rays')

# -----------------------------------------------------------------------------
# Declare some useful functions.


# -----------------------------------------------------------------------------
# Draw the actual page
st.write("Michael Pimble, Damla Akdogan, Carrie Wang")

st.markdown("## Our motivation behind the project:\n- During the COVID-19 pandemic, it was important to know how sick a patient is to decide where to send patients for car\n- Manually evaluating severity is time consuming and scores can be inconsistent depending on the expert\n- Automated tools like ML helps doctors make faster decisions")

st.markdown("## Objective- To develop and compare a convolutional neural network (CNN) and a collaboartive filtering (CF) approach for predicting COVID-19 severity from chest X-ray images ")

st.markdown("## Our Methods:\n- Collaborative Filtering\n  - User-User: Each image was treated as a user with two severity scores. We masked one score per test image and manually computed cosine similarity to find the top-k similar users. Their ratings were combined using a weighted average to predict the missing score.\n  - Item-Item- Similar to user-user CF, but each prediction was essentially (known_value * sim(geo_mean, opac_mean)) since each image only had 2 features\n- Convolutional Neural Network\n  - Multi-output regression CNN\n  - 2 continuous outputs that predict geographic extent and opacity")
# Add some spacing
''
''
