import streamlit as st

st.set_page_config(
    page_title='COVID Chest X-rays',
)
st.title('Predicting COVID-19 Pneumonia Severity from Chest X-rays')

st.markdown("## Results")
st.markdown("User-User Collaborative Filtering:\n- Showed weak performance with an MAE of 1.53, RMSE of 1,098 and a negative R² score\n- Given that severity scores range from 0 to 8, these errors represent a substantial portion of the scale")
st.markdown("Item-Item Collaborative Filtering:\n- Performed better than user-user CF, opacity predictions had an MAE of 0.06, RMSE of 0.08, and an R² score of 0.92\n- Geographic predictions were worse with an R² score of 0.65")
st.image("data/item_item_opacity_mean.png")
st.image("data/item_item_geographic_mean.png")
st.markdown("Convolutional Neural Network:\n- Opacity predictions more accurate than geographic extent\n- Combined average MAE: 1.63\n- Model shows overfitting (train MAE: 1.25, test MAE: 1.63)")