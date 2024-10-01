import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import beta as beta_dist, chi2, expon

st.set_page_config(page_title="Distribution Table", page_icon="📈")

st.sidebar.header("Distribution Table")

st.title('Distribution Table Generator')

def create_z_table(start, end, step):
    try:
        z_scores = np.arange(start, end + step, step)
        cumulative_probabilities = stats.norm.cdf(z_scores)
        
        df = pd.DataFrame({
            'Z-Score': z_scores,
            'Cumulative Probability': cumulative_probabilities
        })
        
        pivot_df = df.pivot_table(values='Cumulative Probability', 
                                  index=[z_scores.round(1)], 
                                  columns=[(z_scores * 100).astype(int) % 10 / 100])
        return pivot_df
    except Exception as e:
        st.error(f"Error generating Z-table: {e}")
        return pd.DataFrame()

def create_beta_table(start, end, step, alpha, beta):
    try:
        x_values = np.arange(start, end + step, step)
        cumulative_probabilities = beta_dist.cdf(x_values, alpha, beta)
        
        df = pd.DataFrame({
            'X-Value': x_values,
            'Cumulative Probability': cumulative_probabilities
        })
        
        pivot_df = df.pivot_table(values='Cumulative Probability', 
                                  index=[x_values.round(2)], 
                                  columns=[(x_values * 100).astype(int) % 100 / 100])
        return pivot_df
    except Exception as e:
        st.error(f"Error generating Beta-table: {e}")
        return pd.DataFrame()

def create_chi2_table(start, end, step, df):
    try:
        x_values = np.arange(start, end + step, step)
        cumulative_probabilities = chi2.cdf(x_values, df)
        
        df_table = pd.DataFrame({
            'X-Value': x_values,
            'Cumulative Probability': cumulative_probabilities
        })
        
        pivot_df = df_table.pivot_table(values='Cumulative Probability', 
                                        index=[x_values.round(2)], 
                                        columns=[(x_values * 100).astype(int) % 100 / 100])
        return pivot_df
    except Exception as e:
        st.error(f"Error generating Chi-Square table: {e}")
        return pd.DataFrame()

def create_expon_table(start, end, step, scale):
    try:
        x_values = np.arange(start, end + step, step)
        cumulative_probabilities = expon.cdf(x_values, scale=scale)
        
        df_table = pd.DataFrame({
            'X-Value': x_values,
            'Cumulative Probability': cumulative_probabilities
        })
        
        pivot_df = df_table.pivot_table(values='Cumulative Probability', 
                                        index=[x_values.round(2)], 
                                        columns=[(x_values * 100).astype(int) % 100 / 100])
        return pivot_df
    except Exception as e:
        st.error(f"Error generating Exponential table: {e}")
        return pd.DataFrame()

# Selector for distribution type
distribution_type = st.sidebar.selectbox(
    'Select Distribution Type',
    ('Normal Distribution', 'Beta Distribution', 'Chi-Square Distribution', 'Exponential Distribution')
)

if distribution_type == 'Normal Distribution':
    st.sidebar.subheader("Normal Distribution Parameters")
    start = st.sidebar.number_input('Start Z-score', value=-3.4, step=0.1, format="%.1f", key="start_z")
    end = st.sidebar.number_input('End Z-score', value=3.4, step=0.1, format="%.1f", key="end_z")
    step = st.sidebar.number_input('Step', value=0.01, step=0.01, format="%.2f", key="step_z")

    if st.sidebar.button('Generate Normal Table'):
        z_table = create_z_table(start, end, step)
        if not z_table.empty:
            st.write(z_table)
        
            @st.cache_data
            def convert_df(df):
                import io
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df.to_excel(writer, index=True)
                return buf.getvalue()

            excel_data = convert_df(z_table)
            
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name='standard_normal_distribution_table.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.error("No data to download. Please check the input parameters.")

elif distribution_type == 'Beta Distribution':
    st.sidebar.subheader("Beta Distribution Parameters")
    start_beta = st.sidebar.number_input('Start X-value', value=0.0, step=0.01, format="%.2f", key="start_beta")
    end_beta = st.sidebar.number_input('End X-value', value=1.0, step=0.01, format="%.2f", key="end_beta")
    step_beta = st.sidebar.number_input('Step', value=0.01, step=0.01, format="%.2f", key="step_beta")
    alpha = st.sidebar.number_input('Alpha (α)', value=2.0, step=0.1, key="alpha")
    beta = st.sidebar.number_input('Beta (β)', value=5.0, step=0.1, key="beta")

    if st.sidebar.button('Generate Beta Table'):
        beta_table = create_beta_table(start_beta, end_beta, step_beta, alpha, beta)
        if not beta_table.empty:
            st.write(beta_table)
        
            @st.cache_data
            def convert_df(df):
                import io
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df.to_excel(writer, index=True)
                return buf.getvalue()

            excel_data = convert_df(beta_table)
            
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name='beta_distribution_table.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.error("No data to download. Please check the input parameters.")

elif distribution_type == 'Chi-Square Distribution':
    st.sidebar.subheader("Chi-Square Distribution Parameters")
    start_chi2 = st.sidebar.number_input('Start X-value', value=0.0, step=0.01, format="%.2f", key="start_chi2")
    end_chi2 = st.sidebar.number_input('End X-value', value=10.0, step=0.01, format="%.2f", key="end_chi2")
    step_chi2 = st.sidebar.number_input('Step', value=0.01, step=0.01, format="%.2f", key="step_chi2")
    df_chi2 = st.sidebar.number_input('Degrees of Freedom', value=2, step=1, key="df_chi2")

    if st.sidebar.button('Generate Chi-Square Table'):
        chi2_table = create_chi2_table(start_chi2, end_chi2, step_chi2, df_chi2)
        if not chi2_table.empty:
            st.write(chi2_table)
        
            @st.cache_data
            def convert_df(df):
                import io
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df.to_excel(writer, index=True)
                return buf.getvalue()

            excel_data = convert_df(chi2_table)
            
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name='chi_square_distribution_table.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.error("No data to download. Please check the input parameters.")

elif distribution_type == 'Exponential Distribution':
    st.sidebar.subheader("Exponential Distribution Parameters")
    start_expon = st.sidebar.number_input('Start X-value', value=0.0, step=0.01, format="%.2f", key="start_expon")
    end_expon = st.sidebar.number_input('End X-value', value=10.0, step=0.01, format="%.2f", key="end_expon")
    step_expon = st.sidebar.number_input('Step', value=0.01, step=0.01, format="%.2f", key="step_expon")
    scale = st.sidebar.number_input('Scale (λ)', value=1.0, step=0.1, key="scale")

    if st.sidebar.button('Generate Exponential Table'):
        expon_table = create_expon_table(start_expon, end_expon, step_expon, scale)
        if not expon_table.empty:
            st.write(expon_table)
        
            @st.cache_data
            def convert_df(df):
                import io
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df.to_excel(writer, index=True)
                return buf.getvalue()

            excel_data = convert_df(expon_table)
            
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name='exponential_distribution_table.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.error("No data to download. Please check the input parameters.")


footer = """<style>
a:link, a:visited {
    color: black;
    background-color: transparent;
    text-decoration: none; /* Remove underline */
}

a:hover, a:active {
    color: red;
    background-color: transparent;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}

.footer p {
    font-size: 12px; /* Adjust the font size as needed */
    
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='text-align: center;' href="https://www.linkedin.com/in/ahpratama/" target="_blank">AH Pratama </a>|
<a style='text-align: center;' href="https://enibly.com/" target="_blank"> enibly.com</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)