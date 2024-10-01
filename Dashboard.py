import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, gamma, expon, chi2, lognorm, weibull_min, beta as beta_dist

LOGO = "images/logo.png"
st.logo(
    LOGO,
    link="https://enibly.com",
)


def plot_base_distribution(dist_type, params):
    try:
        if dist_type == 'Normal':
            mu, std = params
            dist = norm(loc=mu, scale=std)
            x_values = np.linspace(mu - 4*std, mu + 4*std, 1000)
        elif dist_type == 'Uniform':
            a, b = params
            dist = uniform(loc=a, scale=b-a)
            x_values = np.linspace(a - 1, b + 1, 1000)
        elif dist_type == 'Gamma':
            shape, scale = params
            dist = gamma(a=shape, scale=scale)
            x_values = np.linspace(0, shape * scale * 3, 1000)
        elif dist_type == 'Exponential':
            scale = params[0]
            dist = expon(scale=scale)
            x_values = np.linspace(0, scale * 5, 1000)
        elif dist_type == 'Chi-Square':
            df = params[0]
            dist = chi2(df=df)
            x_values = np.linspace(0, df * 3, 1000)
        elif dist_type == 'Lognormal':
            mu, sigma = params
            dist = lognorm(s=sigma, scale=np.exp(mu))
            x_values = np.linspace(0, np.exp(mu + 4*sigma), 1000)
        elif dist_type == 'Weibull':
            alpha, beta = params
            scale = (1 / alpha) ** (1 / beta)
            dist = weibull_min(c=beta, scale=scale)
            x_values = np.linspace(0, 3 / alpha, 1000)
        elif dist_type == 'Beta':
            alpha, beta = params
            dist = beta_dist(a=alpha, b=beta)
            x_values = np.linspace(0, 1, 1000)
        else:
            raise ValueError("Unsupported distribution type")

        y_values = dist.pdf(x_values)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_values, y_values, label=f'{dist_type} Distribution')
        ax.set_title(f'{dist_type} Distribution')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.legend()

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting distribution: {e}")

def plot_calculation_distribution(dist_type, params, x_range=None, x_thresh=None):
    try:
        if dist_type == 'Normal':
            mu, std = params
            dist = norm(loc=mu, scale=std)
            x_values = np.linspace(mu - 4*std, mu + 4*std, 1000)
        elif dist_type == 'Uniform':
            a, b = params
            dist = uniform(loc=a, scale=b-a)
            x_values = np.linspace(a - 1, b + 1, 1000)
        elif dist_type == 'Gamma':
            shape, scale = params
            dist = gamma(a=shape, scale=scale)
            x_values = np.linspace(0, shape * scale * 3, 1000)
        elif dist_type == 'Exponential':
            scale = params[0]
            dist = expon(scale=scale)
            x_values = np.linspace(0, scale * 5, 1000)
        elif dist_type == 'Chi-Square':
            df = params[0]
            dist = chi2(df=df)
            x_values = np.linspace(0, df * 3, 1000)
        elif dist_type == 'Lognormal':
            mu, sigma = params
            dist = lognorm(s=sigma, scale=np.exp(mu))
            x_values = np.linspace(0, np.exp(mu + 4*sigma), 1000)
        elif dist_type == 'Weibull':
            alpha, beta = params
            scale = (1 / alpha) ** (1 / beta)
            dist = weibull_min(c=beta, scale=scale)
            x_values = np.linspace(0, 3 / alpha, 1000)
        elif dist_type == 'Beta':
            alpha, beta = params
            dist = beta_dist(a=alpha, b=beta)
            x_values = np.linspace(0, 1, 1000)
        else:
            raise ValueError("Unsupported distribution type")

        y_values = dist.pdf(x_values)

        if x_range or x_thresh is not None:
            if x_range:
                x_min, x_max = x_range
            elif x_thresh is not None:
                x_min, x_max = (0 if dist_type in ['Gamma', 'Exponential'] else -np.inf), x_thresh

            probability = dist.cdf(x_max) - dist.cdf(x_min)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x_values, y_values, label=f'{dist_type} Distribution')
            ax.fill_between(x_values, y_values, where=((x_values >= x_min) & (x_values <= x_max)), color='r', alpha=0.5)
            ax.axvline(x_min, color='r', linestyle='--', label=f'x_min = {x_min}')
            ax.axvline(x_max, color='r', linestyle='--', label=f'x_max = {x_max}')
            ax.set_title(f'{dist_type} Distribution with Highlighted Range')
            ax.set_xlabel('x')
            ax.set_ylabel('Probability Density')
            ax.legend()

            st.pyplot(fig)

            return probability

        return None

    except Exception as e:
        st.error(f"Error plotting calculation distribution: {e}")
        return None

st.set_page_config(layout="wide")
st.title('Distribution Probability Calculator and Visualizer')
st.sidebar.title('Distribution Settings')

dist_type = st.sidebar.selectbox('Select Distribution Type', ['Normal', 'Uniform', 'Gamma', 'Exponential', 'Chi-Square', 'Lognormal', 'Weibull', 'Beta'])

params = None
if dist_type == 'Normal':
    mu = st.sidebar.number_input('Mean (mu)', min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    std = st.sidebar.number_input('Standard Deviation (std)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    params = (mu, std)
elif dist_type == 'Uniform':
    a = st.sidebar.number_input('Lower Bound (a)', min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
    b = st.sidebar.number_input('Upper Bound (b)', min_value=-50.0, max_value=50.0, value=1.0, step=0.1)
    params = (a, b)
elif dist_type == 'Gamma':
    shape = st.sidebar.number_input('Shape Parameter', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    scale = st.sidebar.number_input('Scale Parameter', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    params = (shape, scale)
elif dist_type == 'Exponential':
    scale = st.sidebar.number_input('Scale Parameter', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    params = (scale,)
elif dist_type == 'Chi-Square':
    df = st.sidebar.number_input('Degrees of Freedom (df)', min_value=0.1, max_value=30.0, value=1.0, step=0.1)
    alpha = st.sidebar.number_input('Significance Level (alpha)', min_value=0.01, max_value=0.1, value=0.05, step=0.01)
    params = (df,)
elif dist_type == 'Lognormal':
    mu = st.sidebar.number_input('Mean (mu)', min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    sigma = st.sidebar.number_input('Standard Deviation (sigma)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    params = (mu, sigma)
elif dist_type == 'Weibull':
    alpha = st.sidebar.number_input('Shape Parameter (alpha)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    beta = st.sidebar.number_input('Scale Parameter (beta)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    params = (alpha, beta)
elif dist_type == 'Beta':
    alpha = st.sidebar.number_input('Alpha (alpha)', min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    beta = st.sidebar.number_input('Beta (beta)', min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    params = (alpha, beta)

col1, col2 = st.columns([0.4, 0.6], gap="medium")
with col1:
    plot_base_distribution(dist_type, params)

with col2:
    range_type = st.radio('Range Type', ['Range', 'Threshold'])
    if range_type == 'Range':
        x_min = st.number_input('x_min', value=0.0, step=0.1)
        x_max = st.number_input('x_max', value=1.0, step=0.1)
        x_range = (x_min, x_max)
        x_thresh = None
    else:
        x_thresh = st.number_input('x_thresh', value=1.0, step=0.1)
        x_range = None

    if st.button('Calculate Probability'):
        probability = plot_calculation_distribution(dist_type, params, x_range=x_range, x_thresh=x_thresh)
        if probability is not None:
            st.success(f'The probability is: {probability:.4f}', icon="✅")
        else:
            st.error('Please provide a valid range or threshold.')

    if dist_type == 'Chi-Square' and st.button('Calculate Critical Value'):
        critical_value = chi2.ppf(1 - alpha, df)
        st.success(f'The critical value X² for alpha={alpha} and df={df} is: {critical_value:.4f}', icon="✅")

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