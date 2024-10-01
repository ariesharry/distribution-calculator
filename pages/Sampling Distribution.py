import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sampling Distribution Calculator", page_icon="📊")

st.sidebar.header("Variance Hypothesis Testing")

# Sidebar options to choose the test
test_type = st.sidebar.selectbox("Select the Hypothesis Test", ("Chi-Square Test", "T-Test", "F-Test"))

if test_type == "Chi-Square Test":
    st.title('Variance Hypothesis Test using Chi-Square Distribution')

    # Input data for Chi-Square Test
    sample = st.sidebar.text_area('Enter the sample data (comma separated)', '950,1020,1010,980,970,960,1000,1015,990,985')
    sample = np.array([float(x) for x in sample.split(',')])

    # Input known values
    sigma_squared_0 = st.sidebar.number_input("Hypothesized Population Variance (σ₀²)", value=400.0)
    alpha = st.sidebar.number_input("Significance Level (α)", value=0.05)

    # Sample statistics
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample, ddof=1)

    # Chi-Square test statistic
    chi_square_stat = (n - 1) * sample_variance / sigma_squared_0
    df = n - 1

    # Critical Chi-Square values
    critical_value_low = stats.chi2.ppf(alpha / 2, df)
    critical_value_high = stats.chi2.ppf(1 - alpha / 2, df)

    # P-value
    p_value = 2 * min(stats.chi2.cdf(chi_square_stat, df), 1 - stats.chi2.cdf(chi_square_stat, df))

    # Display results
    st.write(f"Sample Size (n): {n}")
    st.write(f"Sample Variance (s²): {sample_variance:.2f}")
    st.write(f"Test Statistic (χ²): {chi_square_stat:.2f}")
    st.write(f"Degrees of Freedom (df): {df}")
    st.write(f"Critical Value (Lower): {critical_value_low:.2f}")
    st.write(f"Critical Value (Upper): {critical_value_high:.2f}")
    st.write(f"P-Value: {p_value:.4f}")

    # Hypothesis decision
    if chi_square_stat < critical_value_low or chi_square_stat > critical_value_high:
        st.error("Reject the Null Hypothesis (H₀): The variance is significantly different from the hypothesized variance.")
    else:
        st.success("Accept the Null Hypothesis (H₀): The variance is not significantly different from the hypothesized variance.")

    # Plotting the Chi-Square distribution
    x = np.linspace(0, 30, 500)
    y = stats.chi2.pdf(x, df)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Chi-Square Distribution', color='blue')

    # Fill the critical regions
    ax.fill_between(x, 0, y, where=(x <= critical_value_low), color='red', alpha=0.5, label='Rejection Region (Lower)')
    ax.fill_between(x, 0, y, where=(x >= critical_value_high), color='red', alpha=0.5, label='Rejection Region (Upper)')

    # Mark the test statistic
    ax.axvline(chi_square_stat, color='green', linestyle='--', label=f'Test Statistic (χ² = {chi_square_stat:.2f})')

    ax.set_title("Chi-Square Distribution with Critical Regions")
    ax.set_xlabel("Chi-Square Value")
    ax.set_ylabel("Probability Density")
    ax.legend()

    st.pyplot(fig)

elif test_type == "T-Test":
    st.title('Hypothesis Test using T-Distribution')

    # Option to input sample data or summary statistics
    input_type = st.sidebar.radio("Do you want to input raw data or summary statistics?", ("Raw Data", "Summary Statistics"))

    if input_type == "Raw Data":
        # Input raw sample data
        sample = st.sidebar.text_area('Enter the sample data (comma separated)', '5.2, 6.3, 7.1, 5.8, 6.0, 6.5')
        sample = np.array([float(x) for x in sample.split(',')])

        # Sample statistics
        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)

    else:
        # Input summary statistics if raw data is not known
        sample_mean = st.sidebar.number_input("Sample Mean (x̄)", value=6.0)
        sample_std = st.sidebar.number_input("Sample Standard Deviation (s)", value=1.0)
        n = st.sidebar.number_input("Sample Size (n)", value=10, min_value=1)

    # Input hypothesized population mean and significance level
    mu_0 = st.sidebar.number_input("Hypothesized Population Mean (μ₀)", value=6.0)
    alpha = st.sidebar.number_input("Significance Level (α)", value=0.05)

    # T-test statistic calculation
    t_stat = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
    df = n - 1

    # Critical T values
    critical_value_low = stats.t.ppf(alpha / 2, df)
    critical_value_high = stats.t.ppf(1 - alpha / 2, df)

    # P-value calculation
    p_value = 2 * min(stats.t.cdf(t_stat, df), 1 - stats.t.cdf(t_stat, df))

    # Display results
    st.write(f"Sample Size (n): {n}")
    st.write(f"Sample Mean (x̄): {sample_mean:.2f}")
    st.write(f"Sample Standard Deviation (s): {sample_std:.2f}")
    st.write(f"Test Statistic (t): {t_stat:.2f}")
    st.write(f"Degrees of Freedom (df): {df}")
    st.write(f"Critical Value (Lower): {critical_value_low:.2f}")
    st.write(f"Critical Value (Upper): {critical_value_high:.2f}")
    st.write(f"P-Value: {p_value:.4f}")

    # Hypothesis decision
    if t_stat < critical_value_low or t_stat > critical_value_high:
        st.error("Reject the Null Hypothesis (H₀): The mean is significantly different from the hypothesized mean.")
    else:
        st.success("Accept the Null Hypothesis (H₀): The mean is not significantly different from the hypothesized mean.")

    # Plotting the T-distribution
    x = np.linspace(-5, 5, 500)
    y = stats.t.pdf(x, df)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='T-Distribution', color='blue')

    # Fill the critical regions
    ax.fill_between(x, 0, y, where=(x <= critical_value_low), color='red', alpha=0.5, label='Rejection Region (Lower)')
    ax.fill_between(x, 0, y, where=(x >= critical_value_high), color='red', alpha=0.5, label='Rejection Region (Upper)')

    # Mark the test statistic
    ax.axvline(t_stat, color='green', linestyle='--', label=f'Test Statistic (t = {t_stat:.2f})')

    ax.set_title("T-Distribution with Critical Regions")
    ax.set_xlabel("T Value")
    ax.set_ylabel("Probability Density")
    ax.legend()

    st.pyplot(fig)


elif test_type == "F-Test":
    st.title('Variance Comparison using F-Distribution')

    # Input for two samples or summary statistics
    input_type_f = st.sidebar.radio("Do you want to input raw data or summary statistics?", ("Raw Data", "Summary Statistics"))

    if input_type_f == "Raw Data":
        # Input raw sample data
        sample1 = st.sidebar.text_area('Enter Sample 1 data (comma separated)', '5.2, 6.3, 7.1, 5.8, 6.0, 6.5')
        sample2 = st.sidebar.text_area('Enter Sample 2 data (comma separated)', '4.8, 5.5, 6.2, 5.0, 5.3, 6.1')

        sample1 = np.array([float(x) for x in sample1.split(',')])
        sample2 = np.array([float(x) for x in sample2.split(',')])

        # Sample statistics
        n1 = len(sample1)
        n2 = len(sample2)
        variance1 = np.var(sample1, ddof=1)
        variance2 = np.var(sample2, ddof=1)

    else:
        # Input summary statistics if raw data is not known
        variance1 = st.sidebar.number_input("Sample 1 Variance (s₁²)", value=1.2)
        variance2 = st.sidebar.number_input("Sample 2 Variance (s₂²)", value=1.0)
        n1 = st.sidebar.number_input("Sample 1 Size (n₁)", value=10, min_value=1)
        n2 = st.sidebar.number_input("Sample 2 Size (n₂)", value=10, min_value=1)

    # Input significance level
    alpha = st.sidebar.number_input("Significance Level (α)", value=0.05)

    # F-test statistic calculation
    f_stat = variance1 / variance2
    df1 = n1 - 1
    df2 = n2 - 1

    # Upper critical value
    critical_value_high = stats.f.ppf(1 - alpha, df1, df2)

    # P-value calculation
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    # Display results
    st.write(f"Sample 1 Size (n₁): {n1}")
    st.write(f"Sample 2 Size (n₂): {n2}")
    st.write(f"Sample 1 Variance (s₁²): {variance1:.2f}")
    st.write(f"Sample 2 Variance (s₂²): {variance2:.2f}")
    st.write(f"Test Statistic (F): {f_stat:.2f}")
    st.write(f"Degrees of Freedom 1 (df₁): {df1}")
    st.write(f"Degrees of Freedom 2 (df₂): {df2}")
    st.write(f"Critical Value (Upper): {critical_value_high:.2f}")
    st.write(f"P-Value: {p_value:.4f}")

    # Hypothesis decision
    if f_stat > critical_value_high:
        st.error("Reject the Null Hypothesis (H₀): The variances are significantly different.")
    else:
        st.success("Accept the Null Hypothesis (H₀): The variances are not significantly different.")

    # Plotting the F-distribution
    x = np.linspace(0, 5, 500)
    y = stats.f.pdf(x, df1, df2)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='F-Distribution', color='blue')

    # Fill the critical regions
    ax.fill_between(x, 0, y, where=(x >= critical_value_high), color='red', alpha=0.5, label='Rejection Region (Upper)')

    # Mark the test statistic
    ax.axvline(f_stat, color='green', linestyle='--', label=f'Test Statistic (F = {f_stat:.2f})')

    ax.set_title("F-Distribution with Critical Region (Upper Tail)")
    ax.set_xlabel("F Value")
    ax.set_ylabel("Probability Density")
    ax.legend()

    st.pyplot(fig)



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