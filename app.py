import streamlit as st
import pandas as pd
from cost_calculator import (
    generate_cost_comparison,
    print_detailed_calculation
)

st.set_page_config(
    page_title="LLM Cost Calculator",
    page_icon="💰",
    layout="wide"
)

st.title("LLM Cost Calculator 💰")
st.write("Compare costs across different LLM providers with various optimization settings")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

num_reports = st.sidebar.number_input(
    "Number of Reports",
    min_value=1,
    max_value=10_000_000,
    value=2_000_000,
    step=100_000,
    help="Number of reports to process"
)

system_tokens = st.sidebar.number_input(
    "System Tokens",
    min_value=1,
    max_value=10_000,
    value=1_000,
    step=100,
    help="Number of tokens in system prompt"
)

user_tokens = st.sidebar.number_input(
    "User Tokens",
    min_value=1,
    max_value=10_000,
    value=500,
    step=100,
    help="Number of tokens in user prompt"
)

output_tokens = st.sidebar.number_input(
    "Output Tokens",
    min_value=1,
    max_value=10_000,
    value=1_000,
    step=100,
    help="Average number of tokens in output"
)

# Calculate costs
costs = generate_cost_comparison(num_reports, system_tokens, user_tokens, output_tokens)

# Convert to DataFrame for better display
df_data = []
for model, cost in costs.items():
    # Extract base model name and optimization settings
    base_model = model.split(' (')[0]
    has_batch = '(batch' in model.lower()
    has_cached = 'cached' in model.lower()
    
    df_data.append({
        'Model': base_model,
        'Batch Processing': '✓' if has_batch else '✗',
        'Cached Input': '✓' if has_cached else '✗',
        'Total Cost ($)': cost,
        'Cost Per Report ($)': cost / num_reports
    })

df = pd.DataFrame(df_data)

# Display summary metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Reports", f"{num_reports:,}")
with col2:
    st.metric("Tokens per Report", f"{(system_tokens + user_tokens + output_tokens):,}")
with col3:
    st.metric("Total Tokens", f"{num_reports * (system_tokens + user_tokens + output_tokens):,}")

# Display the results
st.header("Cost Comparison")
st.dataframe(
    df.style.format({
        'Total Cost ($)': '${:,.2f}',
        'Cost Per Report ($)': '${:,.6f}'
    }),
    hide_index=True,
    use_container_width=True
)

# Bar chart
st.header("Cost Visualization")
chart_data = df.pivot_table(
    index='Model',
    values='Total Cost ($)',
    aggfunc='min'  # Show the lowest cost for each model
)
st.bar_chart(chart_data)

# Detailed breakdown for selected model
st.header("Detailed Cost Breakdown")

# Get unique base model names
base_models = sorted(list(set([model.split(' (')[0] for model in costs.keys() 
                             if model.startswith('OpenAI') or model.startswith('Claude')])))

selected_model = st.selectbox(
    "Select a model for detailed cost breakdown",
    options=base_models
)

# Create columns for optimization options
col1, col2 = st.columns(2)
with col1:
    cached_input = st.checkbox("Use Cached Input", value=False)
with col2:
    batch_discount = st.checkbox("Apply Batch Discount", value=False)

# Display detailed calculation
if selected_model:
    st.text("Detailed Calculation:")
    import io
    from contextlib import redirect_stdout

    # Capture the printed output
    f = io.StringIO()
    with redirect_stdout(f):
        print_detailed_calculation(
            selected_model,
            num_reports,
            system_tokens,
            user_tokens,
            output_tokens,
            cached_input=cached_input,
            batch_discount=batch_discount
        )
    
    # Display the captured output in a code block
    st.code(f.getvalue()) 