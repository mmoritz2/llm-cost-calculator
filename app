def calculate_openai_cost(model, num_reports, system_tokens, user_tokens, output_tokens, batch_discount=False, cached_input=False):
    """
    Calculate OpenAI API costs with optional batch processing and cached input discounts.
    
    Parameters:
    - model: Model name from OpenAI pricing table
    - num_reports: Number of reports to process
    - system_tokens: Number of tokens in system prompt
    - user_tokens: Number of tokens in user prompt
    - output_tokens: Average number of tokens in output
    - batch_discount: Apply 50% discount for batch processing
    - cached_input: Apply 50% discount on system tokens (cached inputs)
    
    Returns:
    - Total cost in USD
    """
    # Updated pricing from provided image ($ per million tokens)
    pricing = {
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'o1': {'input': 15.00, 'output': 60.00},
        'o3-mini': {'input': 1.10, 'output': 4.40},
        'o1-mini': {'input': 1.10, 'output': 4.40}
    }
    
    if model not in pricing:
        raise ValueError(f"Unknown model: {model}")
    
    # Calculate input tokens (system + user)
    total_system_tokens = num_reports * system_tokens
    total_user_tokens = num_reports * user_tokens
    total_output_tokens = num_reports * output_tokens
    
    # Calculate costs for input tokens
    if cached_input:
        # Only system prompt can be cached, user prompt will be different each time
        system_cost = (total_system_tokens / 1_000_000) * pricing[model]['input'] * 0.5  # 50% discount
        user_cost = (total_user_tokens / 1_000_000) * pricing[model]['input']
        input_cost = system_cost + user_cost
    else:
        input_cost = ((total_system_tokens + total_user_tokens) / 1_000_000) * pricing[model]['input']
    
    # Calculate output cost
    output_cost = (total_output_tokens / 1_000_000) * pricing[model]['output']
    
    # Calculate total cost
    total_cost = input_cost + output_cost
    
    # Apply batch discount if enabled
    if batch_discount:
        total_cost *= 0.5
    
    return total_cost

def calculate_google_cost(model, num_reports, system_tokens, user_tokens, output_tokens):
    """
    Calculate Google Gemini API costs.
    
    Parameters:
    - model: 'gemini-2.0-flash'
    - num_reports: Number of reports to process
    - system_tokens: Number of tokens in system prompt
    - user_tokens: Number of tokens in user prompt
    - output_tokens: Average number of tokens in output
    
    Returns:
    - Total cost in USD
    """
    pricing = {
        'gemini-2.0-flash': {'input': 0.35, 'output': 1.05}  # $ per million tokens
    }
    
    if model not in pricing:
        raise ValueError(f"Unknown model: {model}")
    
    # Calculate total tokens
    total_input_tokens = num_reports * (system_tokens + user_tokens)
    total_output_tokens = num_reports * output_tokens
    
    # Calculate cost
    input_cost = (total_input_tokens / 1_000_000) * pricing[model]['input']
    output_cost = (total_output_tokens / 1_000_000) * pricing[model]['output']
    total_cost = input_cost + output_cost
    
    return total_cost

def calculate_anthropic_cost(model, num_reports, system_tokens, user_tokens, output_tokens):
    """
    Calculate Anthropic Claude API costs.
    
    Parameters:
    - model: 'claude-3-5-sonnet', 'claude-3-7-sonnet'
    - num_reports: Number of reports to process
    - system_tokens: Number of tokens in system prompt
    - user_tokens: Number of tokens in user prompt
    - output_tokens: Average number of tokens in output
    
    Returns:
    - Total cost in USD
    """
    pricing = {
        'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},  # $ per million tokens
        'claude-3-7-sonnet': {'input': 15.00, 'output': 75.00}  # $ per million tokens
    }
    
    if model not in pricing:
        raise ValueError(f"Unknown model: {model}")
    
    # Calculate total tokens
    total_input_tokens = num_reports * (system_tokens + user_tokens)
    total_output_tokens = num_reports * output_tokens
    
    # Calculate cost
    input_cost = (total_input_tokens / 1_000_000) * pricing[model]['input']
    output_cost = (total_output_tokens / 1_000_000) * pricing[model]['output']
    total_cost = input_cost + output_cost
    
    return total_cost

def generate_cost_comparison(num_reports=2_000_000, system_tokens=1_000, user_tokens=500, output_tokens=1_000):
    """
    Generate a cost comparison for processing reports with different LLM models.
    """
    openai_models = {
        'OpenAI GPT-4o': 'gpt-4o',
        'OpenAI GPT-4o-mini': 'gpt-4o-mini',
        'OpenAI o1': 'o1',
        'OpenAI o3-mini': 'o3-mini',
        'OpenAI o1-mini': 'o1-mini'
    }
    
    # Create results dictionary
    results = {}
    
    # Calculate costs for each OpenAI model with all discount combinations
    for display_name, model_name in openai_models.items():
        # Standard pricing
        results[f"{display_name}"] = calculate_openai_cost(
            model_name, num_reports, system_tokens, user_tokens, output_tokens
        )
        
        # Batch discount only
        results[f"{display_name} (batch)"] = calculate_openai_cost(
            model_name, num_reports, system_tokens, user_tokens, output_tokens, 
            batch_discount=True
        )
        
        # Cached input only
        results[f"{display_name} (cached)"] = calculate_openai_cost(
            model_name, num_reports, system_tokens, user_tokens, output_tokens, 
            cached_input=True
        )
        
        # Both batch and cached
        results[f"{display_name} (batch+cached)"] = calculate_openai_cost(
            model_name, num_reports, system_tokens, user_tokens, output_tokens, 
            batch_discount=True, cached_input=True
        )
    
    # Add Google model
    results["Google Gemini 2.0 Flash"] = calculate_google_cost(
        'gemini-2.0-flash', num_reports, system_tokens, user_tokens, output_tokens
    )
    
    # Add Claude models
    results["Claude 3.5 Sonnet"] = calculate_anthropic_cost(
        'claude-3-5-sonnet', num_reports, system_tokens, user_tokens, output_tokens
    )
    
    results["Claude 3.7 Sonnet"] = calculate_anthropic_cost(
        'claude-3-7-sonnet', num_reports, system_tokens, user_tokens, output_tokens
    )
    
    # Sort results by cost
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
    
    return sorted_results

def print_detailed_calculation(model_name, num_reports, system_tokens, user_tokens, output_tokens, cached_input=False, batch_discount=False):
    """
    Print a detailed breakdown of the cost calculation for a specific model.
    """
    if model_name.startswith('OpenAI'):
        openai_models = {
            'OpenAI GPT-4o': 'gpt-4o',
            'OpenAI GPT-4o-mini': 'gpt-4o-mini',
            'OpenAI o1': 'o1',
            'OpenAI o3-mini': 'o3-mini',
            'OpenAI o1-mini': 'o1-mini'
        }
        internal_name = openai_models[model_name]
        pricing = {
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'o1': {'input': 15.00, 'output': 60.00},
            'o3-mini': {'input': 1.10, 'output': 4.40},
            'o1-mini': {'input': 1.10, 'output': 4.40}
        }[internal_name]
        
        total_system_tokens = num_reports * system_tokens
        total_user_tokens = num_reports * user_tokens
        total_output_tokens = num_reports * output_tokens
        
        # Calculate input costs
        if cached_input:
            system_cost = (total_system_tokens / 1_000_000) * pricing['input'] * 0.5
            user_cost = (total_user_tokens / 1_000_000) * pricing['input']
            input_breakdown = f"System tokens: {total_system_tokens:,} × ${pricing['input'] * 0.5 / 1_000_000:.8f}/token = ${system_cost:,.2f}\n"
            input_breakdown += f"User tokens: {total_user_tokens:,} × ${pricing['input'] / 1_000_000:.8f}/token = ${user_cost:,.2f}\n"
            input_cost = system_cost + user_cost
        else:
            input_cost = ((total_system_tokens + total_user_tokens) / 1_000_000) * pricing['input']
            input_breakdown = f"Input tokens: {total_system_tokens + total_user_tokens:,} × ${pricing['input'] / 1_000_000:.8f}/token = ${input_cost:,.2f}\n"
        
        # Calculate output cost
        output_cost = (total_output_tokens / 1_000_000) * pricing['output']
        output_breakdown = f"Output tokens: {total_output_tokens:,} × ${pricing['output'] / 1_000_000:.8f}/token = ${output_cost:,.2f}\n"
        
        # Calculate total before batch discount
        pre_batch_total = input_cost + output_cost
        
        # Apply batch discount if enabled
        if batch_discount:
            final_total = pre_batch_total * 0.5
            batch_breakdown = f"Batch discount: ${pre_batch_total:,.2f} × 0.5 = ${final_total:,.2f}\n"
        else:
            final_total = pre_batch_total
            batch_breakdown = ""
        
        # Print the breakdown
        print(f"Detailed cost calculation for {model_name}:")
        print(f"Prices: ${pricing['input']} per 1M input tokens, ${pricing['output']} per 1M output tokens")
        print("-" * 80)
        print(input_breakdown)
        print(output_breakdown)
        print(f"Subtotal: ${pre_batch_total:,.2f}")
        if batch_discount:
            print(batch_breakdown)
        print(f"Total cost: ${final_total:,.2f}")
        print(f"Cost per report: ${final_total / num_reports:.6f}")
        print("-" * 80)
        print()

# Run the calculator with the specified parameters
num_reports = 2_000_000
system_tokens = 1_000
user_tokens = 500
output_tokens = 1_000

costs = generate_cost_comparison(num_reports, system_tokens, user_tokens, output_tokens)

# Print the results in a formatted table
print(f"Cost Comparison for Processing {num_reports:,} Imaging Reports")
print(f"System Prompt: {system_tokens:,} tokens | User Prompt: {user_tokens:,} tokens | Average Output: {output_tokens:,} tokens\n")
print(f"{'Model':<35} {'Total Cost ($)':<15} {'Cost Per Report ($)':<20}")
print("-" * 75)

for model, cost in costs.items():
    cost_per_report = cost / num_reports
    print(f"{model:<35} ${cost:,.2f}{'':>5} ${cost_per_report:.6f}{'':>10}")

# Calculate total tokens and tokens per report
total_tokens_per_report = system_tokens + user_tokens + output_tokens
total_tokens = num_reports * total_tokens_per_report

# Print additional information
print(f"\nTotal tokens processed: {total_tokens:,}")
print(f"Total tokens per report: {total_tokens_per_report:,}")

# Print detailed calculations for selected models
print("\n\nDETAILED COST BREAKDOWNS:")
print("=" * 80)

# Print detailed calculations for the cheapest and most expensive options
cheapest_model = list(costs.keys())[0]
most_expensive_model = list(costs.keys())[-1]

print_detailed_calculation(
    "OpenAI GPT-4o-mini", 
    num_reports, 
    system_tokens, 
    user_tokens, 
    output_tokens, 
    cached_input=True, 
    batch_discount=True
)

print_detailed_calculation(
    "OpenAI GPT-4o-mini", 
    num_reports, 
    system_tokens, 
    user_tokens, 
    output_tokens
)

print_detailed_calculation(
    "OpenAI o3-mini", 
    num_reports, 
    system_tokens, 
    user_tokens, 
    output_tokens, 
    cached_input=True, 
    batch_discount=True
)

print_detailed_calculation(
    "Claude 3.5 Sonnet", 
    num_reports, 
    system_tokens, 
    user_tokens, 
    output_tokens
)