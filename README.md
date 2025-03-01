# LLM Cost Calculator 💰

An interactive web application to compare costs across different LLM providers (OpenAI, Anthropic, Google) with various optimization settings. Built with Streamlit.

## Features

- Compare costs across multiple LLM providers:
  - OpenAI models (GPT-4o, GPT-4o-mini, o1, o3-mini, o1-mini)
  - Google Gemini
  - Anthropic Claude (3.5 and 3.7 Sonnet)
- Interactive parameters adjustment:
  - Number of reports
  - System tokens
  - User tokens
  - Output tokens
- Cost optimization options:
  - Batch processing discount
  - Cached input discount
- Detailed cost breakdown for each model
- Visual cost comparisons

## Live Demo

Visit the live application at: [Streamlit Cloud](https://llm-cost-calculator.streamlit.app)

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-cost-calculator.git
cd llm-cost-calculator
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Deployment

This application is ready to be deployed on Streamlit Cloud:

1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

## License

MIT License - feel free to use this project for your own purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 