
# Finance Tracker - Personal Finance Management Application



## Overview
Finance Tracker is a Streamlit-based web application that helps users manage their personal finances. It provides tools for tracking income and expenses, visualizing spending patterns, and getting AI-powered financial advice through a chatbot interface.

## ✨ Key Features

### 📊 Dashboard  
**Visualize your financial data with interactive charts**  
- View expense distribution by category  
- Track account balance history over time  
- See monthly income vs. expenses  
- Review recent transactions at a glance  

### ➕ Transaction Management  
**Add individual transactions or bulk upload via CSV**  
- Manually log each transaction with:  
  - Date & amount  
  - Type (Income/Expense)  
  - Category & description  
- Import existing records via CSV files  
- Automatic balance calculation  

### 🤖 Finance Assistant  
**AI-powered chatbot for financial questions**  
- Ask natural language questions like:  
  - "How much did I spend on dining last month?"  
  - "What's my current balance?"  
  - "How can I save $1000 faster?"  
- Get instant insights from your transaction history  

### 📈 Trend Analysis  
**Track spending patterns over time**  
- Analyze spending by day/week/month  
- Compare income vs. expenses  
- Identify spending trends by category  
- Custom date range reporting  

### 💰 Budget Tools  
**Personalized budget advice and savings projections**  
- Monthly budget planning assistant  
- Daily spending limit recommendations  
- Savings timeline calculator  
- Overspending alerts  
- Expense optimization tips  

## 🛠️ Technologies Used

### Core Technologies
- **Python** - Primary programming language
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis

### Visualization
- **Plotly** - Interactive data visualization

### AI & NLP Components
- **LangChain** - LLM integration and RAG implementation
- **Groq API** - High-performance LLM inference
- **Cohere API** - Embeddings and document reranking
- **FAISS** - Vector similarity search

### Data Management
- **CSV** - Local data storage format
- **NLTK** - Natural language processing utilities
## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/finance-tracker.git
   cd finance-tracker

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
### 4. Configure environment variables:

- **Create** a `.env` file in the root directory
- **Add** your API keys:


   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
## Usage

1. **Run the application**:

    ```bash
    streamlit run pr.py
    ```

2. The application will open in your default web browser at [`http://localhost:8501`](http://localhost:8501)

3. **Use the tabs to navigate between different features:**

    - **Dashboard**: View financial summaries and visualizations  
    - **Add Transaction**: Manually enter new transactions  
    - **Upload Data**: Bulk upload transactions via CSV  
    - **Finance Assistant**: Chat with the AI financial advisor

## Data Format

When uploading CSV files, ensure they contain the following columns:

- `date` (YYYY-MM-DD format)
- `amount` (numerical value)
- `type` ("Income" or "Expense")
- `category_description` (category name)
- `description` (optional transaction details)

---

## Sample Questions for Finance Assistant

Try asking:

- "How much did I spend on groceries this month?"
- "What's my current balance?"
- "How long will it take me to save $1000?"
- "Give me budget advice for this month."
- "Show me my recent restaurant expenses"
## Configuration

You can customize the application by modifying:

- `CSV_FILE_PATH` in `pr.py` to change where transaction data is stored  
- The Groq model parameters (temperature, max_tokens, etc.)  
- The transaction categories in the "Add Transaction" tab  

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Support

For issues or feature requests, please open an issue in the GitHub repository.
