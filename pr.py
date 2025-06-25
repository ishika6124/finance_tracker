import os
import traceback
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st
import warnings
import nltk
import csv
import io
import base64
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional, Any

warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(
    page_title="Finance Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Function to get API keys from either .env file or Streamlit secrets
def get_api_key(key_name):
    # First try to get from Streamlit secrets
    if hasattr(st, 'secrets') and 'api_keys' in st.secrets:
        api_key = st.secrets.api_keys.get(key_name)
        if api_key:
            return api_key
    
    # Fall back to environment variables
    return os.getenv(key_name)

# Set API keys
os.environ["GROQ_API_KEY"] = get_api_key("GROQ_API_KEY")
os.environ["COHERE_API_KEY"] = get_api_key("COHERE_API_KEY")

# Check if required API keys are available
if not os.environ.get("GROQ_API_KEY") or not os.environ.get("COHERE_API_KEY"):
    st.error("⚠️ Required API keys are missing. Please set the GROQ_API_KEY and COHERE_API_KEY in your environment or Streamlit secrets.")
    st.stop()

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Initialize Language Model
@st.cache_resource
def get_llm():
    return ChatGroq(
        model_name="Llama3-70b-8192",
        temperature=0.1,
        max_tokens=2048,
        top_p=0.95
    )

model = get_llm()

# Initialize embeddings with Cohere
@st.cache_resource
def get_embeddings():
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=get_api_key("COHERE_API_KEY"),
        user_agent="finance-tracker-app",
        request_timeout=60
    )

embeddings = get_embeddings()

# Define CSV file path
CSV_FILE_PATH = "finance_data.csv"

# Create CSV file if it doesn't exist
def initialize_csv():
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['transaction_id', 'date', 'amount', 'type', 'category_description', 'description', 'balance_left'])

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Return empty DataFrame with correct columns if file doesn't exist or is empty
        return pd.DataFrame({
            'transaction_id': [],
            'date': [],
            'amount': [],
            'type': [],
            'category_description': [],
            'description': [],
            'balance_left': []
        })

# Function to add a new transaction
def add_transaction(date, amount, type_val, category, description, balance=None):
    df = load_data()
    
    # Generate transaction_id
    if df.empty:
        transaction_id = 1
        # If balance is not provided, assume this is the first entry
        if balance is None:
            if type_val.lower() == 'income':
                balance = amount
            else:
                balance = -amount
    else:
        transaction_id = df['transaction_id'].max() + 1
        # Calculate new balance if not provided
        if balance is None:
            previous_balance = df['balance_left'].iloc[-1]
            if type_val.lower() == 'income':
                balance = previous_balance + amount
            else:
                balance = previous_balance - amount
    
    # Create new transaction
    new_transaction = pd.DataFrame({
        'transaction_id': [transaction_id],
        'date': [pd.to_datetime(date)],
        'amount': [amount],
        'type': [type_val],
        'category_description': [category],
        'description': [description],
        'balance_left': [balance]
    })
    
    # Append to existing data
    updated_df = pd.concat([df, new_transaction], ignore_index=True)
    
    # Save to CSV
    updated_df.to_csv(CSV_FILE_PATH, index=False)
    
    # Clear cache to refresh data
    load_data.clear()
    
    return transaction_id

# Function to process CSV upload
def process_csv_upload(uploaded_file):
    try:
        # Read the uploaded CSV
        uploaded_df = pd.read_csv(uploaded_file)
        
        # Check if the uploaded CSV has the required columns
        required_columns = ['date', 'amount', 'type', 'category_description', 'description']
        missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Load existing data
        existing_df = load_data()
        
        # Start with a new transaction_id
        if existing_df.empty:
            next_id = 1
        else:
            next_id = existing_df['transaction_id'].max() + 1
        
        # Add transaction_id to uploaded data
        uploaded_df['transaction_id'] = range(next_id, next_id + len(uploaded_df))
        
        # Calculate balance_left for each row
        if existing_df.empty:
            starting_balance = 0
        else:
            starting_balance = existing_df['balance_left'].iloc[-1]
        
        balances = []
        current_balance = starting_balance
        
        for _, row in uploaded_df.iterrows():
            if row['type'].lower() == 'income':
                current_balance += row['amount']
            else:
                current_balance -= row['amount']
            balances.append(current_balance)
        
        uploaded_df['balance_left'] = balances
        
        # Combine with existing data
        combined_df = pd.concat([existing_df, uploaded_df], ignore_index=True)
        
        # Save to CSV
        combined_df.to_csv(CSV_FILE_PATH, index=False)
        
        # Clear cache to refresh data
        load_data.clear()
        
        return True, f"Successfully added {len(uploaded_df)} transactions"
    
    except Exception as e:
        return False, f"Error processing CSV: {str(e)}"

# Create vector store with better chunking
@st.cache_resource
def create_vector_store(_df):
    if _df.empty:
        # Return an empty vector store
        return FAISS.from_texts(
            ["No transactions available"],
            embeddings,
            metadatas=[{"transaction_id": 0}]
        )
    
    # Create a combined text field for embedding with more context
    texts = []
    metadatas = []
    
    for _, row in _df.iterrows():
        text = f"""Transaction Details:
        - ID: {row['transaction_id']}
        - Amount: ${row['amount']:.2f}
        - Type: {row['type']}
        - Category: {row['category_description']}
        - Date: {row['date']}
        - Description: {row['description']}
        - Balance After: ${row['balance_left']:.2f}"""
        texts.append(text)
        metadatas.append({
            "transaction_id": row['transaction_id'],
            "amount": row['amount'],
            "type": row['type'],
            "category": row['category_description'],
            "date": str(row['date']),
            "description": row['description'],
            "balance": row['balance_left']
        })
    
    # Create vector store with metadata
    return FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas
    )

# Create compression retriever  
@st.cache_resource
@st.cache_resource
def get_retriever(_vectorstore):
    return ContextualCompressionRetriever(
        base_compressor=CohereRerank(
            model="rerank-english-v3.0",  # ✅ updated model
            cohere_api_key=get_api_key("COHERE_API_KEY")
        ),
        base_retriever=_vectorstore.as_retriever(search_kwargs={"k": 5})
    )


# Finance analysis tools
class FinanceTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_spending(self, date: str) -> str:
        """Analyze spending for a specific date"""
        try:
            daily_data = self.df[self.df['date'].dt.date == pd.to_datetime(date).date()]
            if daily_data.empty:
                return f"No transactions found for {date}"
            
            expenses = daily_data[daily_data['type'].str.lower() != 'expense']
            income = daily_data[daily_data['type'].str.lower() == 'income']
            
            total_spent = expenses['amount'].sum() if not expenses.empty else 0
            total_income = income['amount'].sum() if not income.empty else 0
            transaction_count = len(daily_data)
            balance = daily_data['balance_left'].iloc[-1]
            
            result = f"On {date}, you had {transaction_count} transactions:\n"
            if not expenses.empty:
                result += f"- Spent ${total_spent:.2f}\n"
            if not income.empty:
                result += f"- Received ${total_income:.2f}\n"
            result += f"Your balance at the end of the day was ${balance:.2f}"
            
            return result
        except Exception as e:
            return f"Error analyzing spending: {str(e)}"

    def analyze_category_spending(self, category: str, period: str = 'all') -> str:
        """Analyze spending for a specific category"""
        try:
            # Filter by category (case insensitive)
            category_data = self.df[self.df['category_description'].str.lower() == category.lower()]
            
            if category_data.empty:
                return f"No transactions found for category '{category}'"
            
            # Apply period filter if specified
            if period != 'all':
                if period == 'month':
                    current_month = datetime.now().month
                    current_year = datetime.now().year
                    category_data = category_data[
                        (category_data['date'].dt.month == current_month) & 
                        (category_data['date'].dt.year == current_year)
                    ]
                elif period == 'year':
                    current_year = datetime.now().year
                    category_data = category_data[category_data['date'].dt.year == current_year]
            
            if category_data.empty:
                return f"No transactions found for category '{category}' in the specified period"
            
            expenses = category_data[category_data['type'].str.lower() != 'income']
            income = category_data[category_data['type'].str.lower() == 'income']
            
            total_spent = expenses['amount'].sum() if not expenses.empty else 0
            total_income = income['amount'].sum() if not income.empty else 0
            transaction_count = len(category_data)
            
            period_text = "all time" if period == 'all' else f"this {period}"
            result = f"For category '{category}' ({period_text}):\n"
            if not expenses.empty:
                result += f"- Total spent: ${total_spent:.2f} across {len(expenses)} transactions\n"
                result += f"- Average expense: ${expenses['amount'].mean():.2f}\n"
            if not income.empty:
                result += f"- Total income: ${total_income:.2f} across {len(income)} transactions\n"
            
            return result
        except Exception as e:
            return f"Error analyzing category: {str(e)}"

    def analyze_trends(self, start_date: str, end_date: str) -> str:
        """Analyze spending trends between two dates"""
        try:
            mask = (self.df['date'].dt.date >= pd.to_datetime(start_date).date()) & \
                   (self.df['date'].dt.date <= pd.to_datetime(end_date).date())
            period_data = self.df[mask]
            
            if period_data.empty:
                return f"No transactions found between {start_date} and {end_date}"
            
            expenses = period_data[period_data['type'].str.lower() != 'income']
            income = period_data[period_data['type'].str.lower() == 'income']
            
            total_spent = expenses['amount'].sum() if not expenses.empty else 0
            total_income = income['amount'].sum() if not income.empty else 0
            avg_expense = expenses['amount'].mean() if not expenses.empty else 0
            net_savings = total_income - total_spent
            
            # Category breakdown
            if not expenses.empty:
                category_spending = expenses.groupby('category_description')['amount'].sum().sort_values(ascending=False)
                top_categories = category_spending.head(3)
                categories_text = "\n".join([f"- {cat}: ${amt:.2f}" for cat, amt in top_categories.items()])
            else:
                categories_text = "No expense categories found"
            
            result = f"Between {start_date} and {end_date}:\n"
            if not expenses.empty:
                result += f"- Total spent: ${total_spent:.2f} across {len(expenses)} transactions\n"
                result += f"- Average expense: ${avg_expense:.2f}\n"
            if not income.empty:
                result += f"- Total income: ${total_income:.2f} across {len(income)} transactions\n"
            result += f"- Net savings: ${net_savings:.2f}\n\n"
            result += f"Top spending categories:\n{categories_text}"
            
            return result
        except Exception as e:
            return f"Error analyzing trends: {str(e)}"

    def get_balance_info(self) -> str:
        """Get current balance and recent changes"""
        try:
            if self.df.empty:
                return "No transaction data available"
            
            current_balance = self.df['balance_left'].iloc[-1]
            
            # Get balance from a week ago if available
            week_ago = datetime.now() - pd.Timedelta(days=7)
            week_ago_data = self.df[self.df['date'] <= week_ago]
            
            if not week_ago_data.empty:
                last_week_balance = week_ago_data['balance_left'].iloc[-1]
                balance_change = current_balance - last_week_balance
                change_text = f"Balance change in the last week: ${balance_change:.2f}"
            else:
                change_text = "No data available for balance change calculation"
            
            return f"Current balance: ${current_balance:.2f}. {change_text}"
        except Exception as e:
            return f"Error retrieving balance info: {str(e)}"

    def search_transactions(self, retriever, query: str) -> str:
        """Search for specific transactions using RAG"""
        try:
            relevant_docs = retriever.get_relevant_documents(query)
            if not relevant_docs:
                return "No relevant transactions found."
            
            results = []
            for i, doc in enumerate(relevant_docs[:5]):
                results.append(f"{i+1}. Transaction {doc.metadata['transaction_id']}: ${doc.metadata['amount']:.2f} {doc.metadata['type']} " +
                              f"for {doc.metadata['category']} on {doc.metadata['date']} " +
                              f"({doc.metadata['description']})")
            
            return "Found these relevant transactions:\n\n" + "\n\n".join(results)
        except Exception as e:
            return f"Error searching transactions: {str(e)}"
    
    def savings_projection(self, target_amount: float) -> str:
        """Project how long it would take to save a specific amount"""
        try:
            if self.df.empty:
                return "No transaction data available for savings projection"
            
            # Filter for recent data (last 30 days or all if less)
            if len(self.df) >= 30:
                recent_df = self.df.iloc[-30:]
            else:
                recent_df = self.df
            
            # Calculate average daily savings
            expenses = recent_df[recent_df['type'].str.lower() != 'income']
            income = recent_df[recent_df['type'].str.lower() == 'income']
            
            total_expenses = expenses['amount'].sum() if not expenses.empty else 0
            total_income = income['amount'].sum() if not income.empty else 0
            
            days_spanned = (recent_df['date'].max() - recent_df['date'].min()).days + 1
            if days_spanned < 1:
                days_spanned = 1
            
            daily_income = total_income / days_spanned
            daily_expenses = total_expenses / days_spanned
            daily_savings = daily_income - daily_expenses
            
            if daily_savings <= 0:
                return f"Based on your recent spending patterns, you're currently not saving money. " + \
                       f"You're spending ${daily_expenses:.2f} per day while earning ${daily_income:.2f} per day. " + \
                       f"To save ${target_amount:.2f}, you'll need to reduce expenses or increase income."
            
            days_needed = target_amount / daily_savings
            months_needed = days_needed / 30
            
            return f"Based on your recent finances (last {days_spanned} days):\n" + \
                   f"- Daily income: ${daily_income:.2f}\n" + \
                   f"- Daily expenses: ${daily_expenses:.2f}\n" + \
                   f"- Daily savings: ${daily_savings:.2f}\n\n" + \
                   f"To save ${target_amount:.2f}, you'll need approximately:\n" + \
                   f"- {days_needed:.0f} days\n" + \
                   f"- {months_needed:.1f} months"
        except Exception as e:
            return f"Error calculating savings projection: {str(e)}"
    
    def budget_advice(self, monthly_budget: float = None) -> str:
        """Provide budget advice based on spending patterns"""
        try:
            if self.df.empty:
                return "No transaction data available for budget advice"
            
            # Get current month data
            current_month = datetime.now().month
            current_year = datetime.now().year
            month_data = self.df[
                (self.df['date'].dt.month == current_month) & 
                (self.df['date'].dt.year == current_year)
            ]
            
            if month_data.empty:
                return "No transaction data available for the current month"
            
            # Calculate spending by category
            expenses = month_data[month_data['type'].str.lower() != 'income']
            income = month_data[month_data['type'].str.lower() == 'income']
            
            total_expenses = expenses['amount'].sum() if not expenses.empty else 0
            total_income = income['amount'].sum() if not income.empty else 0
            
            # If budget not provided, estimate from income
            if monthly_budget is None:
                if not income.empty:
                    monthly_budget = total_income * 0.9  # 90% of income
                else:
                    monthly_budget = total_expenses * 1.1  # 110% of expenses
            
            # Budget progress
            budget_remaining = monthly_budget - total_expenses
            days_in_month = pd.Timestamp(current_year, current_month + 1, 1) - pd.Timestamp(current_year, current_month, 1)
            days_remaining = days_in_month.days - datetime.now().day + 1
            daily_budget = budget_remaining / max(days_remaining, 1)
            
            # Category breakdown
            if not expenses.empty:
                category_spending = expenses.groupby('category_description')['amount'].sum().sort_values(ascending=False)
                top_categories = category_spending.head(3)
                categories_text = "\n".join([f"- {cat}: ${amt:.2f} ({(amt/total_expenses*100):.1f}%)" for cat, amt in top_categories.items()])
            else:
                categories_text = "No expense categories found"
            
            if budget_remaining < 0:
                status = f"⚠️ You've exceeded your monthly budget by ${abs(budget_remaining):.2f}"
                advice = "Consider reducing spending in your top expense categories to get back on track."
            else:
                status = f"You have ${budget_remaining:.2f} remaining in your budget for this month"
                advice = f"With {days_remaining} days remaining, you can spend about ${daily_budget:.2f} per day to stay on budget."
            
            return f"Budget Summary (Monthly budget: ${monthly_budget:.2f}):\n" + \
                   f"{status}\n\n" + \
                   f"Top spending categories this month:\n{categories_text}\n\n" + \
                   f"Advice: {advice}"
        except Exception as e:
            return f"Error providing budget advice: {str(e)}"

# RAG Chatbot for financial questions
class FinanceChatbot:
    def __init__(self, model, retriever, finance_tools):
        self.model = model
        self.retriever = retriever
        self.finance_tools = finance_tools
        
    def get_response(self, query):
        try:
            # First check if this is a direct question for finance tools
            if "how much" in query.lower() and "save" in query.lower() and "buy" in query.lower():
                # Extract amount from query
                import re
                amount_match = re.search(r"[\$£€]?(\d+(?:\.\d+)?)", query)
                if amount_match:
                    target_amount = float(amount_match.group(1))
                    return self.finance_tools.savings_projection(target_amount) 
            
            elif "spend" in query.lower() and "on" in query.lower() and any(cat in query.lower() for cat in ["food", "grocery", "dining", "shopping", "entertainment", "transport", "utilities"]):
                # Try to extract category
                categories = ["food", "grocery", "groceries", "dining", "restaurant", "shopping", "entertainment", "transport", "transportation", "utilities", "bills"]
                for cat in categories:
                    if cat in query.lower():
                        return self.finance_tools.analyze_category_spending(cat, "month")
            
            elif "budget" in query.lower() or "target" in query.lower():
                return self.finance_tools.budget_advice()
            
            # Get relevant context based on user query
            relevant_docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful financial assistant for a personal finance tracking application.
                Answer the user's question based on their financial transaction data.
                If you don't know the answer based on the data provided, say so.
                Focus only on providing factual financial analysis, budgeting advice, or answering transaction-related questions.
                Be concise and helpful. Format your response in a clean, readable way.
                
                Context from transaction data:
                {context}"""),
                ("human", "{query}")
            ])
            
            # Create chain and get response
            chain = prompt | self.model | StrOutputParser()
            response = chain.invoke({
                "context": context if context else "No relevant transaction data available.",
                "query": query
            })
            
            return response
        
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your question."

# Initialize CSV file if it doesn't exist
initialize_csv()

# Main Streamlit UI
def main():
    st.title("📊 Personal Finance Tracker")
    
    # Set up tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Add Transaction", "Upload Data", "Finance Assistant"])
    
    # Load data
    df = load_data()
    
    # Create vector store and retriever if data exists
    if not df.empty:
        vectorstore = create_vector_store(df)
        compression_retriever = get_retriever(vectorstore)
    else:
        vectorstore = None
        compression_retriever = None
    
    # Initialize finance tools
    finance_tools = FinanceTools(df)
    
    # TAB 1: DASHBOARD
    with tab1:
        st.header("Financial Dashboard")
        
        if df.empty:
            st.info("No financial data available. Please add transactions or upload data.")
        else:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_balance = df['balance_left'].iloc[-1]
                st.metric("Current Balance", f"${current_balance:.2f}")
            
            with col2:
                current_month = datetime.now().month
                current_year = datetime.now().year
                month_data = df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]
                
                if not month_data.empty:
                    monthly_expenses = month_data[month_data['type'].str.lower() != 'income']['amount'].sum()
                    st.metric("Monthly Expenses", f"${monthly_expenses:.2f}")
                else:
                    st.metric("Monthly Expenses", "$0.00")
            
            with col3:
                if not month_data.empty:
                    monthly_income = month_data[month_data['type'].str.lower() == 'income']['amount'].sum()
                    st.metric("Monthly Income", f"${monthly_income:.2f}")
                else:
                    st.metric("Monthly Income", "$0.00")
            
            # Plots
            st.subheader("Spending by Category")
            expenses = df[df['type'].str.lower() != 'income']
            if not expenses.empty:
                category_spending = expenses.groupby('category_description')['amount'].sum().reset_index()
                fig = px.pie(category_spending, values='amount', names='category_description', title="Expense Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No expense data to display")
            
            st.subheader("Balance Over Time")
            if len(df) > 1:
                fig = px.line(df, x='date', y='balance_left', title="Account Balance History")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data points to show balance trend")
            
            st.subheader("Recent Transactions")
            st.dataframe(df.tail(10)[['date', 'amount', 'type', 'category_description', 'description']], use_container_width=True)
    
    # TAB 2: ADD TRANSACTION
    with tab2:
        st.header("Add New Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.now())
            amount = st.number_input("Amount", min_value=0.01, step=0.01)
            
        with col2:
            type_val = st.selectbox("Type", ["Expense", "Income"])
            category = st.selectbox("Category", [
                "Food & Dining", "Groceries", "Transportation", "Utilities", 
                "Entertainment", "Shopping", "Housing", "Health", "Travel", 
                "Education", "Salary", "Investment", "Other"
            ])
        
        description = st.text_input("Description")
        
        if st.button("Add Transaction"):
            transaction_id = add_transaction(
                date=date,
                amount=amount,
                type_val=type_val,
                category=category,
                description=description
            )
            
            st.success(f"Transaction added successfully! (ID: {transaction_id})")
            
            # Clear cache to refresh data
            load_data.clear()
    
    # TAB 3: UPLOAD DATA
    with tab3:
        st.header("Upload Transaction Data")
        
        st.write("Upload a CSV file with your financial data.")
        st.write("Required columns: date, amount, type, category_description, description")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            st.write("Preview of uploaded data:")
            preview_df = pd.read_csv(uploaded_file)
            st.dataframe(preview_df.head(), use_container_width=True)
            
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            
            if st.button("Process Upload"):
                success, message = process_csv_upload(uploaded_file)
                if success:
                    st.success(message)
                    
                    # Clear cache to refresh data
                    load_data.clear()
                else:
                    st.error(message)
    
    # TAB 4: FINANCE ASSISTANT
    with tab4:
        st.header("Finance Assistant")
        
        if df.empty:
            st.info("No financial data available. Please add transactions or upload data to use the assistant.")
        else:
            # Initialize chatbot
            chatbot = FinanceChatbot(model, compression_retriever, finance_tools)
            
            # Initialize chat history in session state if it doesn't exist
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me about your finances..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get bot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):

                        response = chatbot.get_response(prompt)
                        st.markdown(response)
                
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# Instructions for setting up and sample queries
def display_instructions():
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    ## Getting Started
    1. Add transactions manually via the 'Add Transaction' tab
    2. Or upload a CSV file with your financial data via the 'Upload Data' tab
    3. View your financial summary on the Dashboard
    4. Ask questions to the Finance Assistant
    
    ## Sample Questions
    Try asking the assistant questions like:
    - How much did I spend on groceries this month?
    - What's my current balance?
    - How much should I save to buy a $500 item?
    - What's my spending trend for the last month?
    - How can I meet my budget target this month?
    """)

# Export data function
def export_data():
    st.sidebar.header("Export Data")
    
    df = load_data()
    if not df.empty:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="finance_data.csv">Download CSV File</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    else:
        st.sidebar.info("No data to export")

# Sidebar components
def sidebar():
    st.sidebar.title("Finance Tracker")
    st.sidebar.image("https://img.icons8.com/color/96/000000/financial-growth.png", width=100)
    
    display_instructions()
    export_data()
    
    st.sidebar.header("Settings")
    if st.sidebar.button("Clear Sample Data"):
        # Create an empty DataFrame with the correct columns
        empty_df = pd.DataFrame({
            'transaction_id': [],
            'date': [],
            'amount': [],
            'type': [],
            'category_description': [],
            'description': [],
            'balance_left': []
        })
        
        # Save empty DataFrame to CSV
        empty_df.to_csv(CSV_FILE_PATH, index=False)
        
        # Clear caches
        load_data.clear()
        st.sidebar.success("Sample data cleared!")
    
    st.sidebar.header("About")
    st.sidebar.info("""
    This finance tracker application helps you manage your personal finances using 
    Retrieval-Augmented Generation (RAG) with the Groq API to provide intelligent
    financial insights and answer your questions.
    """)

# Run the application
if __name__ == "__main__":
    sidebar()
    main()