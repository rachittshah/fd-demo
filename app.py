import time
import streamlit as st
import os
from vanna_openai import CustomVanna
from vanna_calls import (
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    generate_summary_cached
)

# Must be the first Streamlit command
st.set_page_config(layout="wide")

# Initialize conversation history in session state if it doesn't exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize Vanna with OpenAI
@st.cache_resource
def init_vanna():
    config = {
        'model': 'gpt-4o',  # Keep the existing model name as requested
        'allow_llm_to_see_data': True
    }
    vn = CustomVanna(config=config)
    return vn

vn = init_vanna()

# Get database path from secrets or use default path
def get_db_path():
    if hasattr(st, "secrets") and "DB_PATH" in st.secrets:
        return st.secrets.DB_PATH
    return os.path.abspath("./dental.sqlite")

# Connect to database (this will create a new connection in the current thread)
db_path = get_db_path()
if not os.path.exists(db_path):
    st.error(f"""
    Database file not found at {db_path}. 
    
    If you're running locally:
    1. Please run create_db.py first to generate the database.
    
    If you're deploying to Streamlit Cloud:
    1. Make sure to upload the dental.sqlite file
    2. Set the correct DB_PATH in your secrets
    """)
    st.stop()

if not vn.connect_to_sqlite(db_path):
    st.error("Failed to connect to the database. Please check your database configuration.")
    st.stop()

# Train on schema if not already trained
if 'schema_trained' not in st.session_state:
    with st.spinner('Training on database schema...'):
        if vn.train_on_schema():
            st.session_state['schema_trained'] = True
        else:
            st.error("Failed to train on database schema")
            st.stop()

st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
st.sidebar.checkbox("Show Insights", value=True, key="show_insights")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")

def clear_conversation():
    st.session_state.conversation_history = []
    st.session_state.my_question = None

st.sidebar.button("Clear Conversation", on_click=clear_conversation, use_container_width=True)

st.title("Frontier Dental AI")

def set_question(question):
    st.session_state["my_question"] = question

@st.cache_data(ttl=3600)
def generate_questions():
    return [
        "What are the top 5 selling products by total order value?",
        "Show me the monthly revenue trend",
        "Which dental practices have the highest order frequency?",
        "What is the average order value by product category?",
        "Show me overdue invoices and their total value",
        "Which products are running low on stock?",
        "What is the distribution of payment methods used?",
        "Show me the sales performance by manufacturer"
    ]

# Display conversation history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        if message.get("type") == "code":
            st.code(message["content"], language="sql", line_numbers=True)
        elif message.get("type") == "dataframe":
            if isinstance(message["content"], str):
                st.text(message["content"])
            else:
                st.dataframe(message["content"])
        elif message.get("type") == "plot":
            st.plotly_chart(message["content"])
        else:
            st.write(message["content"])

# Show suggested questions only if no conversation history
if not st.session_state.conversation_history:
    assistant_message_suggested = st.chat_message("assistant")
    if assistant_message_suggested.button("Click to show suggested questions"):
        questions = generate_questions()
        for question in questions:
            st.button(
                question,
                on_click=set_question,
                args=(question,),
            )

# Always show the chat input
my_question = st.chat_input("Ask me a question about your data")

if my_question:
    # Add user question to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": my_question})
    
    # Display user message
    with st.chat_message("user"):
        st.write(my_question)

    # Generate SQL using Vanna with context
    sql = vn.generate_sql(question=my_question, conversation_history=st.session_state.conversation_history)

    if sql:
        if st.session_state.get("show_sql", True):
            with st.chat_message("assistant"):
                st.code(sql, language="sql", line_numbers=True)
                st.session_state.conversation_history.append({"role": "assistant", "type": "code", "content": sql})

        df = vn.run_sql(sql=sql)

        if df is not None:
            st.session_state["df"] = df

            if st.session_state.get("show_table", True):
                with st.chat_message("assistant"):
                    if len(df) > 10:
                        st.text("First 10 rows of data")
                        st.dataframe(df.head(10))
                        st.session_state.conversation_history.append(
                            {"role": "assistant", "type": "dataframe", "content": "First 10 rows of data"}
                        )
                    else:
                        st.dataframe(df)
                    st.session_state.conversation_history.append(
                        {"role": "assistant", "type": "dataframe", "content": df}
                    )

            # Add insights generation with context
            if st.session_state.get("show_insights", True):
                try:
                    with st.chat_message("assistant"):
                        with st.spinner('Analyzing data and generating insights...'):
                            insights = vn.generate_insights(
                                question=my_question,
                                sql=sql,
                                df=df,
                                conversation_history=st.session_state.conversation_history
                            )
                            if insights:
                                st.markdown("### Data Insights")
                                st.markdown(insights)
                                st.session_state.conversation_history.append(
                                    {"role": "assistant", "content": f"### Data Insights\n{insights}"}
                                )
                            else:
                                st.warning("Could not generate insights for this query")
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")

            # Generate and display chart
            if should_generate_chart_cached(question=my_question, sql=sql, df=df):
                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        with st.chat_message("assistant"):
                            fig = generate_plot_cached(code=code, df=df)
                            if fig is not None:
                                st.plotly_chart(fig)
                                st.session_state.conversation_history.append(
                                    {"role": "assistant", "type": "plot", "content": fig}
                                )
                            else:
                                st.error("I couldn't generate a chart")

            # Add follow-up suggestions
            if st.session_state.get("show_followup", True):
                with st.chat_message("assistant"):
                    followup_questions = generate_followup_cached(
                        question=my_question,
                        sql=sql,
                        df=df
                    )
                    if followup_questions:
                        st.write("You might want to ask:")
                        for question in followup_questions[:3]:
                            if question:  # Check if question is not None or empty
                                st.button(
                                    question,
                                    on_click=set_question,
                                    args=(question,),
                                    key=f"followup_{hash(question)}"  # Add unique key for each button
                                )
                        st.session_state.conversation_history.append(
                            {"role": "assistant", "content": "Here are some follow-up questions you might want to ask."}
                        )

    else:
        with st.chat_message("assistant"):
            st.error("I wasn't able to generate SQL for that question")
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": "I wasn't able to generate SQL for that question"}
            )
