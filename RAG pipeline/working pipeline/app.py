import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from joblib import load  # New import to load the CatBoost model
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# LangChain and AI Tools
from catboost import CatBoostRegressor
from pydantic import BaseModel, Field
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# from langchain_community.document_loaders.csv_loader import CSVLoader # Not needed for loading FAISS
from langchain_community.vectorstores import FAISS

# from langchain_text_splitters import RecursiveCharacterTextSplitter # Not needed for loading FAISS

# --- Global Configurations ---
app = Flask(__name__)
MODEL_PATH = "catboost_actual.cbm"
LLM_MODEL = "gpt-4o"
FAISS_INDEX_PATH = "faiss_news_index"

# !!! CRITICAL: 11 Features in the EXACT order of training !!!
# Multi-output model predicts both MMR and selling price
EXPECTED_FEATURES = [
    "make",
    "model",
    "body",
    "transmission",
    "state",
    "condition_category",
    "odometer",
    "car_age",
    "sale_year",
    "sale_month",
    "seller_category",
]

# =================================================================
# 1. MODEL LOADING AND PREDICTION FUNCTION (FIXED)
# =================================================================

# Global variable to hold the CatBoost model
catboost_model = None


def load_catboost_model():
    """Loads the pre-trained CatBoost model."""
    global catboost_model
    try:
        catboost_model = CatBoostRegressor().load_model(MODEL_PATH)

        # BIAS CORRECTION FOR MULTI-OUTPUT MODEL:
        # Multi-output models return lists for scale and bias (one per output)
        # Uncomment below if bias correction is needed after testing predictions
        """
        scale, bias = catboost_model.get_scale_and_bias()
        # For multi-output: bias is a list [bias_mmr, bias_selling_price]
        # Apply correction to each output
        new_bias = [b + 0.02 for b in bias]  
        catboost_model.set_scale_and_bias(scale, new_bias)
        print(f"Bias corrected: {bias} -> {new_bias}")
        """

        print(f"CatBoost multi-output model loaded from {MODEL_PATH}")
        print(f"Model will predict: [MMR, Selling Price]")

    except Exception as e:
        print(f"Error loading CatBoost model: {e}")
        catboost_model = None


# Load the model immediately on startup
load_catboost_model()


# Input Schema for LangChain Tool
class CarPriceInput(BaseModel):
    """Input for the CatBoost Car Price Predictor tool (11 fields + prediction_type)."""

    # ... (Your existing fields are correct and match EXPECTED_FEATURES) ...
    make: str = Field(description="The brand of the car (e.g., 'Honda', 'Ford').")
    model: str = Field(description="The specific model (e.g., 'Camry', 'F-150').")
    body: str = Field(description="The body style (e.g., 'Sedan', 'SUV', 'Truck').")
    transmission: str = Field(
        description="The transmission type ('Automatic' or 'Manual')."
    )
    state: str = Field(
        description="The state where the car is being sold (e.g., 'CA', 'FL')."
    )
    condition_category: str = Field(
        description="The subjective condition of the car ('Good', 'Okay', 'Poor')."
    )
    odometer: int = Field(description="The mileage of the car.")
    car_age: int = Field(description="The age of the car in years.")
    sale_year: int = Field(description="The year the car is being valued (e.g., 2024).")
    sale_month: int = Field(
        description="The month the car is being valued (1=Jan, 12=Dec)."
    )
    seller_category: str = Field(
        description="The seller category ('Financial', 'Fleet', 'Rental', 'Manufacturer')."
    )
    prediction_type: str = Field(
        default="both",
        description="What to predict: 'mmr' (Manheim Market Report), 'selling_price', or 'both'. Default is 'both'.",
    )


# Prediction function for multi-output model
def get_catboost_car_price(**kwargs) -> str:
    """Predicts MMR and/or selling price using the multi-output CatBoost model."""
    global catboost_model
    if not catboost_model:
        return "PREDICTION_RESULT: Prediction model is unavailable. Cannot calculate price."

    try:
        # Extract prediction type preference
        prediction_type = kwargs.get("prediction_type", "both").lower()
        if prediction_type not in ["mmr", "selling_price", "both"]:
            return f"ERROR: Invalid prediction_type '{prediction_type}'. Must be 'mmr', 'selling_price', or 'both'."

        feature_values = {}
        for feature in EXPECTED_FEATURES:
            value = kwargs.get(feature)
            if value is None:
                return f"ERROR: Missing required feature '{feature}'."

            # Apply log transformations to 'odometer', 'car_age'
            if feature in ["odometer", "car_age"]:
                # Store the transformed (float) value
                feature_values[feature] = np.log1p(float(value))
            elif feature in ["sale_year", "sale_month"]:
                # Store numerical values as integers/floats
                feature_values[feature] = int(value)
            else:
                # Store categorical values as strings
                feature_values[feature] = str(value)

        # --- CRITICAL FIX: Use Pandas DataFrame for robust type handling ---
        # 1. Create a DataFrame from the single row of features
        input_df = pd.DataFrame([feature_values], columns=EXPECTED_FEATURES)

        # 2. Predict (CatBoost handles the DataFrame types reliably)
        # Prediction output is a 2D numpy array: [[log_mmr, log_selling_price]]
        log_predictions = catboost_model.predict(input_df)[0]
        # --- END OF FIX ---

        # Inverse transform to get dollar values
        # Index 0 = MMR, Index 1 = Selling Price
        mmr = np.expm1(log_predictions[0])
        selling_price = np.expm1(log_predictions[1])

        # Return based on user preference
        if prediction_type == "mmr":
            return (
                f"PREDICTION_RESULT: The predicted MMR (auction value) is ${mmr:,.2f}"
            )
        elif prediction_type == "selling_price":
            return f"PREDICTION_RESULT: The predicted retail selling price is ${selling_price:,.2f}"
        else:  # both
            return f"PREDICTION_RESULT: The predicted MMR is ${mmr:,.2f} and the predicted retail selling price is ${selling_price:,.2f}"

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"ERROR: Prediction failed. Details: {e}"


def process_car_price_input(json_input: str) -> str:
    """Parses JSON string input and calls the prediction function."""
    try:
        # LOGGING: Capture what the agent is sending
        with open("agent_input.log", "a") as f:
            f.write(f"RAW_INPUT: {json_input}\n")
            f.flush()

        # The agent returns a string like '{"year": 2021, ...}'
        if isinstance(json_input, dict):
            data = json_input
        else:
            # Clean up potential markdown code blocks if the agent adds them
            cleaned_input = json_input.strip().strip("`").replace("json\n", "")
            data = json.loads(cleaned_input)

        # --- ROBUSTNESS FIX: Normalize Categorical Inputs ---
        # Map common synonyms to the exact strings expected by CatBoost

        # Seller Category Normalization
        seller = data.get("seller_category", "").title()  # Ensure Title Case
        if (
            "Manufact" in seller
        ):  # Matches "Manufacturing", "Manufacturer", "Manufactoring"
            data["seller_category"] = "Manufacturer"
        elif "Rent" in seller:
            data["seller_category"] = "Rental"
        elif "Fleet" in seller:
            data["seller_category"] = "Fleet"
        elif "Financ" in seller:
            data["seller_category"] = "Financial"

        # Condition Normalization - convert to lowercase as expected by model
        cond = data.get("condition_category", "").lower()
        if "excell" in cond:
            data["condition_category"] = "Good"  # Map 'excellent' to 'good'
        elif "good" in cond:
            data["condition_category"] = "Good"
        elif "fair" in cond or "okay" in cond or "ok" in cond:
            data["condition_category"] = "Okay"
        elif "poor" in cond or "bad" in cond:
            data["condition_category"] = "Poor"

        # Transmission Normalization
        trans = data.get("transmission", "").title()
        if "Auto" in trans:
            data["transmission"] = "Automatic"
        elif "Man" in trans:
            data["transmission"] = "Manual"

        return get_catboost_car_price(**data)

    except json.JSONDecodeError:
        return 'ERROR: Input must be a valid JSON string. Example: {"year": 2020, "make": "Toyota", ...}'
    except Exception as e:
        return f"ERROR: Invalid input format or missing fields. Details: {e}"


catboost_tool = Tool.from_function(
    func=process_car_price_input,
    name="CarPricePredictor",
    description='Use this tool to predict MMR and/or selling price of a car. INPUT must be a valid JSON string containing 11 required fields + optional \'prediction_type\'. CRITICAL: \'seller_category\' MUST be one of [\'Financial\', \'Fleet\', \'Rental\', \'Manufacturer\']. \'state\' must be a 2-letter code (e.g., \'CA\'). \'prediction_type\' can be \'mmr\', \'selling_price\', or \'both\' (default). Example Input: \'{"make": "Honda", "model": "Civic", "body": "Sedan", "transmission": "Automatic", "state": "CA", "condition_category": "Good", "odometer": 50000, "car_age": 5, "sale_year": 2024, "sale_month": 12, "seller_category": "Manufacturer", "prediction_type": "both"}\'',
)

# =================================================================
# 2. RAG CONTEXT TOOL SETUP
# =================================================================


def load_rag_tool():
    """Loads the pre-built FAISS index from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"RAG Error: Saved index folder '{FAISS_INDEX_PATH}' not found.")
        print("Please run the index creation script first!")
        return None

    try:
        # Load embeddings (CRITICAL: Must use the SAME embedding model as used for saving)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Load the index from disk (FAST!)
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        news_retriever = vector_store.as_retriever(k=3)

        def get_market_context(query: str) -> str:
            """Retrieves relevant market news context."""
            docs = news_retriever.invoke(query)

            # FIX 3: Ensure metadata keys are correct ('headline', not 'title')
            context = "\n---\n".join(
                [
                    f"HEADLINE: {doc.metadata.get('headline', 'N/A')} ({doc.metadata.get('category', 'N/A')})\nTEXT: {doc.page_content}"
                    for doc in docs
                ]
            )
            return f"RETRIEVED_MARKET_CONTEXT:\n{context}"

        rag_tool = Tool.from_function(
            func=get_market_context,
            name="MarketNewsRetriever",
            description="Use this tool to search for recent market news, supply chain issues, or general trends related to a specific car make/model or market condition.",
        )
        print("RAG tool successfully loaded from disk.")
        return rag_tool

    except Exception as e:
        print(f"RAG loading failed: {e}")
        return None


rag_tool = load_rag_tool()

# =================================================================
# 3. LANGCHAIN AGENT SETUP
# =================================================================

# FIX 4: Add the catboost_tool to the initial list
tools = [catboost_tool]
if rag_tool:
    tools.append(rag_tool)
else:
    print("Warning: MarketNewsRetriever (RAG) tool not included in agent.")

llm = ChatOpenAI(temperature=0, model=LLM_MODEL)

# START OF MODIFIED PROMPT
REACT_PROMPT_TEMPLATE = """\
You are an expert Used Car Price Predictor and Market Analyst. Your primary goal is to **Predict the Manheim Market Report (MMR) value and/or Final Retail Selling Price of a Used Car based on detailed vehicle characteristics.**

You have access to the following tools:

{tools}

**GUIDELINES:**
1. **Price Prediction:** If the user asks for MMR, selling price, or both, and provides *all 11 necessary characteristics* (make, model, body, transmission, state, condition_category, odometer, car_age, sale_year, sale_month, seller_category), use the **CarPricePredictor** tool with the appropriate prediction_type ('mmr', 'selling_price', or 'both').
2. **Market Context:** If the user asks about market conditions, trends, or is missing information for a prediction, use the **MarketNewsRetriever** tool to gather context before formulating an answer.
3. **Final Answer:** Always use the precise **PREDICTION_RESULT** or **RETRIEVED_MARKET_CONTEXT** from the tool output to construct your final, informative answer.


Use the following strict ReAct format:

Question: the input question you must answer
Thought: you should always think about what to do, based on the primary goal and guidelines.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must match the tool's required schema)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**EXAMPLE SESSIONS:**
Question: "Can you predict both the MMR and retail price for a 2018 Toyota Camry Sedan with Automatic transmission in CA? It's in Good condition with 50000 miles, sold in December 2024 by a Financial institution."
Thought: The user wants both MMR and selling price. I have all 11 features. I should use CarPricePredictor with prediction_type='both'.
Action: CarPricePredictor
Action Input: {{"make": "Toyota", "model": "Camry", "body": "Sedan", "transmission": "Automatic", "state": "CA", "condition_category": "Good", "odometer": 50000, "car_age": 6, "sale_year": 2024, "sale_month": 12, "seller_category": "Financial", "prediction_type": "both"}}
Observation: PREDICTION_RESULT: The predicted MMR is $18,500.00 and the predicted retail selling price is $19,800.00
Thought: I have both predictions.
Final Answer: For a 2018 Toyota Camry in Good condition, the estimated MMR (auction value) is $18,500.00 and the estimated retail selling price is $19,800.00.
**BEGIN ACTUAL SESSION:**
Question: {input}
Thought: you should always think about what to do, based on the primary goal and guidelines.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must match the tool's required schema)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
# END OF MODIFIED PROMPT

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


# =================================================================
# 4. FLASK ROUTES (No changes needed here)
# =================================================================


@app.route("/")
def home():
    """Renders the main chatbot interface page."""
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask_agent():
    """Receives the user's question and runs the LangChain agent."""
    try:
        data = request.get_json()
        user_input = data.get("question", "").strip()

        # CRITICAL FIX: Ensure user_input is not empty before invoking the agent
        if not user_input:
            return jsonify(
                {
                    "response": "Please enter a question about a car price or market trend to begin."
                }
            )

        # Run the Agent Executor
        result = agent_executor.invoke({"input": user_input})

        return jsonify({"response": result["output"]})

    except Exception as e:
        error_msg = f"An unexpected error occurred during agent execution: {e}. Try rephrasing your query."
        return jsonify({"response": error_msg})


if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")
    app.run(host="0.0.0.0", port=5001, debug=True)
