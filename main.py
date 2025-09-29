import json
import pandas as pd
import numpy as np
import os
import re
import argparse
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

# -------------------- Load environment --------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in .env. Gemini fallback will not work.")

# -------------------- Constants --------------------
MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = "models/finetuned_t5"
DATA_PATH = "data/data.json"
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4

# -------------------- Load JSON data --------------------
with open(DATA_PATH, "r") as f:
    data_json = json.load(f)
# print(f"Raw data_json keys: {list(data_json.keys())}")
metadata_json = {}
for state, records in data_json.items():
    print(f"Processing state: {state}, Records: {len(records)}")
    for r in records:
        r['state'] = state.strip().upper()
        r['location'] = r['location'].strip().upper()
        try:
            extraction = r.get('ground_water_extraction_ham')
            extractable = r.get('annual_extractable_ground_water_resources_ham')
            if extraction is None or extractable is None or extractable == 0:
                # print(f"Skipping record due to invalid groundwater data: {r}")
                r['sgwd'] = None
                r['approx_category'] = 'N/A'
            else:
                r['sgwd'] = (float(extraction) / float(extractable)) * 100
                r['approx_category'] = ('Safe' if r['sgwd'] <= 70 else
                                       'Semi-Critical' if r['sgwd'] <= 90 else
                                       'Critical' if r['sgwd'] < 100 else
                                       'Over-Exploited')
        except (TypeError, ValueError) as e:
            print(f"Error processing record {r}: {e}")
            r['sgwd'] = None
            r['approx_category'] = 'N/A'
        key = f"{r['location']}_{r['state']}_{r['year'].strip()}"
        metadata_json[key] = r
# print(f"Metadata keys: {list(metadata_json.keys())}")  # Debug print 

translator = GoogleTranslator(source='auto', target='en')

# -------------------- Helper Functions --------------------

def translate_to_en(text):
    try:
        return translator.translate(text)
    except:
        return text

def translate_to_target(text, lang):
    if lang == 'en':
        return text
    try:
        target_translator = GoogleTranslator(source='en', target=lang)
        return target_translator.translate(text)
    except:
        return text

def detect_query_type(query):
    q = query.lower()
    if "rain" in q:
        return "rainfall"
    elif "groundwater" in q or "sgwd" in q or "extraction" in q or "category" in q:
        return "groundwater"
    else:
        return "unknown"

# -------------------- Helper Functions (Updated) --------------------
def extract_location_and_year(query):
    print(f"Raw query: {query}")  # Debug print to verify input
    # Extract year (handles 2023, 2023-24, 2023-2024)
    match = re.search(r"\b(\d{4}(?:-\d{2,4})?)\b", query)
    year = match.group(1) if match else None
    
    # Remove year from query for cleaning
    query_clean = query if not year else query.replace(year, "").strip()
    
    # Define query type patterns and their mappings
    query_type_patterns = {
        r"(?:rainfall|rain)\s*(?:details|level|information|of)?": "rainfall",
        r"(?:groundwater|gw)\s*(?:details|level|information|of)?": "groundwater",
        r"full\s*(?:information|category|details)\s*of": "full_info",
        r"details\s*of\s*(?:rainfall|groundwater)": lambda m: "rainfall" if "rainfall" in m.group(0).lower() else "groundwater"
    }
    
    # Determine query type
    query_type = None
    for pattern, q_type in query_type_patterns.items():
        if re.search(pattern, query_clean.lower()):
            query_type = q_type if isinstance(q_type, str) else q_type(re.search(pattern, query_clean.lower()))
            break
    
    # Default to "groundwater" if no specific type detected but "details" or "level" is present
    if not query_type and re.search(r"\b(details|level|information)\b", query_clean.lower()):
        query_type = "groundwater"
    
    # Remove query type keywords and prepositions, preserve location terms
    query_clean = re.sub(r"(groundwater|gw|rainfall|rain|details|level|information|full|category|of|for|in|\?|and)(?=\s+|$)", "", query_clean, flags=re.I).strip()
    query_clean = re.sub(r"\s+", " ", query_clean)  # Replace multiple spaces with single space
    print(f"Query clean: {query_clean}")  # Debug to check cleaning
    
    # Split into words and filter meaningful location parts
    parts = [p for p in query_clean.split() if p and not p.lower() in {"the", "at", "on", "of"}]
    print(f"Raw parts after cleaning: {parts}")  # Debug print
    
    # Use the first non-empty part or join parts as location, infer state
    state = None
    location = " ".join(parts) if parts else None  # Join all parts to preserve the location
    
    if location:
        # Infer state based on the location
        state = infer_state(location)
        location = normalize(location) if location else None
    
    # Handle "full_info" query type by mapping to existing types
    if query_type == "full_info":
        if re.search(r"rainfall", query.lower()):
            query_type = "rainfall"
        elif re.search(r"groundwater", query.lower()):
            query_type = "groundwater"
    
    # Normalize location for matching
    print(f"Parsed: location={location}, state={state}, year={year}, query_type={query_type}")  # Debug print
    return location, state, year, query_type

def normalize(s):
    if not s:
        return ""
    # Simple normalization: remove extra spaces, convert to uppercase
    return re.sub(r"\s+", " ", s.strip().upper())

def infer_state(location):
    loc_norm = normalize(location)
    if not loc_norm:
        print(f"No state inferred for empty location")
        return None
    for record in metadata_json.values():
        if normalize(record['location']).startswith(loc_norm) or normalize(record['location']) == loc_norm:
            print(f"Inferred state for {loc_norm}: {record['state']}")
            return record['state']
    print(f"No state inferred for {loc_norm}")
    return None

def find_json_record(location, state, year, query_type):
    if not location or not year:
        print(f"Invalid input: location={location}, year={year}")
        return None
    loc_norm = normalize(location)
    year_norm = re.match(r"(\d{4})", year.replace(" ", "").replace("-", "")).group(1) if year else ""
    print(f"Searching for: location={loc_norm}, state={state}, year={year_norm}, query_type={query_type}")  # Debug print
    
    for record in metadata_json.values():
        record_loc_norm = normalize(record['location'])
        record_year_norm = re.match(r"(\d{4})", record['year'].replace(" ", "").replace("-", "")).group(1)
        print(f"Checking record: {record_loc_norm}, {record['state']}, {record['year']}")  # Debug print
        
        # Flexible matching for location and year
        if (loc_norm == record_loc_norm or loc_norm in record_loc_norm or 
            record_loc_norm.startswith(loc_norm)) and \
           (not state or normalize(record['state']) == normalize(state)) and \
           (record_year_norm == year_norm or record_year_norm.startswith(year_norm)):
            print(f"JSON Match found: {record['location']}, {record['state']}, {record['year']}")
            result = format_record(record, query_type)
            if result:
                return result
            else:
                print(f"Format record failed for match: {record}")
    print(f"No JSON match found for location={loc_norm}, state={state}, year={year_norm}")
    return None

def format_record(record, query_type):
    print(f"Formatting record: {record}, Query type: {query_type}, rainfall_mm: {record.get('rainfall_mm')}")
    if query_type == "rainfall" and 'rainfall_mm' in record and record['rainfall_mm'] is not None:
        return f"Rainfall: {record['rainfall_mm']:.1f} mm"
    elif query_type == "groundwater" and 'sgwd' in record and record['sgwd'] is not None:
        return (f"SGWD: {record['sgwd']:.1f}% ; "
                f"Extraction: {record['ground_water_extraction_ham']:.1f} ha m ; "
                f"Extractable: {record['annual_extractable_ground_water_resources_ham']:.1f} ha m ; "
                f"Category: {record.get('approx_category', 'N/A')}")
    elif query_type in ["full_info", "full_category"]:
        rainfall_str = f"Rainfall: {record['rainfall_mm']:.1f} mm" if 'rainfall_mm' in record and record['rainfall_mm'] is not None else "No rainfall data available."
        groundwater_str = (f"SGWD: {record['sgwd']:.1f}% ; "
                           f"Extraction: {record['ground_water_extraction_ham']:.1f} ha m ; "
                           f"Extractable: {record['annual_extractable_ground_water_resources_ham']:.1f} ha m ; "
                           f"Category: {record.get('approx_category', 'N/A')}") if 'sgwd' in record and record['sgwd'] is not None else "No groundwater data available."
        return f"{rainfall_str}\n{groundwater_str}"
    print("Format record failed, returning None")
    return None

# -------------------- Data Processing --------------------
def load_data():
    df = pd.DataFrame([v for v in metadata_json.values()])
    return df

def generate_qa(df):
    qa_pairs = []
    for _, row in df.head(500).iterrows():
        sgwd_str = f"{row['sgwd']:.1f}%" if pd.notnull(row['sgwd']) else "N/A"
        extraction_str = f"{row['ground_water_extraction_ham']:.1f}" if pd.notnull(row['ground_water_extraction_ham']) else "N/A"
        extractable_str = f"{row['annual_extractable_ground_water_resources_ham']:.1f}" if pd.notnull(row['annual_extractable_ground_water_resources_ham']) else "N/A"
        questions = [
            f"What is the SGWD for {row['location']} in {row['state']} {row['year']}?",
            f"Full category for {row['location']}, {row['state']} {row['year']} with decline?",
            f"Extraction trend in {row['state']} {row['location']}?"
        ]
        answers = [
            f"SGWD: {sgwd_str}. Approx category: {row['approx_category']}.",
            f"Approx category: {row['approx_category']} (SGWD {sgwd_str}). Full needs decline data.",
            f"Extraction: {extraction_str} ha m, Extractable: {extractable_str} ha m."
        ]
        for q, a in zip(questions, answers):
            qa_pairs.append({'question': q, 'answer': a})
    return Dataset.from_list(qa_pairs)

# -------------------- Fine-Tune Model --------------------
def fine_tune_model(qa_dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q", "v"])
    model = get_peft_model(model, lora_config)
    def preprocess_function(examples):
        inputs = [f"question: {q}" for q in examples['question']]
        targets = [f"answer: {a}" for a in examples['answer']]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
        labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    tokenized_dataset = qa_dataset.map(preprocess_function, batched=True, remove_columns=qa_dataset.column_names)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

# -------------------- FAISS --------------------
def build_faiss_index(df):
    texts = []
    for _, row in df.iterrows():
        sgwd_str = f"{row['sgwd']:.1f}%" if pd.notnull(row['sgwd']) else "N/A"
        rainfall_str = f"{row['rainfall_mm']:.1f} mm" if pd.notnull(row['rainfall_mm']) else "N/A"
        texts.append(
            f"State: {row['state']}, Location: {row['location']}, Year: {row['year']}, SGWD: {sgwd_str}, Rainfall: {rainfall_str}"
        )
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embedded_docs = embeddings_model.embed_documents(texts)
    documents = [
        Document(
            page_content=t,
            metadata=row.to_dict()
        )
        for t, (_, row) in zip(texts, df.iterrows())
    ]
    vectorstore = FAISS.from_documents(documents, embeddings_model)
    vectorstore.save_local('faiss_index')
    return vectorstore

# -------------------- RAG with Fallback --------------------
def rag_with_fallback(query, vectorstore=None, lang='en'):
    en_query = translate_to_en(query)
    query_type = detect_query_type(en_query)
    location, state, year, query_type = extract_location_and_year(en_query)
    print(f"Query: {en_query}, Type: {query_type}, Location: {location}, State: {state}, Year: {year}")

    valid_states = {normalize(record['state']) for record in metadata_json.values()}
    if state and normalize(state) in valid_states:
        json_result = find_json_record(location, state, year, query_type)
        if json_result:
            print("Source: JSON")
            return json_result, []
        else:
            print("No JSON match found.")

    if vectorstore:
        results = vectorstore.similarity_search_with_score(en_query, k=3)
        if results and results[0][1] < 0.3:
            top_doc = results[0][0]
            meta = top_doc.metadata
            print(f"FAISS Match: {meta['location']}, {meta['state']}, {meta['year']}")
            if query_type == "rainfall" and 'rainfall_mm' in meta and meta['rainfall_mm'] is not None:
                return f"Rainfall: {meta['rainfall_mm']:.1f} mm", [doc.metadata for doc, _ in results]
            elif query_type == "groundwater" and 'sgwd' in meta and meta['sgwd'] is not None:
                return (f"SGWD: {meta['sgwd']:.1f}% ; "
                        f"Extraction: {meta['ground_water_extraction_ham']:.1f} ha m ; "
                        f"Extractable: {meta['annual_extractable_ground_water_resources_ham']:.1f} ha m ; "
                        f"Category: {meta['approx_category']}"), [doc.metadata for doc, _ in results]
        else:
            print("No suitable FAISS match found.")

    loc_info = f"{location or 'Unknown'}, {state or 'Unknown'}, {year or 'Unknown'}"
    print("Source: Gemini")
    if GOOGLE_API_KEY is not None and GOOGLE_API_KEY.strip() != '':
        # Try fine-tuned T5 first
        try:
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
            inputs = tokenizer(f"question: {en_query}", return_tensors="pt", max_length=128, truncation=True)
            outputs = model.generate(**inputs)
            t5_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if t5_answer.startswith("answer: "):
                return translate_to_target(t5_answer[8:], lang), []
        except Exception as e:
            print(f"T5 error: {e}")
        
        # Gemini fallback
        if query_type == "rainfall":
            prompt_template = """You are a meteorological data expert. For {location_info}, respond to: {question}
            If no specific rainfall data is available, state clearly that data is unavailable for the specified location and year, and suggest checking with local meteorological agencies."""
        else:
            prompt_template = """You are a groundwater expert. For {location_info}, respond to: {question}
            If no specific groundwater data is available, state clearly that data is unavailable and suggest checking with local groundwater agencies.
            Use this matrix for groundwater categorization if relevant:
            Sr. No. | SGWD | Pre-monsoon Decline | Post-monsoon Decline | Category
            1 | <=70% | No | No | Safe
            2 | >70% & <90% | No | No | Safe
            3 | >70% & <90% | Yes/No | - | Semi-Critical
            4 | >90% & <100% | Yes/No | - | Semi-Critical
            5 | >90% & <100% | Yes | Yes | Critical
            6 | >100% | Yes/No | - | Over-Exploited
            7 | >100% | Yes | Yes | Over-Exploited"""
        structured_prompt = PromptTemplate(
            input_variables=["location_info", "question"],
            template=prompt_template
        )
        gemini_llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
        answer = gemini_llm.invoke(structured_prompt.format(location_info=loc_info, question=en_query))
        print(f"Raw Gemini response: {answer}")
        return translate_to_target(answer, lang), []
    print("Source: None (Gemini unavailable)")
    return f"No data found for {location or 'Unknown'} in {year or 'Unknown'}. Gemini fallback unavailable due to missing API key.", []

# -------------------- Visualization --------------------
def plot_extraction_trend(state, location=None):
    df = load_data()
    if location:
        filtered = df[(df['state'].str.upper() == state.upper()) & (df['location'].str.upper() == location.upper())]
    else:
        filtered = df[df['state'].str.upper() == state.upper()]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    if filtered.empty:
        ax.text(0.5, 0.5, 'No data available for selected state and location.', ha='center', va='center')
        ax.axis('off')
    else:
        ax.plot(filtered['year'], filtered['ground_water_extraction_ham'], marker='o', label='Extraction')
        ax.axhline(y=filtered['annual_extractable_ground_water_resources_ham'].mean(), color='r', linestyle='--', label='Avg Extractable')
        ax.set_title(f"Trend: {state} ({location or 'All'})")
        ax.set_xlabel('Year')
        ax.set_ylabel('ha m')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('trend.png')
    plt.close()
    return fig

def plot_category_bar(state, year_str):
    df = load_data()
    filtered = df[(df['state'].str.upper() == state.upper()) & (df['year'].str.contains(year_str))]
    if filtered.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No data available for selected state and year.', ha='center', va='center')
        ax.axis('off')
        return fig
    cat_counts = filtered['approx_category'].value_counts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red'], ax=ax)
    ax.set_title(f"Categories: {state} {year_str}")
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('categories.png')
    plt.close()
    return fig


# -------------------- Main Execution --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INGRES Groundwater Chatbot")
    parser.add_argument("--train", action="store_true", help="Run fine-tuning")
    parser.add_argument("--test-query", type=str, help="Test a specific query", default="rainfall in Delhi Civil Lines 2024")
    args = parser.parse_args()

    df = load_data()
    if args.train:
        qa_dataset = generate_qa(df)
        fine_tune_model(qa_dataset)
    else:
        if os.path.exists("faiss_index"):
            embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            if not os.path.isdir("faiss_index"):
                raise ValueError("FAISS index directory is invalid or corrupted.")
            vectorstore = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
            print("Loaded FAISS index from 'faiss_index'.")
        else:
            vectorstore = build_faiss_index(df)
        # Test query with command-line option
        test_query = args.test_query
        answer, sources = rag_with_fallback(test_query, vectorstore)
        print(f"Test Query: {test_query}")
        print(f"Test Answer: {answer}")