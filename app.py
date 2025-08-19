# import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import time
import streamlit as st
import datetime
import plotly.express as px
import pandas as pd
import re
from io import StringIO
import docx
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = "./flan t5"

# Custom CSS for professional styling
def load_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .stAlert > div {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .summary-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .input-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        margin: 15px 0;
    }
    
    .stats-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load base model + LoRA adapter, and tokenizer with progress tracking."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading tokenizer...")
        progress_bar.progress(25)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        status_text.text("Loading base model...")
        progress_bar.progress(50)
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        
        status_text.text("Loading LoRA adapter...")
        progress_bar.progress(75)
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        
        status_text.text("Model loaded successfully!")
        progress_bar.progress(100)
        time.sleep(1)
        
        progress_bar.empty()
        status_text.empty()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def analyze_text(text):
    """Analyze text statistics."""
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    sentences = len(re.findall(r'[.!?]+', text))
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    
    return {
        'words': words,
        'characters': chars,
        'characters_no_spaces': chars_no_spaces,
        'sentences': sentences,
        'paragraphs': paragraphs,
        'avg_words_per_sentence': round(words / max(sentences, 1), 2)
    }

def summarize_text(text, tokenizer, model, max_new_tokens=80, num_beams=4, length_penalty=2.0):
    """Enhanced summarization with error handling and timing."""
    start_time = time.time()
    
    input_text = "summarize: " + text
    
    try:
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        )

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                length_penalty=length_penalty,
                early_stopping=True,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        
        return summary, processing_time
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None, 0

def save_summary_history(original_text, summary, stats):
    """Save summary to session state history."""
    if 'summary_history' not in st.session_state:
        st.session_state.summary_history = []
    
    entry = {
        'timestamp': datetime.datetime.now(),
        'original_text': original_text[:100] + "..." if len(original_text) > 100 else original_text,
        'summary': summary,
        'original_length': stats['words'],
        'summary_length': len(summary.split()),
        'compression_ratio': round(len(summary.split()) / stats['words'] * 100, 1)
    }
    
    st.session_state.summary_history.append(entry)

def read_uploaded_file(uploaded_file):
    """Read content from uploaded file."""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.error("Unsupported file type. Please upload .txt or .docx files.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(
        page_title="Professional Text Summarizer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    # Header
    st.markdown('<h1 class="title-gradient" style="font-size: 4rem;">🚀 Professional Text Summarizer 💎</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle" style="font-size: 1.8rem;">🏆 Advanced T5 with LoRA Fine-tuning 🎯 | 👨‍💻 By AbdUllah Samir 💪</p>', unsafe_allow_html=True)
    # Load model
    with st.spinner("🔄 Loading AI model... Please wait"):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("❌ Failed to load model. Please check your model path and try again.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Advanced settings
        st.subheader("Summarization Parameters")
        max_tokens = st.slider("Max New Tokens", 30, 300, 80, 10)
        num_beams = st.slider("Beam Search Size", 1, 8, 4, 1)
        length_penalty = st.slider("Length Penalty", 0.5, 3.0, 2.0, 0.1)
        
        st.subheader("Input Options")
        input_method = st.radio("Choose Input Method:", ["✍️ Type Text", "📁 Upload File"])
        
        st.subheader("Output Options")
        show_stats = st.checkbox("📊 Show Text Statistics", True)
        show_comparison = st.checkbox("📈 Show Compression Analysis", True)
        
        # Model Info
        st.subheader("🤖 Model Information")
        st.info(f"""
        **Base Model:** {BASE_MODEL}
        **Adapter:** LoRA Fine-tuned
        **Task:** Text Summarization
        **Language:** Multi-language Support
        """)
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        if input_method == "✍️ Type Text":
            article = st.text_area(
                "📝 Enter your text to summarize:",
                height=300,
                placeholder="Paste your article, dialogue, or any text here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "📁 Upload a text file",
                type=['txt', 'docx'],
                help="Supported formats: .txt, .docx"
            )
            article = ""
            if uploaded_file:
                article = read_uploaded_file(uploaded_file)
                if article:
                    st.text_area("📝 File Content:", value=article[:500] + "..." if len(article) > 500 else article, height=200, disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            summarize_btn = st.button("🚀 Summarize Text", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("🗑️ Clear Text", use_container_width=True):
                st.rerun()
        
        with col_btn3:
            if st.button("📜 View History", use_container_width=True):
                st.session_state.show_history = not st.session_state.get('show_history', False)
    
    with col2:
        if article and article.strip():
            stats = analyze_text(article)
            
            if show_stats:
                st.markdown("### 📊 Text Statistics")
                
                # Create metrics display
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Words", stats['words'])
                    st.metric("Sentences", stats['sentences'])
                    st.metric("Paragraphs", stats['paragraphs'])
                
                with col_stat2:
                    st.metric("Characters", stats['characters'])
                    st.metric("Avg Words/Sentence", stats['avg_words_per_sentence'])
                    
                # Reading time estimation
                reading_time = max(1, round(stats['words'] / 200))  # 200 words per minute
                st.info(f"📖 Estimated reading time: {reading_time} minute(s)")
    
    # Summarization Process
    if summarize_btn:
        if not article or not article.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("🔄 Analyzing and summarizing your text..."):
                summary, processing_time = summarize_text(
                    article, tokenizer, model, 
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                    length_penalty=length_penalty
                )
            
            if summary:
                # Display results
                st.markdown("### ✨ Summary Result")
                st.markdown(f'<div class="summary-container"><h4>📋 Generated Summary:</h4><p>{summary}</p></div>', unsafe_allow_html=True)
                
                # Performance metrics
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                
                original_stats = analyze_text(article)
                summary_stats = analyze_text(summary)
                compression_ratio = round((summary_stats['words'] / original_stats['words']) * 100, 1)
                
                with col_perf1:
                    st.markdown(f'<div class="stats-card"><h4>⚡ Processing Time</h4><p>{processing_time:.2f} seconds</p></div>', unsafe_allow_html=True)
                
                with col_perf2:
                    st.markdown(f'<div class="stats-card"><h4>📏 Compression Ratio</h4><p>{compression_ratio}%</p></div>', unsafe_allow_html=True)
                
                with col_perf3:
                    st.markdown(f'<div class="stats-card"><h4>💾 Words Saved</h4><p>{original_stats["words"] - summary_stats["words"]} words</p></div>', unsafe_allow_html=True)
                
                if show_comparison:
                    # Comparison chart
                    comparison_data = pd.DataFrame({
                        'Metric': ['Words', 'Characters', 'Sentences'],
                        'Original': [original_stats['words'], original_stats['characters'], original_stats['sentences']],
                        'Summary': [summary_stats['words'], summary_stats['characters'], summary_stats['sentences']]
                    })
                    
                    fig = px.bar(comparison_data, x='Metric', y=['Original', 'Summary'], 
                               title="📈 Text Comparison: Original vs Summary",
                               barmode='group',
                               color_discrete_map={'Original': '#ff6b6b', 'Summary': '#4ecdc4'})
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save to history
                save_summary_history(article, summary, original_stats)
                
                # Download options
                st.markdown("### 💾 Download Options")
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        "📄 Download Summary (TXT)",
                        summary,
                        f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )
                
                with col_dl2:
                    report = f"""
SUMMARY REPORT
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ORIGINAL TEXT STATS:
- Words: {original_stats['words']}
- Characters: {original_stats['characters']}
- Sentences: {original_stats['sentences']}

SUMMARY STATS:
- Words: {summary_stats['words']}
- Compression Ratio: {compression_ratio}%
- Processing Time: {processing_time:.2f}s

ORIGINAL TEXT:
{article}

GENERATED SUMMARY:
{summary}
                    """
                    st.download_button(
                        "📊 Download Full Report",
                        report,
                        f"summary_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )
    
    # History Section
    if st.session_state.get('show_history', False):
        st.markdown("### 📜 Summary History")
        if 'summary_history' in st.session_state and st.session_state.summary_history:
            history_df = pd.DataFrame(st.session_state.summary_history)
            history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                history_df[['timestamp', 'original_length', 'summary_length', 'compression_ratio']],
                column_config={
                    'timestamp': 'Time',
                    'original_length': 'Original Words',
                    'summary_length': 'Summary Words',
                    'compression_ratio': 'Compression %'
                },
                use_container_width=True
            )
            
            if st.button("🗑️ Clear History"):
                st.session_state.summary_history = []
                st.rerun()
        else:
            st.info("📝 No summaries generated yet.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 Professional Text Summarizer | Powered by T5 + LoRA | Created by AbdUllah Samir</p>
        <p>⚡ Advanced AI • 🎯 High Accuracy • 🚀 Fast Processing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()