

A complete, locally-run information retrieval system implementing multiple retrieval strategies with comprehensive evaluation metrics.


## Features

### Multiple Retrieval Strategies
- **Boolean Retrieval**: Classic Boolean AND model with term matching
- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting
- **BM25**: Okapi BM25 probabilistic ranking function
- **Cosine Similarity**: Vector space model with cosine similarity scoring

### Similarity Metrics
- Jaccard Coefficient
- Dice Coefficient
- Overlap Coefficient
- TF-IDF Score
- BM25 Score

### Evaluation & Visualization
- Precision, Recall, F1-Score calculations
- Search time performance analysis
- Similarity metrics heatmaps
- Strategy comparison charts
- Batch query evaluation

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Operating System: Windows, macOS, or Linux

## Installation

### 1. Clone or Download Repository
```bash
git clone https://github.com/yourusername/ir-system-cs516.git
cd ir-system-cs516
```

### 2. Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation

### Document Format
Your documents file should be in one of these formats:

**CSV Format** (Recommended):
```csv
text
"This is the first document content"
"This is the second document content"
"This is the third document content"
```

**TXT Format** (One document per line):
```
This is the first document content
This is the second document content
This is the third document content
```

### Query Format
Queries should be in a text file, one query per line:
```
machine learning algorithms
information retrieval systems
natural language processing
```

## Usage

### Basic Usage - Interactive Mode

```bash
python main.py
```

The system will prompt you for:
1. Path to documents file
2. Enter queries interactively

### Example Session

```
Enter path to documents file (CSV or TXT): data/documents.csv
Loading documents from data/documents.csv...
Loaded 1000 documents
Preprocessing documents...
Preprocessing complete
Building inverted index...
Index built in 2.34 seconds
Vocabulary size: 5432
Average document length: 45.67 tokens

Enter query: machine learning algorithms
Enter strategy (default: bm25): 

Results will be displayed with scores and metrics
```

### Comparing All Strategies

```
Enter query: compare
Enter query to compare: information retrieval

# System will show results from all 4 strategies side-by-side
```

### Batch Evaluation

```
Enter query: evaluate
Enter path to queries file: data/queries.txt

# System will:
# 1. Process all queries with all strategies
# 2. Generate visualization plots
# 3. Save results to results/evaluation_results.json
```

## Project Structure

```
ir-system-cs516/
├── main.py                      # Main IR system implementation
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/                        # Data directory
│   ├── documents.csv           # Your document dataset
│   └── queries.txt             # Your query dataset
├── results/                     # Output directory (auto-created)
│   ├── evaluation_results.json
│   ├── search_time_comparison.png
│   └── similarity_metrics_heatmap.png
└── screenshots/                 # AI usage disclosure screenshots
    └── (add your screenshots here)
```

## Implementation Details

### Preprocessing Pipeline
1. **Lowercasing**: Convert all text to lowercase
2. **Tokenization**: Split text into individual words
3. **Punctuation Removal**: Remove special characters
4. **Stopword Removal**: Remove common English stopwords
5. **Length Filtering**: Remove tokens shorter than 3 characters

### Indexing
- **Inverted Index**: Maps terms to document IDs and positions
- **Document Statistics**: Tracks document lengths and term frequencies
- **Vocabulary**: Complete set of unique terms across all documents

### Retrieval Strategies

#### 1. Boolean Retrieval
- Uses AND semantics across query terms
- Scores documents by proportion of query terms matched
- Fast but limited ranking capability

#### 2. TF-IDF
- Term Frequency: `tf = term_count / doc_length`
- Inverse Document Frequency: `idf = log((N + 1) / (df + 1)) + 1`
- Score: `score = sum(tf * idf)` for all query terms

#### 3. BM25
- Parameters: k1=1.5, b=0.75 (tunable)
- Formula: `BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))`
- Best performing general-purpose ranking function

#### 4. Cosine Similarity
- Creates TF-IDF vectors for query and documents
- Computes cosine of angle between vectors
- Score: `cosine = dot_product / (||query|| * ||doc||)`

## Evaluation Metrics

### Retrieval Effectiveness
- **Precision**: Proportion of retrieved documents that are relevant
- **Recall**: Proportion of relevant documents that are retrieved
- **F1-Score**: Harmonic mean of precision and recall

### Efficiency Metrics
- **Search Time**: Time taken to process query and return results
- **Index Size**: Memory footprint of inverted index
- **Query Processing Speed**: Queries per second

### Similarity Metrics
- **Jaccard**: `|Q ∩ D| / |Q ∪ D|`
- **Dice**: `2 * |Q ∩ D| / (|Q| + |D|)`
- **Overlap**: `|Q ∩ D| / min(|Q|, |D|)`

## Advanced Usage

### Using as a Library

```python
from main import IRSystem, Evaluator

# Initialize system
ir_system = IRSystem(remove_stopwords=True)

# Load documents
ir_system.load_documents('data/documents.csv')

# Build index
ir_system.build_index()

# Search with specific strategy
results = ir_system.search("machine learning", strategy='bm25', top_k=10)

# Compare strategies
comparison = ir_system.compare_strategies("deep learning", top_k=5)

# Evaluate
evaluator = Evaluator(ir_system)
queries = ["query1", "query2", "query3"]
evaluator.evaluate_queries(queries)
evaluator.visualize_results()
```

### Saving and Loading Index

```python
# Save index (faster subsequent runs)
ir_system.save_index('index/my_index.pkl')

# Load index
ir_system.load_index('index/my_index.pkl')
```

## Performance Optimization Tips

1. **For Large Datasets (>10K docs)**:
   - Consider using `remove_stopwords=True` (default)
   - Save the index after building to avoid rebuilding
   
2. **For Fast Searches**:
   - Use Boolean or BM25 (fastest)
   - Cosine similarity is slowest but most accurate

3. **For Better Quality**:
   - Use BM25 or Cosine similarity
   - Increase `top_k` for more comprehensive results

## Troubleshooting

### Issue: "No module named 'numpy'"
**Solution**: Make sure you've installed requirements: `pip install -r requirements.txt`

### Issue: Out of memory
**Solution**: Reduce dataset size or increase system RAM

### Issue: Very slow search times
**Solution**: 
- Ensure index is built before searching
- Try simpler strategies (Boolean, TF-IDF)
- Reduce vocabulary size by adjusting `min_word_length`

### Issue: No results found
**Solution**:
- Check if query terms exist in vocabulary
- Try broader queries
- Disable stopword removal for short queries

## Extending the System

### Adding New Retrieval Strategy

```python
class CustomRetrieval(RetrievalStrategy):
    def search(self, query: List[str], top_k: int = 10):
        # Your implementation here
        pass

# Add to strategies dict
ir_system.strategies['custom'] = CustomRetrieval(ir_system.index, ir_system.processed_documents)
```

### Adding Custom Preprocessing

```python
# Modify TextPreprocessor class
class CustomPreprocessor(TextPreprocessor):
    def preprocess(self, text: str) -> List[str]:
        # Add your custom preprocessing
        tokens = super().preprocess(text)
        # Additional processing
        return tokens
```

## Known Limitations

1. **No Query Expansion**: System uses exact term matching
2. **No Semantic Understanding**: Doesn't understand synonyms or context
3. **Memory Intensive**: Large datasets require significant RAM
4. **Single Language**: Optimized for English text only

## Future Improvements

- [ ] Add query expansion using WordNet
- [ ] Implement relevance feedback
- [ ] Add support for phrase queries
- [ ] Include semantic similarity using embeddings
- [ ] Add multi-threaded search for faster queries
- [ ] Support for multiple languages
- [ ] Web interface for easier interaction

## References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
2. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*.




---

**Disclosure**: This implementation was developed with assistance from AI tools (Claude). See technical report for detailed disclosure of AI usage.