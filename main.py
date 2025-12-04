

import numpy as np
import pandas as pd
import json
import time
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import re
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class TextPreprocessor:
    """Handles all text preprocessing operations"""
    
    def __init__(self, remove_stopwords=True, min_word_length=2):
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.stopwords = self._load_stopwords()
        
    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords"""
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
        }
        return stopwords
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords
        """
        text = text.lower()
        
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        tokens = text.split()
        
        tokens = [t for t in tokens if len(t) > self.min_word_length]
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens


class InvertedIndex:
    """Build and manage inverted index"""
    
    def __init__(self):
        self.index = defaultdict(list)  
        self.doc_lengths = {} 
        self.doc_freq = defaultdict(int) 
        self.total_docs = 0
        self.avg_doc_length = 0
        self.vocabulary = set()
        
    def build(self, documents: List[List[str]]):
        """Build inverted index from preprocessed documents"""
        self.total_docs = len(documents)
        total_length = 0
        
        for doc_id, tokens in enumerate(documents):
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            seen_terms = set()
            
            for position, term in enumerate(tokens):
                self.vocabulary.add(term)
                self.index[term].append((doc_id, position))
                
                if term not in seen_terms:
                    self.doc_freq[term] += 1
                    seen_terms.add(term)
        
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
    
    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        """Get posting list for a term"""
        return self.index.get(term, [])
    
    def get_doc_freq(self, term: str) -> int:
        """Get document frequency for a term"""
        return self.doc_freq.get(term, 0)
    
    def save(self, filepath: str):
        """Save index to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class RetrievalStrategy:
    """Base class for retrieval strategies"""
    
    def __init__(self, index: InvertedIndex, documents: List[List[str]]):
        self.index = index
        self.documents = documents
    
    def search(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """Search and return top-k (doc_id, score) pairs"""
        raise NotImplementedError


class BooleanRetrieval(RetrievalStrategy):
    """Boolean retrieval with AND semantics"""
    
    def search(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        if not query:
            return []
        
        first_term = query[0]
        doc_ids = set([doc_id for doc_id, _ in self.index.get_postings(first_term)])
        
        for term in query[1:]:
            term_docs = set([doc_id for doc_id, _ in self.index.get_postings(term)])
            doc_ids = doc_ids.intersection(term_docs)
        
        results = []
        for doc_id in doc_ids:
            matches = sum(1 for term in query if any(d == doc_id for d, _ in self.index.get_postings(term)))
            score = matches / len(query)
            results.append((doc_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class TFIDFRetrieval(RetrievalStrategy):
    """TF-IDF based retrieval"""
    
    def _calculate_tf(self, term: str, doc_id: int) -> float:
        """Calculate term frequency"""
        term_count = sum(1 for d, _ in self.index.get_postings(term) if d == doc_id)
        doc_length = self.index.doc_lengths[doc_id]
        return term_count / doc_length if doc_length > 0 else 0
    
    def _calculate_idf(self, term: str) -> float:
        """Calculate inverse document frequency"""
        df = self.index.get_doc_freq(term)
        return math.log((self.index.total_docs + 1) / (df + 1)) + 1
    
    def search(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        scores = defaultdict(float)
        
        for term in query:
            idf = self._calculate_idf(term)
            postings = self.index.get_postings(term)
            
            for doc_id, _ in postings:
                tf = self._calculate_tf(term, doc_id)
                scores[doc_id] += tf * idf
        
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]


class BM25Retrieval(RetrievalStrategy):
    """BM25 ranking function"""
    
    def __init__(self, index: InvertedIndex, documents: List[List[str]], k1=1.5, b=0.75):
        super().__init__(index, documents)
        self.k1 = k1
        self.b = b
    
    def search(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        scores = defaultdict(float)
        
        for term in query:
            df = self.index.get_doc_freq(term)
            if df == 0:
                continue
            
            idf = math.log((self.index.total_docs - df + 0.5) / (df + 0.5) + 1)
            postings = self.index.get_postings(term)
            
            for doc_id, _ in postings:
                tf = sum(1 for d, _ in postings if d == doc_id)
                doc_length = self.index.doc_lengths[doc_id]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.index.avg_doc_length))
                
                scores[doc_id] += idf * (numerator / denominator)
        
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]


class CosineRetrieval(RetrievalStrategy):
    """Cosine similarity with TF-IDF vectors"""
    
    def _calculate_tfidf_vector(self, terms: List[str], doc_id: int = None) -> Dict[str, float]:
        """Calculate TF-IDF vector for document or query"""
        vector = {}
        term_counts = Counter(terms)
        
        for term, count in term_counts.items():
            tf = count / len(terms) if len(terms) > 0 else 0
            df = self.index.get_doc_freq(term)
            idf = math.log((self.index.total_docs + 1) / (df + 1)) + 1
            vector[term] = tf * idf
        
        return vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
        
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)
    
    def search(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        query_vector = self._calculate_tfidf_vector(query)
        scores = {}
        
        candidate_docs = set()
        for term in query:
            for doc_id, _ in self.index.get_postings(term):
                candidate_docs.add(doc_id)
        
        for doc_id in candidate_docs:
            doc_vector = self._calculate_tfidf_vector(self.documents[doc_id], doc_id)
            scores[doc_id] = self._cosine_similarity(query_vector, doc_vector)
        
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]


class SimilarityMetrics:
    """Calculate various similarity metrics"""
    
    @staticmethod
    def jaccard_similarity(query: List[str], document: List[str]) -> float:
        """Calculate Jaccard coefficient"""
        query_set = set(query)
        doc_set = set(document)
        
        intersection = query_set.intersection(doc_set)
        union = query_set.union(doc_set)
        
        return len(intersection) / len(union) if len(union) > 0 else 0
    
    @staticmethod
    def dice_coefficient(query: List[str], document: List[str]) -> float:
        """Calculate Dice coefficient"""
        query_set = set(query)
        doc_set = set(document)
        
        intersection = query_set.intersection(doc_set)
        
        return (2 * len(intersection)) / (len(query_set) + len(doc_set)) if (len(query_set) + len(doc_set)) > 0 else 0
    
    @staticmethod
    def overlap_coefficient(query: List[str], document: List[str]) -> float:
        """Calculate overlap coefficient"""
        query_set = set(query)
        doc_set = set(document)
        
        intersection = query_set.intersection(doc_set)
        
        return len(intersection) / min(len(query_set), len(doc_set)) if min(len(query_set), len(doc_set)) > 0 else 0


class IRSystem:
    """Main Information Retrieval System"""
    
    def __init__(self, remove_stopwords=True):
        self.preprocessor = TextPreprocessor(remove_stopwords=remove_stopwords)
        self.index = None
        self.raw_documents = []
        self.processed_documents = []
        self.strategies = {}
        self.metrics = SimilarityMetrics()
        
    def load_documents(self, filepath: str):
        """Load documents from CSV or text file"""
        print(f"Loading documents from {filepath}...")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'document' in col.lower()]
            if text_columns:
                self.raw_documents = df[text_columns[0]].astype(str).tolist()
            else:
                self.raw_documents = df.iloc[:, 0].astype(str).tolist()
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.raw_documents = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.raw_documents)} documents")
        
        print("Preprocessing documents...")
        self.processed_documents = [self.preprocessor.preprocess(doc) for doc in self.raw_documents]
        print("Preprocessing complete")
        
    def build_index(self):
        """Build inverted index"""
        print("Building inverted index...")
        start_time = time.time()
        
        self.index = InvertedIndex()
        self.index.build(self.processed_documents)
        
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.2f} seconds")
        print(f"Vocabulary size: {len(self.index.vocabulary)}")
        print(f"Average document length: {self.index.avg_doc_length:.2f} tokens")
        
        self.strategies = {
            'boolean': BooleanRetrieval(self.index, self.processed_documents),
            'tfidf': TFIDFRetrieval(self.index, self.processed_documents),
            'bm25': BM25Retrieval(self.index, self.processed_documents),
            'cosine': CosineRetrieval(self.index, self.processed_documents)
        }
        
    def search(self, query: str, strategy: str = 'bm25', top_k: int = 10) -> List[Dict]:
        """Search using specified strategy"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
        
        processed_query = self.preprocessor.preprocess(query)
        
        start_time = time.time()
        results = self.strategies[strategy].search(processed_query, top_k)
        search_time = time.time() - start_time
        
        detailed_results = []
        for doc_id, score in results:
            result = {
                'doc_id': doc_id,
                'score': score,
                'text': self.raw_documents[doc_id][:200] + '...',
                'full_text': self.raw_documents[doc_id],
                'jaccard': self.metrics.jaccard_similarity(processed_query, self.processed_documents[doc_id]),
                'dice': self.metrics.dice_coefficient(processed_query, self.processed_documents[doc_id]),
                'overlap': self.metrics.overlap_coefficient(processed_query, self.processed_documents[doc_id])
            }
            detailed_results.append(result)
        
        return {
            'query': query,
            'strategy': strategy,
            'results': detailed_results,
            'search_time': search_time,
            'num_results': len(results)
        }
    
    def compare_strategies(self, query: str, top_k: int = 10) -> Dict:
        """Compare all retrieval strategies for a query"""
        comparison = {
            'query': query,
            'strategies': {}
        }
        
        for strategy_name in self.strategies.keys():
            results = self.search(query, strategy_name, top_k)
            comparison['strategies'][strategy_name] = results
        
        return comparison
    
    def save_index(self, filepath: str):
        """Save index to disk"""
        self.index.save(filepath)
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load index from disk"""
        self.index = InvertedIndex.load(filepath)
        print(f"Index loaded from {filepath}")


class Evaluator:
    """Evaluate IR system performance"""
    
    def __init__(self, ir_system: IRSystem):
        self.ir_system = ir_system
        self.results = []
        
    def evaluate_queries(self, queries: List[str], strategies: List[str] = None, top_k: int = 10):
        """Evaluate multiple queries across strategies"""
        if strategies is None:
            strategies = list(self.ir_system.strategies.keys())
        
        print(f"Evaluating {len(queries)} queries with {len(strategies)} strategies...")
        
        evaluation_results = {
            'strategies': {s: [] for s in strategies},
            'queries': queries,
            'timestamp': datetime.now().isoformat()
        }
        
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            for strategy in strategies:
                result = self.ir_system.search(query, strategy, top_k)
                evaluation_results['strategies'][strategy].append(result)
        
        self.results = evaluation_results
        return evaluation_results
    
    def calculate_metrics(self, relevance_judgments: Dict[str, Set[int]] = None):
        """Calculate precision, recall, F1 for each strategy"""
        if not self.results:
            raise ValueError("No evaluation results. Run evaluate_queries() first.")
        
        metrics = {}
        
        for strategy_name, strategy_results in self.results['strategies'].items():
            precisions = []
            recalls = []
            f1_scores = []
            search_times = []
            
            for i, result in enumerate(strategy_results):
                search_times.append(result['search_time'])
                
                if relevance_judgments and self.results['queries'][i] in relevance_judgments:
                    relevant_docs = relevance_judgments[self.results['queries'][i]]
                    retrieved_docs = set([r['doc_id'] for r in result['results']])
                    
                    tp = len(relevant_docs.intersection(retrieved_docs))
                    precision = tp / len(retrieved_docs) if retrieved_docs else 0
                    recall = tp / len(relevant_docs) if relevant_docs else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
            
            metrics[strategy_name] = {
                'avg_precision': np.mean(precisions) if precisions else None,
                'avg_recall': np.mean(recalls) if recalls else None,
                'avg_f1': np.mean(f1_scores) if f1_scores else None,
                'avg_search_time': np.mean(search_times),
                'total_queries': len(strategy_results)
            }
        
        return metrics
    
    def visualize_results(self, output_dir: str = 'results'):
        """Generate visualization plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        if not self.results:
            raise ValueError("No evaluation results to visualize")
        
        sns.set_style("whitegrid")
        
        plt.figure(figsize=(10, 6))
        strategy_names = list(self.results['strategies'].keys())
        avg_times = [np.mean([r['search_time'] for r in self.results['strategies'][s]]) 
                     for s in strategy_names]
        
        plt.bar(strategy_names, avg_times, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        plt.xlabel('Retrieval Strategy', fontsize=12)
        plt.ylabel('Average Search Time (seconds)', fontsize=12)
        plt.title('Search Time Comparison Across Strategies', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/search_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.results['strategies']:
            first_strategy = list(self.results['strategies'].keys())[0]
            first_result = self.results['strategies'][first_strategy][0]['results'][:5]
            
            metrics_data = []
            doc_ids = []
            for r in first_result:
                doc_ids.append(f"Doc {r['doc_id']}")
                metrics_data.append([r['score'], r['jaccard'], r['dice'], r['overlap']])
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=['Score', 'Jaccard', 'Dice', 'Overlap'],
                       yticklabels=doc_ids)
            plt.title(f'Similarity Metrics Heatmap\nQuery: {self.results["queries"][0][:50]}...', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/similarity_metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")


def main():
    """Main execution function"""
    print("="*60)
    print("Information Retrieval System - CS516")
    print("="*60)
    print()
    
    ir_system = IRSystem(remove_stopwords=True)
    
    doc_file = input("Enter path to documents file (CSV or TXT): ").strip()
    ir_system.load_documents(doc_file)
    
    ir_system.build_index()
    
    print("\n" + "="*60)
    print("System ready! You can now search.")
    print("Available strategies: boolean, tfidf, bm25, cosine")
    print("Type 'compare' to compare all strategies")
    print("Type 'evaluate' to run batch evaluation")
    print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        query = input("\nEnter query: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if query.lower() == 'compare':
            query = input("Enter query to compare: ").strip()
            comparison = ir_system.compare_strategies(query, top_k=5)
            
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            for strategy_name, results in comparison['strategies'].items():
                print(f"\n{strategy_name.upper()} Results (Time: {results['search_time']:.4f}s):")
                print("-" * 60)
                for i, result in enumerate(results['results'][:5], 1):
                    print(f"{i}. [Doc {result['doc_id']}] Score: {result['score']:.4f}")
                    print(f"   Jaccard: {result['jaccard']:.3f} | Dice: {result['dice']:.3f} | Overlap: {result['overlap']:.3f}")
                    print(f"   {result['text']}")
                    print()
        
        elif query.lower() == 'evaluate':
            queries_file = input("Enter path to queries file: ").strip()
            with open(queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            evaluator = Evaluator(ir_system)
            evaluator.evaluate_queries(queries[:10], top_k=10)  
            evaluator.visualize_results()
            evaluator.save_results('results/evaluation_results.json')
            
            print("\nEvaluation complete! Check 'results/' directory for outputs.")
        
        else:
            strategy = input("Enter strategy (default: bm25): ").strip() or 'bm25'
            results = ir_system.search(query, strategy, top_k=10)
            
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"Strategy: {strategy.upper()}")
            print(f"Search time: {results['search_time']:.4f} seconds")
            print(f"{'='*60}\n")
            
            for i, result in enumerate(results['results'], 1):
                print(f"{i}. [Doc {result['doc_id']}] Score: {result['score']:.4f}")
                print(f"   Jaccard: {result['jaccard']:.3f} | Dice: {result['dice']:.3f} | Overlap: {result['overlap']:.3f}")
                print(f"   {result['text']}")
                print()


if __name__ == "__main__":
    main()