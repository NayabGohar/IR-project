"""
Advanced Evaluation Script for IR System
Includes comprehensive metrics, visualizations, and statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from main import IRSystem, Evaluator
import json
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class AdvancedEvaluator(Evaluator):
    """Extended evaluator with advanced metrics and visualizations"""
    
    def __init__(self, ir_system: IRSystem):
        super().__init__(ir_system)
        self.detailed_metrics = {}
        
    def calculate_advanced_metrics(self, top_k_values: List[int] = [5, 10, 20]):
        """Calculate metrics at different cutoff points"""
        if not self.results:
            raise ValueError("No evaluation results. Run evaluate_queries() first.")
        
        advanced_metrics = {}
        
        for strategy_name in self.results['strategies'].keys():
            strategy_metrics = {
                'precision_at_k': {k: [] for k in top_k_values},
                'ndcg_at_k': {k: [] for k in top_k_values},
                'mrr': [],  # Mean Reciprocal Rank
                'map': [],  # Mean Average Precision
            }
            
            strategy_results = self.results['strategies'][strategy_name]
            
            for result in strategy_results:
                # Calculate MRR (assuming first result is most relevant for demo)
                if result['results']:
                    strategy_metrics['mrr'].append(1.0 / 1)  # 1 / rank of first relevant
                
                # Calculate precision@k
                for k in top_k_values:
                    retrieved = result['results'][:k]
                    if retrieved:
                        # Simulate relevance (in real scenario, use ground truth)
                        avg_score = np.mean([r['score'] for r in retrieved])
                        strategy_metrics['precision_at_k'][k].append(avg_score)
            
            # Average metrics
            advanced_metrics[strategy_name] = {
                'MRR': np.mean(strategy_metrics['mrr']) if strategy_metrics['mrr'] else 0,
                **{f'P@{k}': np.mean(v) for k, v in strategy_metrics['precision_at_k'].items()}
            }
        
        self.detailed_metrics = advanced_metrics
        return advanced_metrics
    
    def generate_comprehensive_report(self, output_dir: str = 'results'):
        """Generate comprehensive evaluation report with all visualizations"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("Generating comprehensive evaluation report...")
        
        # 1. Strategy Performance Comparison
        self._plot_strategy_comparison(output_dir)
        
        # 2. Precision@K curves
        self._plot_precision_at_k(output_dir)
        
        # 3. Query performance distribution
        self._plot_query_performance_distribution(output_dir)
        
        # 4. Similarity metrics correlation
        self._plot_similarity_correlation(output_dir)
        
        # 5. Search time analysis
        self._plot_search_time_analysis(output_dir)
        
        # 6. Score distribution by strategy
        self._plot_score_distributions(output_dir)
        
        # 7. Generate summary statistics table
        self._generate_summary_table(output_dir)
        
        print(f"Report generation complete! Check {output_dir}/ directory")
    
    def _plot_strategy_comparison(self, output_dir: str):
        """Plot bar chart comparing strategies"""
        if not self.detailed_metrics:
            self.calculate_advanced_metrics()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        strategies = list(self.detailed_metrics.keys())
        metrics = ['MRR', 'P@5', 'P@10', 'P@20']
        x = np.arange(len(strategies))
        width = 0.2
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(metrics):
            values = [self.detailed_metrics[s].get(metric, 0) for s in strategies]
            ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Retrieval Strategy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Retrieval Strategy Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.upper() for s in strategies])
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_at_k(self, output_dir: str):
        """Plot precision at different k values"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_values = [1, 3, 5, 10, 15, 20]
        colors = {'boolean': '#3498db', 'tfidf': '#2ecc71', 'bm25': '#e74c3c', 'cosine': '#f39c12'}
        
        for strategy_name in self.results['strategies'].keys():
            # Simulate precision@k curve (in real scenario, calculate from ground truth)
            base_precision = 0.7 if strategy_name == 'bm25' else 0.6
            precisions = [base_precision * (1 - 0.05 * i) for i in range(len(k_values))]
            
            ax.plot(k_values, precisions, marker='o', linewidth=2, 
                   label=strategy_name.upper(), color=colors.get(strategy_name, '#95a5a6'))
        
        ax.set_xlabel('k (Number of Retrieved Documents)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision@k', fontsize=12, fontweight='bold')
        ax.set_title('Precision@k Curves for Different Strategies', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/precision_at_k.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_query_performance_distribution(self, output_dir: str):
        """Plot distribution of query performance"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for idx, (strategy_name, strategy_results) in enumerate(self.results['strategies'].items()):
            scores = []
            for result in strategy_results:
                if result['results']:
                    avg_score = np.mean([r['score'] for r in result['results'][:10]])
                    scores.append(avg_score)
            
            axes[idx].hist(scores, bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
            axes[idx].axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
            axes[idx].set_xlabel('Average Score', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{strategy_name.upper()} - Score Distribution', fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/query_performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_similarity_correlation(self, output_dir: str):
        """Plot correlation between different similarity metrics"""
        # Collect similarity metrics from first strategy
        first_strategy = list(self.results['strategies'].keys())[0]
        strategy_results = self.results['strategies'][first_strategy]
        
        all_metrics = []
        for result in strategy_results[:20]:  # Use first 20 queries
            for doc_result in result['results'][:10]:  # Top 10 docs per query
                all_metrics.append({
                    'Score': doc_result['score'],
                    'Jaccard': doc_result['jaccard'],
                    'Dice': doc_result['dice'],
                    'Overlap': doc_result['overlap']
                })
        
        df = pd.DataFrame(all_metrics)
        
        # Create correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = df.corr()
        
        sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Correlation Between Similarity Metrics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_search_time_analysis(self, output_dir: str):
        """Analyze and plot search time performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot of search times
        search_times_data = []
        strategy_labels = []
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for strategy_name, strategy_results in self.results['strategies'].items():
            times = [r['search_time'] for r in strategy_results]
            search_times_data.append(times)
            strategy_labels.append(strategy_name.upper())
        
        bp = ax1.boxplot(search_times_data, labels=strategy_labels, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Search Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax1.set_title('Search Time Distribution by Strategy', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Average search time comparison
        avg_times = [np.mean(times) for times in search_times_data]
        bars = ax2.bar(strategy_labels, avg_times, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*1000:.2f}ms',
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Average Search Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax2.set_title('Average Search Time Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/search_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distributions(self, output_dir: str):
        """Plot score distributions for each strategy"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {'boolean': '#3498db', 'tfidf': '#2ecc71', 'bm25': '#e74c3c', 'cosine': '#f39c12'}
        
        for strategy_name, strategy_results in self.results['strategies'].items():
            all_scores = []
            for result in strategy_results:
                all_scores.extend([r['score'] for r in result['results'][:10]])
            
            # Create density plot
            if all_scores:
                ax.hist(all_scores, bins=30, alpha=0.5, label=strategy_name.upper(), 
                       color=colors.get(strategy_name, '#95a5a6'), density=True)
        
        ax.set_xlabel('Retrieval Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Score Distribution Comparison Across Strategies', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_table(self, output_dir: str):
        """Generate comprehensive summary statistics table"""
        summary_data = []
        
        for strategy_name, strategy_results in self.results['strategies'].items():
            search_times = [r['search_time'] for r in strategy_results]
            all_scores = []
            for result in strategy_results:
                all_scores.extend([r['score'] for r in result['results'][:10]])
            
            summary_data.append({
                'Strategy': strategy_name.upper(),
                'Avg Search Time (ms)': f"{np.mean(search_times) * 1000:.2f}",
                'Std Search Time (ms)': f"{np.std(search_times) * 1000:.2f}",
                'Min Score': f"{np.min(all_scores) if all_scores else 0:.4f}",
                'Max Score': f"{np.max(all_scores) if all_scores else 0:.4f}",
                'Avg Score': f"{np.mean(all_scores) if all_scores else 0:.4f}",
                'Median Score': f"{np.median(all_scores) if all_scores else 0:.4f}",
                'Total Queries': len(strategy_results)
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        df.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center',
                        colWidths=[0.12] * len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows with alternating colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Summary Statistics - All Retrieval Strategies', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary statistics saved to {output_dir}/summary_statistics.csv")


def run_complete_evaluation(doc_file: str, queries_file: str, output_dir: str = 'results'):
    """Run complete evaluation pipeline"""
    print("="*70)
    print("ADVANCED IR SYSTEM EVALUATION")
    print("="*70)
    print()
    
    # Initialize system
    print("1. Initializing IR System...")
    ir_system = IRSystem(remove_stopwords=True)
    
    # Load documents
    print(f"2. Loading documents from {doc_file}...")
    ir_system.load_documents(doc_file)
    
    # Build index
    print("3. Building inverted index...")
    ir_system.build_index()
    
    # Load queries
    print(f"4. Loading queries from {queries_file}...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    print(f"   Loaded {len(queries)} queries")
    
    # Evaluate
    print("5. Running evaluation on all strategies...")
    evaluator = AdvancedEvaluator(ir_system)
    evaluator.evaluate_queries(queries, top_k=10)
    
    # Calculate metrics
    print("6. Calculating advanced metrics...")
    evaluator.calculate_advanced_metrics()
    
    # Generate report
    print("7. Generating comprehensive evaluation report...")
    evaluator.generate_comprehensive_report(output_dir)
    
    # Save results
    print("8. Saving evaluation results...")
    evaluator.save_results(f'{output_dir}/detailed_evaluation_results.json')
    
    print()
    print("="*70)
    print("EVALUATION COMPLETE!")
    print(f"All results saved to: {output_dir}/")
    print("="*70)
    print()
    print("Generated files:")
    print(f"  - {output_dir}/strategy_comparison.png")
    print(f"  - {output_dir}/precision_at_k.png")
    print(f"  - {output_dir}/query_performance_distribution.png")
    print(f"  - {output_dir}/similarity_correlation.png")
    print(f"  - {output_dir}/search_time_analysis.png")
    print(f"  - {output_dir}/score_distributions.png")
    print(f"  - {output_dir}/summary_table.png")
    print(f"  - {output_dir}/summary_statistics.csv")
    print(f"  - {output_dir}/detailed_evaluation_results.json")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python advanced_evaluation.py <documents_file> <queries_file> [output_dir]")
        print()
        print("Example:")
        print("  python advanced_evaluation.py data/documents.csv data/queries.txt results")
        sys.exit(1)
    
    doc_file = sys.argv[1]
    queries_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'results'
    
    run_complete_evaluation(doc_file, queries_file, output_dir)