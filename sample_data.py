

import pandas as pd
import re
from collections import Counter
import random

class NewsQueryGenerator:
    """Generate diverse test queries from news dataset"""
    
    def __init__(self, df, text_column='content', title_column='title'):
        self.df = df
        self.text_column = text_column
        self.title_column = title_column
        
        self.stopwords = set([
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
            'it', 'from', 'be', 'are', 'was', 'were', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'said', 'says', 'more'
        ])
    
    def extract_keywords(self, text, n=5):
        """Extract top N keywords from text"""
        if pd.isna(text):
            return []
        
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        
        words = [w for w in words if w not in self.stopwords and len(w) > 3]
        
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(n)]
    
    def generate_title_queries(self, num_queries=20):
        """Generate queries from article titles"""
        queries = []
        
        sampled = self.df.sample(min(num_queries * 2, len(self.df)))
        
        for _, row in sampled.iterrows():
            title = str(row.get(self.title_column, ''))
            if not title or len(title) < 10:
                continue
            
            title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
            words = title.split()
            
            keywords = [w.lower() for w in words if w.lower() not in self.stopwords and len(w) > 3]
            
            if len(keywords) >= 2:
                query_length = random.randint(2, min(4, len(keywords)))
                query = ' '.join(keywords[:query_length])
                queries.append(query)
            
            if len(queries) >= num_queries:
                break
        
        return queries
    
    def generate_content_queries(self, num_queries=20):
        """Generate queries from article content"""
        queries = []
        
        sampled = self.df.sample(min(num_queries * 2, len(self.df)))
        
        for _, row in sampled.iterrows():
            content = str(row.get(self.text_column, ''))
            if not content or len(content) < 50:
                continue
            
            keywords = self.extract_keywords(content, n=10)
            
            if len(keywords) >= 2:
                query_length = random.randint(2, min(3, len(keywords)))
                selected = random.sample(keywords, query_length)
                query = ' '.join(selected)
                queries.append(query)
            
            if len(queries) >= num_queries:
                break
        
        return queries
    
    def generate_category_queries(self, category_column='category', num_per_category=3):
        """Generate queries based on categories"""
        queries = []
        
        if category_column not in self.df.columns:
            return queries
        
        categories = self.df[category_column].dropna().unique()
        
        for category in categories[:10]: 
            cat_articles = self.df[self.df[category_column] == category].sample(min(5, len(self.df)))
            
            for _, row in cat_articles.iterrows():
                title = str(row.get(self.title_column, ''))
                keywords = self.extract_keywords(title, n=5)
                
                if len(keywords) >= 2:
                    query = ' '.join(keywords[:2])
                    queries.append(query)
                    
                    if len([q for q in queries if category.lower() in q.lower()]) >= num_per_category:
                        break
        
        return queries
    
    def generate_phrase_queries(self, num_queries=15):
        """Generate 2-3 word phrase queries"""
        queries = []
        
        sampled = self.df.sample(min(num_queries * 2, len(self.df)))
        
        for _, row in sampled.iterrows():
            content = str(row.get(self.text_column, ''))[:500]  
            
            if not content or len(content) < 50:
                continue
            
            sentences = re.split(r'[.!?]', content)
            
            for sentence in sentences[:3]:
                words = sentence.lower().split()
                keywords = [w for w in words if w not in self.stopwords and len(w) > 3]
                
                if len(keywords) >= 2:
                    if len(keywords) >= 3:
                        start = random.randint(0, len(keywords) - 3)
                        query = ' '.join(keywords[start:start+3])
                    else:
                        query = ' '.join(keywords[:2])
                    
                    queries.append(query)
                    break
            
            if len(queries) >= num_queries:
                break
        
        return queries
    
    def generate_all_queries(self, 
                            num_title=15, 
                            num_content=15, 
                            num_phrase=10,
                            num_category=10,
                            output_file='queries.txt'):
        """Generate diverse set of queries and save to file"""
        
        print("Generating queries from news dataset...")
        print(f"Dataset size: {len(self.df)} articles")
        
        all_queries = []
        
        print("\n1. Title-based queries...")
        title_queries = self.generate_title_queries(num_title)
        all_queries.extend(title_queries)
        print(f"   Generated {len(title_queries)} queries")
        
        print("\n2. Content-based queries...")
        content_queries = self.generate_content_queries(num_content)
        all_queries.extend(content_queries)
        print(f"   Generated {len(content_queries)} queries")
        
        print("\n3. Phrase queries...")
        phrase_queries = self.generate_phrase_queries(num_phrase)
        all_queries.extend(phrase_queries)
        print(f"   Generated {len(phrase_queries)} queries")
        
        print("\n4. Category-based queries...")
        category_queries = self.generate_category_queries(num_per_category=3)
        all_queries.extend(category_queries)
        print(f"   Generated {len(category_queries)} queries")
        
        all_queries = list(set(all_queries))
        
        all_queries.sort(key=len)
        
        with open(output_file, 'w', encoding='latin1') as f:
            for query in all_queries:
                f.write(query + '\n')
        
        print(f"\n✓ Total {len(all_queries)} unique queries saved to {output_file}")
        
        print("\nSample queries:")
        for i, query in enumerate(all_queries[:10], 1):
            print(f"  {i}. {query}")
        
        return all_queries


# Main

def main():
    """Main execution"""
    print("=" * 70)
    print("NEWS QUERY GENERATOR")
    print("=" * 70)
    print()
    
    csv_path = input("Enter path to your news CSV file: ").strip()
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} articles")
        print(f"✓ Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return
    
    print()
    
    print("Available columns:", list(df.columns))
    text_col = input("Enter text/content column name: ").strip()
    title_col = input("Enter title column name (press Enter if none): ").strip()
    
    if not title_col:
        title_col = 'title'  

    output_file = input("Enter output file name (default: queries.txt): ").strip()
    if not output_file:
        output_file = 'queries.txt'
