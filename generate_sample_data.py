"""
Generate sample dataset for testing IR system
Creates realistic documents and queries for demonstration
"""

import pandas as pd
import random
from pathlib import Path

DOCUMENT_TEMPLATES = [
    "Machine learning algorithms are used for pattern recognition and data analysis in various applications.",
    "Deep learning neural networks have revolutionized computer vision and natural language processing tasks.",
    "Information retrieval systems help users find relevant documents from large collections of text data.",
    "Artificial intelligence techniques include search algorithms, knowledge representation, and reasoning methods.",
    "Database management systems organize and store structured data for efficient querying and retrieval.",
    "Cloud computing platforms provide scalable infrastructure for deploying and managing applications.",
    "Cybersecurity measures protect computer systems and networks from unauthorized access and attacks.",
    "Data mining techniques extract patterns and insights from large datasets using statistical methods.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer networks connect devices and systems to enable communication and resource sharing.",
    
    "Climate change affects global temperatures, weather patterns, and sea levels around the world.",
    "Quantum physics explores the behavior of matter and energy at the atomic and subatomic scales.",
    "Genetic engineering techniques modify DNA sequences to create organisms with desired traits.",
    "Renewable energy sources like solar and wind power provide sustainable alternatives to fossil fuels.",
    "Neuroscience research investigates the structure and function of the nervous system and brain.",
    "Astronomy studies celestial objects, space phenomena, and the universe's origin and evolution.",
    "Chemistry examines the properties, composition, and reactions of matter at the molecular level.",
    "Ecology investigates relationships between organisms and their environment in natural ecosystems.",
    "Biotechnology applies biological processes and organisms to develop products and technologies.",
    "Physics explores fundamental laws governing matter, energy, space, and time in the universe.",
    
    "Marketing strategies focus on understanding customer needs and promoting products effectively.",
    "Financial analysis evaluates company performance using metrics like revenue, profit, and cash flow.",
    "Supply chain management optimizes the flow of goods from manufacturers to end consumers.",
    "Human resources departments handle recruitment, training, and employee relations in organizations.",
    "Project management methodologies help teams plan, execute, and complete work within constraints.",
    "Business intelligence tools analyze data to support strategic decision-making and planning.",
    "E-commerce platforms enable online buying and selling of products and services globally.",
    "Corporate strategy defines long-term goals and competitive positioning in the marketplace.",
    "Quality management systems ensure products and services meet customer expectations consistently.",
    "Entrepreneurship involves creating and managing new business ventures to solve market problems.",
    
    "Medical diagnosis uses symptoms, tests, and imaging to identify diseases and health conditions.",
    "Pharmaceutical research develops new drugs and treatments for various diseases and disorders.",
    "Public health initiatives promote wellness and prevent disease in communities and populations.",
    "Telemedicine technology enables remote healthcare delivery through video consultations and monitoring.",
    "Clinical trials test the safety and efficacy of new medical treatments and interventions.",
    "Healthcare informatics applies information technology to improve patient care and outcomes.",
    "Preventive medicine focuses on disease prevention through screening, vaccination, and lifestyle changes.",
    "Emergency medicine provides immediate treatment for acute illnesses and traumatic injuries.",
    "Mental health services address psychological, emotional, and behavioral disorders and conditions.",
    "Nutrition science studies the role of diet and nutrients in health, disease, and wellbeing.",
    
    "Online learning platforms provide digital courses and educational resources accessible worldwide.",
    "Educational psychology examines how people learn and develop cognitive skills over time.",
    "STEM education emphasizes science, technology, engineering, and mathematics in curricula.",
    "Distance education programs allow students to complete degrees remotely through virtual classrooms.",
    "Curriculum development involves designing effective learning experiences and educational materials.",
    "Educational technology integrates digital tools to enhance teaching and learning processes.",
    "Assessment methods measure student learning outcomes and educational program effectiveness.",
    "Special education provides tailored instruction for students with diverse learning needs.",
    "Higher education institutions offer undergraduate and graduate degree programs in various fields.",
    "Literacy programs help individuals develop reading, writing, and communication skills.",
]

def generate_documents(num_docs: int = 100) -> pd.DataFrame:
    """Generate sample documents with variations"""
    documents = []
    
    for i in range(num_docs):
        num_sentences = random.randint(2, 5)
        doc_sentences = random.sample(DOCUMENT_TEMPLATES, num_sentences)
        
        doc_text = ' '.join(doc_sentences)
        
        if random.random() > 0.7:
            words = doc_text.split()
            if len(words) > 10:
                insert_pos = random.randint(5, len(words) - 5)
                additional_words = ["Moreover", "Furthermore", "Additionally", "However", "Therefore"]
                words.insert(insert_pos, random.choice(additional_words).lower() + ',')
                doc_text = ' '.join(words)
        
        documents.append({
            'doc_id': i,
            'text': doc_text
        })
    
    return pd.DataFrame(documents)

def generate_queries() -> list:
    """Generate sample search queries"""
    queries = [
        "machine learning algorithms",
        "deep learning neural networks",
        "information retrieval systems",
        "artificial intelligence techniques",
        "database management",
        "cloud computing platforms",
        "cybersecurity protection",
        "data mining patterns",
        "natural language processing",
        "computer network systems",
        
        "climate change effects",
        "quantum physics behavior",
        "genetic engineering DNA",
        "renewable energy solar wind",
        "neuroscience brain research",
        "astronomy celestial objects",
        "chemistry molecular reactions",
        "ecology ecosystem relationships",
        "biotechnology applications",
        "physics fundamental laws",
        
        "marketing customer strategies",
        "financial analysis metrics",
        "supply chain optimization",
        "human resources management",
        "project management methodologies",
        "business intelligence analytics",
        "e-commerce online platforms",
        "corporate business strategy",
        "quality management systems",
        "entrepreneurship ventures",
        
        "medical diagnosis symptoms",
        "pharmaceutical drug research",
        "public health prevention",
        "telemedicine remote healthcare",
        "clinical trials testing",
        "healthcare informatics technology",
        "preventive medicine screening",
        "emergency medical treatment",
        "mental health services",
        "nutrition diet health",
        
        "online learning platforms",
        "educational psychology learning",
        "STEM education programs",
        "distance education virtual",
        "curriculum development design",
        "educational technology tools",
        "assessment student learning",
        "special education instruction",
        "higher education degrees",
        "literacy reading writing",
        
        "how machine learning improves data analysis",
        "impact of climate change on global ecosystems",
        "role of artificial intelligence in healthcare",
        "benefits of cloud computing for businesses",
        "renewable energy vs fossil fuels comparison",
    ]
    
    return queries

def main():
    """Generate and save sample dataset"""
    print("Generating sample dataset for IR system...")
    print()
    
    Path('data').mkdir(exist_ok=True)
    
    num_docs = 200  
    print(f"Generating {num_docs} sample documents...")
    documents_df = generate_documents(num_docs)
    
    documents_df.to_csv('data/documents.csv', index=False)
    print(f"✓ Saved {num_docs} documents to data/documents.csv")
    
    queries = generate_queries()
    print(f"Generating {len(queries)} sample queries...")
    
    with open('data/queries.txt', 'w') as f:
        f.write('\n'.join(queries))
    print(f"✓ Saved {len(queries)} queries to data/queries.txt")
    
    print()
    print("="*60)
    print("Sample dataset generation complete!")
    print("="*60)
    print()
    print("Files created:")
    print("  - data/documents.csv (document collection)")
    print("  - data/queries.txt (search queries)")
    print()
    print("You can now run the IR system with:")
    print("  python main.py")
    print()
    print("Or run complete evaluation with:")
    print("  python advanced_evaluation.py data/documents.csv data/queries.txt")
    print()

if __name__ == "__main__":
    main()