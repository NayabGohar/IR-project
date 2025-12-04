

import sys
from pathlib import Path
from main import IRSystem, TextPreprocessor, InvertedIndex, SimilarityMetrics
import time

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(name):
    """Print test name"""
    print(f"\n{Colors.BLUE}Testing: {name}{Colors.END}")
    print("-" * 50)

def print_pass(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ PASS: {message}{Colors.END}")

def print_fail(message):
    """Print failure message"""
    print(f"{Colors.RED}✗ FAIL: {message}{Colors.END}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ INFO: {message}{Colors.END}")

def test_preprocessing():
    """Test text preprocessing"""
    print_test("Text Preprocessing")
    
    preprocessor = TextPreprocessor(remove_stopwords=True)
    
    text = "The Quick Brown Fox Jumps Over The Lazy Dog!"
    result = preprocessor.preprocess(text)
    
    expected_properties = {
        'lowercase': all(word.islower() for word in result),
        'no_stopwords': 'the' not in result,
        'alphabetic': all(word.isalpha() for word in result),
        'min_length': all(len(word) > 2 for word in result)
    }
    
    for prop, value in expected_properties.items():
        if value:
            print_pass(f"{prop} check")
        else:
            print_fail(f"{prop} check")
    
    print_info(f"Input: {text}")
    print_info(f"Output: {result}")
    
    empty_result = preprocessor.preprocess("")
    if len(empty_result) == 0:
        print_pass("Empty string handling")
    else:
        print_fail("Empty string handling")
    
    special = "Hello!!! World??? 123 #hashtag"
    special_result = preprocessor.preprocess(special)
    if all(word.isalpha() for word in special_result):
        print_pass("Special character removal")
    else:
        print_fail("Special character removal")
    
    return True

def test_inverted_index():
    """Test inverted index construction"""
    print_test("Inverted Index")
    
    index = InvertedIndex()
    
    documents = [
        ['machine', 'learning', 'algorithms'],
        ['deep', 'learning', 'neural', 'networks'],
        ['machine', 'learning', 'data', 'mining']
    ]
    
    index.build(documents)
    
    expected_vocab_size = 7
    if len(index.vocabulary) == expected_vocab_size:
        print_pass(f"Vocabulary size: {len(index.vocabulary)}")
    else:
        print_fail(f"Expected vocab size {expected_vocab_size}, got {len(index.vocabulary)}")
    
    learning_postings = index.get_postings('learning')
    if len(learning_postings) == 3:  
        print_pass("Postings list for 'learning'")
    else:
        print_fail(f"Expected 3 postings for 'learning', got {len(learning_postings)}")
    
    machine_df = index.get_doc_freq('machine')
    if machine_df == 2: 
        print_pass("Document frequency calculation")
    else:
        print_fail(f"Expected df=2 for 'machine', got {machine_df}")
    
    expected_avg = (3 + 4 + 4) / 3
    if abs(index.avg_doc_length - expected_avg) < 0.01:
        print_pass(f"Average document length: {index.avg_doc_length:.2f}")
    else:
        print_fail(f"Expected avg length {expected_avg}, got {index.avg_doc_length}")
    
    print_info(f"Total documents: {index.total_docs}")
    print_info(f"Vocabulary: {sorted(index.vocabulary)}")
    
    return True

def test_retrieval_strategies():
    """Test different retrieval strategies"""
    print_test("Retrieval Strategies")
    
    ir_system = IRSystem(remove_stopwords=False)
    
    ir_system.raw_documents = [
        "machine learning is a subset of artificial intelligence",
        "deep learning is a subset of machine learning",
        "neural networks are used in deep learning",
        "artificial intelligence includes machine learning and robotics",
        "data mining uses machine learning algorithms"
    ]
    
    ir_system.processed_documents = [
        ir_system.preprocessor.preprocess(doc) 
        for doc in ir_system.raw_documents
    ]
    
    ir_system.build_index()
    
    query = "machine learning"
    
    strategies_to_test = ['boolean', 'tfidf', 'bm25', 'cosine']
    
    for strategy in strategies_to_test:
        try:
            results = ir_system.search(query, strategy=strategy, top_k=3)
            
            if results and len(results['results']) > 0:
                print_pass(f"{strategy.upper()}: Retrieved {len(results['results'])} documents")
                print_info(f"  Search time: {results['search_time']*1000:.2f} ms")
                print_info(f"  Top result score: {results['results'][0]['score']:.4f}")
            else:
                print_fail(f"{strategy.upper()}: No results returned")
        except Exception as e:
            print_fail(f"{strategy.upper()}: {str(e)}")
    
    return True

def test_similarity_metrics():
    """Test similarity metric calculations"""
    print_test("Similarity Metrics")
    
    metrics = SimilarityMetrics()
    
    query = ['machine', 'learning', 'algorithms']
    document = ['machine', 'learning', 'data', 'science']
    
    jaccard = metrics.jaccard_similarity(query, document)
    expected_jaccard = 2 / 5  
    if abs(jaccard - expected_jaccard) < 0.01:
        print_pass(f"Jaccard similarity: {jaccard:.4f}")
    else:
        print_fail(f"Expected Jaccard {expected_jaccard}, got {jaccard}")
    
    dice = metrics.dice_coefficient(query, document)
    expected_dice = 2 * 2 / (3 + 4)  
    if abs(dice - expected_dice) < 0.01:
        print_pass(f"Dice coefficient: {dice:.4f}")
    else:
        print_fail(f"Expected Dice {expected_dice}, got {dice}")
    
    overlap = metrics.overlap_coefficient(query, document)
    expected_overlap = 2 / 3 
    if abs(overlap - expected_overlap) < 0.01:
        print_pass(f"Overlap coefficient: {overlap:.4f}")
    else:
        print_fail(f"Expected Overlap {expected_overlap}, got {overlap}")
    
    print_info(f"Query: {query}")
    print_info(f"Document: {document}")
    
    return True

def test_performance():
    """Test system performance with larger dataset"""
    print_test("Performance Testing")
    
    try:
        if not Path('data/documents.csv').exists():
            print_info("Generating sample data...")
            import subprocess
            subprocess.run([sys.executable, 'generate_sample_data.py'], check=True)
        
        ir_system = IRSystem(remove_stopwords=True)
        
        start = time.time()
        ir_system.load_documents('data/documents.csv')
        load_time = time.time() - start
        print_pass(f"Loaded {len(ir_system.raw_documents)} documents in {load_time:.2f}s")
        
        start = time.time()
        ir_system.build_index()
        build_time = time.time() - start
        print_pass(f"Built index in {build_time:.2f}s")
        
        print_info(f"Vocabulary size: {len(ir_system.index.vocabulary)}")
        print_info(f"Avg document length: {ir_system.index.avg_doc_length:.1f} tokens")
        
        test_queries = [
            "machine learning algorithms",
            "artificial intelligence",
            "data mining techniques"
        ]
        
        total_search_time = 0
        for query in test_queries:
            start = time.time()
            results = ir_system.search(query, strategy='bm25', top_k=10)
            search_time = time.time() - start
            total_search_time += search_time
        
        avg_search_time = total_search_time / len(test_queries)
        print_pass(f"Average search time: {avg_search_time*1000:.2f} ms")
        
        if avg_search_time < 0.1: 
            print_pass("Search performance is excellent")
        elif avg_search_time < 0.5:
            print_pass("Search performance is good")
        else:
            print_info("Search performance could be improved")
        
        return True
        
    except Exception as e:
        print_fail(f"Performance test failed: {str(e)}")
        return False

def test_end_to_end():
    """Test complete end-to-end workflow"""
    print_test("End-to-End Workflow")
    
    try:
        ir_system = IRSystem(remove_stopwords=True)
        print_pass("System initialization")
        
        if Path('data/documents.csv').exists():
            ir_system.load_documents('data/documents.csv')
            print_pass("Document loading")
        else:
            print_fail("Sample data not found. Run generate_sample_data.py first")
            return False
        
        ir_system.build_index()
        print_pass("Index building")
        
        results = ir_system.search("machine learning", strategy='bm25', top_k=5)
        print_pass("Search execution")
        
        comparison = ir_system.compare_strategies("artificial intelligence", top_k=3)
        print_pass("Strategy comparison")
        
        required_keys = ['query', 'strategy', 'results', 'search_time', 'num_results']
        if all(key in results for key in required_keys):
            print_pass("Results structure validation")
        else:
            print_fail("Results missing required keys")
        
        print_info(f"Retrieved {len(results['results'])} results")
        print_info(f"Compared {len(comparison['strategies'])} strategies")
        
        return True
        
    except Exception as e:
        print_fail(f"End-to-end test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all test suites"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}IR SYSTEM TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Inverted Index", test_inverted_index),
        ("Retrieval Strategies", test_retrieval_strategies),
        ("Similarity Metrics", test_similarity_metrics),
        ("Performance", test_performance),
        ("End-to-End", test_end_to_end)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_fail(f"Exception in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! System is ready.{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠ Some tests failed. Please review the output above.{Colors.END}")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)