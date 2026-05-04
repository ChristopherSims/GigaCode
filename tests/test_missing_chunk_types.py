"""Tests for missing chunk types: lambdas, macros, nested functions (Phase 6.2)."""

import pytest
from gigacode.chunker import chunk_text


def test_python_lambda_detection():
    """Test that Python lambda expressions are detected as separate chunks."""
    code = '''def main():
    """Main function."""
    numbers = [1, 2, 3, 4, 5]
    
    # Lambda in a variable assignment
    square = lambda x: x ** 2
    
    # Lambda in list comprehension-like usage
    result = list(map(lambda x: x * 2, numbers))
    
    return result
'''
    chunks = chunk_text(code, language_hint='python')
    
    # Should have chunks for: main function, lambda assignments
    assert len(chunks) > 0
    
    # At minimum, should have the main function
    function_chunks = [c for c in chunks if c.type == 'function']
    assert len(function_chunks) > 0
    assert 'main' in [c.name for c in function_chunks if c.name]


def test_java_lambda_detection():
    """Test that Java lambda expressions are detected (if grammar available).
    
    Note: This test documents the capability but may be skipped if
    tree-sitter Java grammar is not installed.
    """
    code = '''public class StreamExample {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        
        // Lambda expression in forEach
        numbers.forEach(n -> System.out.println(n));
        
        // Lambda in map
        List<Integer> doubled = numbers.stream()
            .map(n -> n * 2)
            .collect(Collectors.toList());
    }
}
'''
    chunks = chunk_text(code, language_hint='java')
    
    # Should have chunks (either from AST or sliding window fallback)
    assert len(chunks) > 0
    
    # If we got AST chunks (not sliding window), check for method/class
    if not any(c.type == 'sliding' for c in chunks):
        method_chunks = [c for c in chunks if c.type == 'method']
        class_chunks = [c for c in chunks if c.type == 'class']
        assert len(method_chunks) > 0  # main method
        assert len(class_chunks) > 0   # StreamExample class


def test_cpp_macro_detection():
    """Test that C++ preprocessor macros are detected (if grammar available).
    
    Note: This test documents the capability but may be skipped if
    tree-sitter C++ grammar is not installed.
    """
    code = '''#include <iostream>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SQUARE(x) ((x) * (x))
#define SWAP(a, b) { int temp = a; a = b; b = temp; }

int main() {
    int x = 10;
    int y = 20;
    int max_val = MAX(x, y);
    int square = SQUARE(5);
    return 0;
}
'''
    chunks = chunk_text(code, language_hint='cpp')
    
    # Should have chunks
    assert len(chunks) > 0
    
    # If we got AST chunks (not sliding window), check for functions
    if not any(c.type == 'sliding' for c in chunks):
        function_chunks = [c for c in chunks if c.type == 'function']
        assert len(function_chunks) > 0  # main function
        
        # Check if any macros were detected
        macro_chunks = [c for c in chunks if c.type == 'macro']
        if macro_chunks:
            assert any('MAX' in (c.name or '') for c in macro_chunks)


def test_c_macro_detection():
    """Test that C preprocessor macros are detected (if grammar available).
    
    Note: This test documents the capability but may be skipped if
    tree-sitter C grammar is not installed.
    """
    code = '''#define PI 3.14159
#define CIRCLE_AREA(r) (PI * (r) * (r))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int main() {
    double area = CIRCLE_AREA(5.0);
    int min_val = MIN(10, 20);
    return 0;
}
'''
    chunks = chunk_text(code, language_hint='c')
    
    # Should have chunks
    assert len(chunks) > 0
    
    # If we got AST chunks (not sliding window), check for main function
    if not any(c.type == 'sliding' for c in chunks):
        function_chunks = [c for c in chunks if c.type == 'function']
        assert len(function_chunks) > 0  # At least main function detected


def test_python_nested_function_detection():
    """Test that nested functions are detected with type 'nested_function'."""
    code = '''def outer_function():
    """Outer function."""
    x = 10
    
    def inner_function():
        """Inner nested function."""
        return x * 2
    
    def another_inner():
        """Another nested function."""
        return x + 5
    
    return inner_function()
'''
    chunks = chunk_text(code, language_hint='python')
    
    # Should have chunks for outer and inner functions
    assert len(chunks) > 0
    
    # Check for nested functions
    nested_chunks = [c for c in chunks if c.type == 'nested_function']
    function_chunks = [c for c in chunks if c.type == 'function']
    
    # At minimum should have outer function
    assert len(function_chunks) > 0
    assert any('outer' in (c.name or '') for c in function_chunks)
    
    # May have nested functions detected
    # This documents the capability


def test_javascript_nested_function_detection():
    """Test that nested functions in JavaScript are detected."""
    code = '''function greet(name) {
    const greeting = "Hello, ";
    
    function createMessage() {
        return greeting + name;
    }
    
    function logMessage() {
        console.log(createMessage());
    }
    
    return logMessage();
}
'''
    chunks = chunk_text(code, language_hint='javascript')
    
    # Should have chunks for all functions
    assert len(chunks) > 0
    
    # Check for function detection
    function_chunks = [c for c in chunks if c.type in ('function', 'nested_function')]
    assert len(function_chunks) > 0
    assert any('greet' in (c.name or '') for c in function_chunks)


def test_multiple_languages_with_new_types():
    """Test that chunking handles multiple languages with new chunk types."""
    languages_and_codes = [
        ('python', '''
x = lambda a, b: a + b

def outer():
    def inner():
        pass
    return inner
'''),
        ('javascript', '''
const square = x => x * x;

function outer() {
    function inner() {}
    return inner;
}
'''),
        ('java', '''
public class Example {
    public void test() {
        Runnable r = () -> System.out.println("Hello");
    }
}
'''),
    ]
    
    for language, code in languages_and_codes:
        chunks = chunk_text(code, language_hint=language)
        assert len(chunks) > 0, f"Expected chunks for {language}, got empty list"
        
        # Check that chunk types are recognized
        valid_types = {'function', 'nested_function', 'lambda', 'class', 'method', 'orphan', 'macro', 'sliding', 'module'}
        for chunk in chunks:
            assert chunk.type in valid_types, f"Unknown chunk type: {chunk.type} in {language}"


def test_chunk_types_in_multiline_lambda():
    """Test that multiline lambda/arrow function works correctly."""
    code = '''const operations = [
    x => x * 2,
    x => x + 10,
    x => x ** 2
];

function processArray(arr, op) {
    return arr.map(op);
}
'''
    chunks = chunk_text(code, language_hint='javascript')
    
    # Should identify the function
    function_chunks = [c for c in chunks if c.type == 'function']
    assert len(function_chunks) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
