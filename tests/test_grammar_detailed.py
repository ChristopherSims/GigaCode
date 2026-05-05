import tree_sitter_python
from tree_sitter import Language

print("tree_sitter_python.language:", tree_sitter_python.language)
print("Is callable?", callable(tree_sitter_python.language))

try:
    # Try different ways to access the language
    lang = tree_sitter_python.language
    print("Type:", type(lang))
    
    # Try calling it
    if callable(lang):
        result = lang()
        print("Called lang():", result)
        grammar = Language(result)
    else:
        print("Attempting to use directly...")
        grammar = Language(lang)
    
    print("Grammar created successfully:", grammar)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
