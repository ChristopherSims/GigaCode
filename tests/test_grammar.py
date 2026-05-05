import tree_sitter_python
print("tree_sitter_python imported successfully")
print("Has language attr:", hasattr(tree_sitter_python, "language"))

if hasattr(tree_sitter_python, "language"):
    lang = tree_sitter_python.language
    print("language object:", lang)
else:
    print("No language attribute found")
    print("Available attributes:", dir(tree_sitter_python))
