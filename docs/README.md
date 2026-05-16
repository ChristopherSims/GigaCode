# GigaCode Documentation

Complete documentation for the GigaCode semantic code search tool.

## Quick Start

View the documentation online at: [docs link]

Or build locally:

```bash
cd docs
pip install sphinx sphinx-rtd-theme
make html
open _build/html/index.html  # macOS
# or
start _build/html/index.html  # Windows
# or
xdg-open _build/html/index.html  # Linux
```

## Documentation Structure

- **[Quick Start](quick_start.rst)** - Get started in 5 minutes
- **[Installation](installation.rst)** - Installation guide for all platforms
- **Tutorials** - Step-by-step guides
  - [Basic Embedding](tutorials/basic_embed.rst)
  - [Search Workflows](tutorials/search_workflows.rst)
  - [Edit Workflows](tutorials/edit_workflows.rst)
- **API Reference** - Complete API documentation
  - [Tools](api/tools.rst)
  - [Managers](api/managers.rst)
  - [Services](api/services.rst)
  - [Phase 3 Optimizations](api/incremental_indexer.rst)
- **[Architecture](architecture.rst)** - Design and architecture
- **[Performance Tuning](performance_tuning.rst)** - Optimization strategies
- **[Contributing](contributing.rst)** - Contribution guidelines

## Building the Docs

**Build HTML:**

```bash
make html
```

**Build PDF:**

```bash
make latexpdf
```

**Clean build:**

```bash
make clean
```

**Preview changes:**

```bash
make html && python -m http.server --directory _build/html
```

Then open http://localhost:8000

## Documentation Standards

### Writing Style

- Clear, concise language
- Active voice preferred
- Code examples for all features
- Performance notes where relevant

### Code Examples

All code examples should:

- Be runnable (where possible)
- Include output or results
- Show both simple and advanced usage
- Reference relevant API docs

### docstrings

Python docstrings should:

- Include description
- Document all parameters
- Document return values
- Include usage example
- Mention exceptions

Example:

```python
def semantic_search(
    self,
    buffer_id: str,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Search semantically for code.
    
    Args:
        buffer_id: Buffer identifier
        query: Search query
        top_k: Number of results
        
    Returns:
        Dictionary with search results
        
    Example:
        >>> results = tool.semantic_search("my_project", "find database operations")
        >>> print(f"Found {results['total']} matches")
    """
```

### Sections

Organize with appropriate RST headers:

- `#` with `=` - Document title
- `~` - Section heading
- `^` - Subsection
- `"` - Sub-subsection

## Phase 3 Documentation

All Phase 3 documentation is in:

- [Incremental Indexer](api/incremental_indexer.rst) - 5-50x faster commits
- [Semantic Cache](api/semantic_cache.rst) - 50% faster searches
- [FAISS Optimizer](api/faiss_optimizer.rst) - Smart index selection

See [Performance Tuning](performance_tuning.rst) for optimization tips.

## Building Locally for Development

### Prerequisites

```bash
pip install sphinx sphinx-rtd-theme
```

### Build and serve

```bash
cd docs
make html
cd _build/html
python -m http.server 8000
```

Visit http://localhost:8000

### Auto-rebuild on changes

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Visit http://localhost:8000 (auto-refreshes on save)

## Contributing Documentation

1. Edit `.rst` files in `docs/`
2. Build with `make html`
3. Verify rendering: `open _build/html/index.html`
4. Submit PR

## Documentation TODO

- [ ] API auto-generation from docstrings
- [ ] Video tutorials
- [ ] Interactive examples
- [ ] API versioning
- [ ] Translation support

## Docs Deployed

This documentation is deployed to:

- ReadTheDocs: [link]
- GitHub Pages: [link]
- Project website: [link]

## License

Documentation is licensed under CC BY 4.0

## Questions?

- GitHub Issues: [link]
- GitHub Discussions: [link]
- Email: [email]
