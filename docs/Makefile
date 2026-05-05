# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PAPER         ?=
BUILDDIR      ?= _build

# The 'make' target is invoked from the docs directory and goes to BUILDDIR.
# The BUILDDIR is declared outside BUILDDIR so you can build the documentation
# in BUILDDIR by running "make".

.PHONY: help Makefile

# Put it first so that "make without argument" is default.
help:
	@$(SPHINXBUILD) -M help "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man texinfo gettext linkcheck doctest coverage

html:
	@$(SPHINXBUILD) -M html "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
	rm -rf $(BUILDDIR)/*

dirhtml:
	@$(SPHINXBUILD) -M dirhtml "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	@$(SPHINXBUILD) -M singlehtml "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/singlehtml."

pickle:
	@$(SPHINXBUILD) -M pickle "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Pickle files are in $(BUILDDIR)/pickle."

json:
	@$(SPHINXBUILD) -M json "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "JSON files are in $(BUILDDIR)/json."

htmlhelp:
	@$(SPHINXBUILD) -M htmlhelp "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "HTML help files are in $(BUILDDIR)/htmlhelp."

qthelp:
	@$(SPHINXBUILD) -M qthelp "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Qt help files are in $(BUILDDIR)/qthelp."

devhelp:
	@$(SPHINXBUILD) -M devhelp "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Devhelp files are in $(BUILDDIR)/devhelp."

epub:
	@$(SPHINXBUILD) -M epub "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Epub file is in $(BUILDDIR)/epub."

latex:
	@$(SPHINXBUILD) -M latex "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "LaTeX files are in $(BUILDDIR)/latex."

latexpdf:
	@$(SPHINXBUILD) -M latexpdf "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "PDF file is in $(BUILDDIR)/latex/gigacode.pdf."

text:
	@$(SPHINXBUILD) -M text "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Text files are in $(BUILDDIR)/text."

man:
	@$(SPHINXBUILD) -M man "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Man files are in $(BUILDDIR)/man."

texinfo:
	@$(SPHINXBUILD) -M texinfo "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Texinfo files are in $(BUILDDIR)/texinfo."

gettext:
	@$(SPHINXBUILD) -M gettext "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "I18n .pot files are in $(BUILDDIR)/gettext."

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Link check complete; see $(BUILDDIR)/linkcheck/index.html for report."

doctest:
	@$(SPHINXBUILD) -M doctest "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Doctest complete; see $(BUILDDIR)/doctest/index.html for report."

coverage:
	@$(SPHINXBUILD) -M coverage "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Coverage complete; see $(BUILDDIR)/coverage/python/.  for report."
