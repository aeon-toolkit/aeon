"""Aeon Mini-Gallery Sphinx Extension.

Automatically injects example notebook galleries into API reference pages
by scanning notebooks for class usage via AST parsing.
"""

import ast
import re
from pathlib import Path

import nbformat
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging, relative_uri

logger = logging.getLogger(__name__)


class AeonUsageVisitor(ast.NodeVisitor):
    """AST visitor that detects Aeon object usage in notebook code cells.

    Tracks imports, resolves aliases, and identifies which fully-qualified
    Aeon class names are actually instantiated or called in the code.
    """

    def __init__(self):
        self.symbol_table = {}
        self.module_aliases = {}
        self.used_objects = set()

    def visit_Import(self, node):
        """Handle 'import aeon.x' statements and track module aliases."""
        for alias in node.names:
            if alias.name.startswith("aeon"):
                local = alias.asname or alias.name
                self.module_aliases[local] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle 'from aeon.x import Y' statements."""
        if node.module and node.module.startswith("aeon"):
            for alias in node.names:
                local = alias.asname or alias.name
                fq_name = f"{node.module}.{alias.name}"
                self.symbol_table[local] = fq_name
        self.generic_visit(node)

    def visit_Call(self, node):
        """Detect function/method calls and record used Aeon objects."""
        fq_name = self._resolve(node.func)
        if fq_name:
            self.used_objects.add(fq_name)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Resolve chained attribute access to fully-qualified names."""
        fq_name = self._resolve(node)
        if fq_name:
            self.used_objects.add(fq_name)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Resolve simple name references using the symbol table."""
        if node.id in self.symbol_table:
            self.used_objects.add(self.symbol_table[node.id])
        self.generic_visit(node)

    def _resolve(self, node):
        """Resolve an AST node to its fully-qualified Aeon object name."""
        if isinstance(node, ast.Name):
            return self.symbol_table.get(node.id)

        if isinstance(node, ast.Attribute):
            chain = []

            while isinstance(node, ast.Attribute):
                chain.append(node.attr)
                node = node.value

            if isinstance(node, ast.Name):
                chain.append(node.id)

            chain.reverse()

            root = chain[0]

            if root in self.symbol_table:
                base = self.symbol_table[root]
                suffix = ".".join(chain[1:])
                return f"{base}.{suffix}" if suffix else base

            if root in self.module_aliases:
                base = self.module_aliases[root]
                suffix = ".".join(chain[1:])
                return f"{base}.{suffix}" if suffix else base

        return None


def scan_notebooks(app):
    """Scan notebooks and build fully-qualified object → notebook mapping."""
    logger.info("Scanning notebooks for aeon object usage...")

    example_dir = (Path(app.srcdir) / "../examples").resolve()

    mapping = {}

    for nb_path in example_dir.rglob("*.ipynb"):
        try:
            nb = nbformat.read(nb_path, as_version=4)
        except Exception:
            continue

        visitor = AeonUsageVisitor()

        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            try:
                visitor.visit(ast.parse(cell.source))
            except Exception:
                continue

        rel_path = nb_path.relative_to(example_dir).with_suffix("")

        for fq_name in visitor.used_objects:
            mapping.setdefault(fq_name, []).append(str(rel_path))

    app.env.aeon_example_map = mapping

    logger.info(f"Object-example mapping built: {len(mapping)} objects found.")


def build_thumbnail_map(app):
    """Parse examples.md and build notebook → thumbnail mapping."""
    logger.info("Building thumbnail map from examples.md")

    examples_md = Path(app.srcdir) / "examples.md"

    if not examples_md.exists():
        logger.warning("examples.md not found.")
        app.env.aeon_thumbnail_map = {}
        return

    content = examples_md.read_text()

    pattern = re.compile(
        r":img-top:\s*(?P<img>.+?)\s+.*?" r":link:\s*/examples/(?P<nb>.+?)\.ipynb",
        re.DOTALL,
    )

    thumbnail_map = {}

    for match in pattern.finditer(content):
        img = match.group("img").strip()
        nb = match.group("nb").strip()

        thumbnail_map[nb] = img

    app.env.aeon_thumbnail_map = thumbnail_map

    logger.info(f"Thumbnail mapping built: {len(thumbnail_map)} entries.")


class AeonMiniGalleryDirective(Directive):
    """Render mini-gallery of related example notebooks."""

    required_arguments = 1

    def run(self):
        """Execute directive and render gallery cards."""
        env = self.state.document.settings.env
        app = env.app
        builder = app.builder

        fq_name = self.arguments[0]

        example_map = getattr(env, "aeon_example_map", {})
        thumbnail_map = getattr(env, "aeon_thumbnail_map", {})

        examples = sorted(set(example_map.get(fq_name, [])))

        if not examples:
            return []

        page_uri = builder.get_target_uri(env.docname)

        cards = []

        for ex in examples:
            thumb_path = thumbnail_map.get(ex)

            if not thumb_path:
                continue

            example_doc = f"examples/{ex}"
            example_uri = builder.get_relative_uri(
                env.docname,
                example_doc,
            )

            thumb_uri = relative_uri(
                page_uri,
                f"_images/{Path(thumb_path).name}",
            )

            card_title = Path(ex).name.replace("_", " ").capitalize()

            cards.append(f"""
                <a class="aeon-mini-card" href="{example_uri}">
                    <div class="aeon-mini-image">
                        <img
                            src="{thumb_uri}"
                            loading="lazy"
                            alt="{card_title}"
                        >
                    </div>
                    <div class="aeon-mini-title">
                        {card_title}
                    </div>
                </a>
                """)

        # Avoiding empty gallery sections
        if not cards:
            return []

        section = nodes.section()
        section["ids"].append("gallery-examples")

        title = nodes.title(text="Gallery Examples")
        section += title

        html_blocks = ['<div class="aeon-mini-gallery">']
        html_blocks.extend(cards)
        html_blocks.append("</div>")

        section += nodes.raw(
            "",
            "\n".join(html_blocks),
            format="html",
        )

        return [section]


def setup(app):
    """Register extension."""
    app.connect("builder-inited", scan_notebooks)
    app.connect("builder-inited", build_thumbnail_map)

    app.add_directive(
        "aeon-mini-gallery",
        AeonMiniGalleryDirective,
    )

    app.add_css_file("css/aeon_gallery.css")

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
