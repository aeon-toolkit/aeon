"""A small sphinx extension to remove toctrees.

Original extension:

https://github.com/executablebooks/sphinx-remove-toctrees

This file was adapted by the developers of the MNE-LSL project, this is just
a copy for use in the aeon documentation.

https://github.com/mne-tools/mne-lsl
https://github.com/mne-tools/mne-lsl/blob/main/doc/_sphinxext/sphinx_remove_toctrees.py
"""

from pathlib import Path

from sphinx import addnodes


def remove_toctrees(app, env):
    """Remove toctrees from pages a user provides.

    This happens at the end of the build process, so even though the toctrees
    are removed, it won't raise sphinx warnings about unreferenced pages.
    """
    patterns = app.config.remove_from_toctrees
    if isinstance(patterns, str):
        patterns = [patterns]

    # figure out the list of patterns to remove from all toctrees
    to_remove = []
    for pattern in patterns:
        # inputs should either be a glob pattern or a direct path so just use glob
        srcdir = Path(env.srcdir)
        for matched in srcdir.glob(pattern):
            to_remove.append(
                str(matched.relative_to(srcdir).with_suffix("").as_posix())
            )

    # loop through all tocs and remove the ones that match our pattern
    for _, tocs in env.tocs.items():
        for toctree in tocs.traverse(addnodes.toctree):
            new_entries = []
            for entry in toctree.attributes.get("entries", []):
                if entry[1] not in to_remove:
                    new_entries.append(entry)
            # if there are no more entries just remove the toctree
            if len(new_entries) == 0:
                toctree.parent.remove(toctree)
            else:
                toctree.attributes["entries"] = new_entries


def setup(app):  # noqa: D103
    app.add_config_value("remove_from_toctrees", [], "html")
    app.connect("env-updated", remove_toctrees)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
