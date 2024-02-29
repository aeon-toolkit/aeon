# Releases

This page is to help core developers in releasing a new version of `aeon`.

Core developers making a release should have write access to the repository.

`aeon` aims for a release every one or two months currently, based on the amount of
new content available for the release.

## Preparing the release version

The release process is as follows, on high-level:

1. **Ensure deprecation actions are carried out.**
  Deprecation actions for a version should be marked by "version number" annotated
  comments in the code. E.g., for the release 0.10.0, search for the string 0.10.0 in
  the code and carry out described deprecation actions. Collect list of deprecation
  actions, as they should go in the release notes.

1. **Create a "release" pull request.**
  Create a branch from main and PR named after the release version. This should make
  changes to the version numbers (root `__init__.py`, `README.md` and `pyproject.toml`)
  and have complete release notes in the changelog webpage.

3. **Merge the "release" pull request.**
  This PR should ideally be the final PR made before the release with the exception of
  any necessary troubleshooting PRs. The PR and release notes should optimally be
  reviewed by the core developers, then merged once tests pass.

4. **Create the GitHub release.**
  This release should create a new tag following the syntax v[MAJOR].[MINOR].[PATCH],
  e.g., the string `v0.10.0` for version 0.10.0. The release name should similarly be
  `aeon v0.10.0`.  The GitHub release notes should contain only "hightlights",
  "new contributors" and "all contributors" sections, and otherwise link to the release
  notes in the changelog, following the pattern of current GitHub release notes. The
  full GitHub commit log between releases can also be included.

## `pypi` release and release validation

Creation of the GitHub release trigger the `pypi` release workflow.

5. **Wait for the ``pypi`` release CI/CD to finish.**
  If tests fail due to sporadic unrelated failure, restart. If tests fail genuinely,
  something went wrong in the above steps, investigate, fix, and repeat.

6. **Release workflow completion tasks.**
  Once the release workflow has passed, check `aeon` version on `pypi`, this should be
  the new version. A validatory installation of `aeon` in a new Python environment
  should be carried out according to the installation instructions. If the installation
  does not succeed or wheels have not been uploaded, action to diagnose and remedy must
  be taken.

## `conda-forge` release and release validation

7. **Merge the ``conda-forge`` release PR.**
  After some time a PR will be automatically created in the [aeon conda-forge feedstock](https://github.com/conda-forge/aeon-feedstock).
  Follow the instructions in the PR to merge it, making sure to update any dependencies
  that have changed and dependency version bounds.

## Release notes

Generally, [release notes](https://www.aeon-toolkit.org/en/latest/changelog.html) should follow the general pattern of previous release notes, with sections:

- Highlights
- Dependency changes, if any
- Deprecations/removals, if any.
- Auto generated PR and contributions sections.
