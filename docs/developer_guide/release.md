# Releases

This page is to help core developers in releasing a new version of `aeon`.

Core developers making a release should have write access to the repository.

`aeon` aims for a release every one or two months currently, based on the amount of
new content available for the release.

## Preparing the release version

The release process is as follows, on high-level:

1. __Ensure deprecation actions are carried out.__
  Deprecation actions for a version should be marked by "version number" annotated
  comments in the code. E.g., for the release 0.10.0, search for the string 0.10.0 in
  the code and carry out described deprecation actions. Collect list of deprecation
  actions, as they should go in the release notes.

2. __Create a "release" pull request.__
  Create a branch from main and PR named after the release version. This should make
  changes to the version numbers (root `__init__.py`, `README.md` and `pyproject.toml`)
  and have complete release notes in the changelog webpage.

3. __Merge the "release" pull request.__
  This PR should ideally be the final PR made before the release with the exception of
  any necessary troubleshooting PRs. The PR and release notes should optimally be
  reviewed by the core developers, then merged once tests pass.

4. __Create the GitHub release.__
  This release should create a new tag following the syntax v[MAJOR].[MINOR].[PATCH],
  e.g., the string `v0.10.0` for version 0.10.0. The release name should similarly be
  `aeon v0.10.0`.  The GitHub release notes should contain only "hightlights",
  "new contributors" and "all contributors" sections, and otherwise link to the release
  notes in the changelog, following the pattern of current GitHub release notes. The
  full GitHub commit log between releases can also be included.

## ``pypi`` release and release validation

Creation of the GitHub release trigger the `pypi` release workflow.

5. __Wait for the ``pypi`` release CI/CD to finish.__
  If tests fail due to sporadic unrelated failure, restart. If tests fail genuinely,
  something went wrong in the above steps, investigate, fix, and repeat.

6. __Release workflow completion tasks.__
  Once the release workflow has passed, check `aeon` version on `pypi`, this should be
  the new version. A validatory installation of `aeon` in a new Python environment
  should be carried out according to the installation instructions. If the installation
  does not succeed or wheels have not been uploaded, action to diagnose and remedy must
  be taken.

## Release notes

Release notes can be generated using the `build_tools/changelog.py` script, and should
be placed at the top of the relevant [MINOR] versions webpage. e.g.,
`docs/chanelogs/v0.10.md` for version 0.10.0. Generally, release notes should follow the
general pattern of previous release notes, with sections:

- Highlights
- Dependency changes, if any
- Deprecations/removals, if any.
- Auto generated PR and contributions sections.
