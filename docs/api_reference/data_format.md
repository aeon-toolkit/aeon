# `.ts` File Format

This document formalises the `.ts` file format used by `aeon`.
Encoded in `utf-8`, a `.t` file stores a time-series dataset and its corresponding
metadata (specified via string identifiers). String identifiers refer to strings
beginning with `@` in the file.

`.ts` files contains information blocks in the following order:

1. A description block.
      Contains any number of continuous lines starting with `#`.
      Each `#` is followed by an arbitrary (`utf-8`) sequence of symbols.
      The `.ts` specification does not prescribe any content for the description
      block, but it is common to include a description of the dataset contained in the
      file i.e. a full data dictionary, citations, etc.
2. A metadata block.
      Contains continuous lines starting with `@`.
      Each `@` is directly followed a string identifier without whitespace
      (`@<identifier>`), followed by an appropriate value for the identifier where
      the value depends on type of identifier. There is no strict order of occurrence
      for all string identifiers, except `@data` which must be at the end of this
      block. The number of lines in this block depends on certain properties of the
      dataset (e.g: if the dataset is multivariate, an additional line is required to
      specify number of channels)
3. A dataset block.
      Contains a multiple collections of float values that represent the dataset. There
      are `n` cases each its own time series, delimited by new lines. The values for a
      series are expressed in a comma `,` seperated list and the index of each value is
      relative to its position in said list (0, 1, ..., `m`). An instance may contain 1
      to `d` channels, where each channel for a case is delimited using a colon `:`.
      In case timestamps are present, each value in a series is enclosed within
      round brackets i.e. `(YYYY-MM-DD HH:mm:ss,<value>)`.
      The response variable is at the end of each case and is seperated via a colon.

Here is an extract from an examlple `.ts`  file that shows portions of all three
blocks:

```
#The data was generated from students wearing a smart watch.
#Consists of four classes, which are walking, resting, running and badminton.
...
@problemName BasicMotions
@timeStamps false
@missing false
...
@data
-0.740653,-0.740653,10.208449,2.867009:-0.194301,-0.194301,-0.249618,0.516079:Standing
-0.247409,-0.247409,-0.77129,-0.576154:-0.368484,-0.020851,-0.020851,-0.465607:Walking
...
```

For  example files, see the files in `aeon/datasets/data/` or the
[tsml Zenodo community](https://zenodo.org/communities/tsml/records?q=&l=list&p=1&s=10&sort=newest).

## Metadata

`aeon`'s core loader/writer functions relies on the existence of metadata to
correctly load data into memory.
It is also helpful to provide information about the dataset to a different user not
familiar with the dataset.

The format of individual string identifier is: `@<identifier> [value]`,
except for `@data` where there is no trailing information. There should be a new
line for each metadata entry.

```{list-table}
:widths: 10 25 15 30 20
:header-rows: 1

* - Identifier
  - Description
  - Value
  - Additional Comments
  - Example
* - `@problemname`
  - The name of the dataset.
  - any `string`
  - Value cannot be space separated
  - `BasicMotions`
* - `@timestamps`
  - Whether timestamps are present.
  - `true`, `false`
  - `true` / `false` only
  - `false`
* - `@missing`
  - Whether there are missing values.
  - `true`, `false`
  - `true` / `false` only
  - `false`
* - `@univariate`
  - Whether there is only one dimension for the time series.
  - `true`, `false`
  - `true` / `false` only
  - `false`
* - `@dimension`
  - The number of channels.
  - integer > 0
  - Only present when `@univariate false`.
  - 6
* - `@equallength`
  - Whether all cases are equal length.
  - `true`, `false`
  - `true` / `false` only
  - `true`
* - `@serieslength`
  - Number of timepoints in each case.
  - integer > 0
  - Only present if `@equallength true`.
  - 100
* - `@targetlabel`
  - Whether there is a target label.
  - `true`, `false`
  - Exclusive to regression data; `true` / `false` only
  - `true`
* - `@classlabel`
  - Whether class labels are present.
  - `false` / `true` `<string-1> <string-2> ..`
  - Exclusive to classification data; when `true`, also contains space-seperated int/strings as labels.
  - `true Standing Running Walking Badminton`
* - `@data`
  - Marks the beginning of data.
  - \-
  - The data begins from the next line.
  - \-
```
