# Glossary of Common Terms

The glossary below defines common terms and API elements used throughout `aeon`.

```{glossary}
:sorted:

Time series data
Time series
Series
    Data with multiple individual {term}`variable` measurements with acoompanying
    {term}`timepoints` which are ordered over time or have an index indicating the
    position of an observation in the sequence of values.

Timepoint
Timepoints
    The point in time that an observation is made for a {term}`time series`. A time
    point may represent an exact point in time (a timestamp), a timeperiod (e.g.
    minutes, hours or days), or simply an index indicating the position of an
    observation in the sequence of values.

Variable
Variables
    Refers to some measurement of interest. Variables may be singular values
    (e.g. time-invariant measurements like a patient's place of birth) or a sequence
    of multiple values as a {term}`time series`.

    For time series data, multiple variables may be refered to as {term}`channels`.

Target variable
Target variables
    The {term}`variable`(s) that are being predicted in a learning task using
    {term}`Independant variables`, past {term}`timepoints` of the variable itself, or
    both. Also referred toas the dependent or endogenous variable(s).

Independant variable
Independant variables
    The {term}`variable`(s) that are used to predict the {term}`target variable`(s)
    in a learning task. Also referred to as exogenous variables Commonly also known as
    features and attributes in traditional machine learning settings.

Channel
Channels
    A channel is a singular {term}`time series` in a data set which contains multiple
    time series {term}`variables`. A dataset with multiple channels is
    {term}`multivariate`.

Time series machine learning
    A general term for using machine learning algorithms to learn predictive models
    from {term}`time series` data. `aeon` is a library for time series machine learning
    algorithms.

Forecasting
    A {term}`Time series machine learning` task focused on prediction future values of
    a {term}`time series`.

Time series classification
    A learning task focused on using the patterns across {term}`instances` between the
    {term}`time series` and a categorical {term}`target variable`.

Time series regression
    A learning task focused on using learning patterns from multiple {term}`time series`
    and a continuous {term}`target variable`. There are two related but distinct
    learning tasks that fall under this category: {term}`time series forecasting
    regression` and {term}`time series extrinsic regression`.

Time series forecasting regression
    This learning relates to {term}`forecasting` {term}`reduced <reduction>` to
    regression through a sliding window. This is the more familiar type of regression
    in literature.

Time series extrinsic regression
    A learning task focused on using the patterns across {term}`instances` between the
    {term}`time series` and a continuous {term}`target variable`. The `aeon`
    `regression` module is focused on this type of regression.

Time series clustering
    A learning task focused on discovering groups consisting of {term}`instances` with
    similar {term}`time series`.

Time series annotation
    A learning task focused on labeling the {term}`variables` of a {term}`time series`.
    This includes the related tasks of outlier detection, anomaly detection,
    change point detection and segmentation.

Time series transformation
Time series transformers
    Transformers usually referes to classes in the `transformation` module of `aeon`.
    These classes are used to transform {term}`time series` data into a different
    format. This may be to reduce the dimensionality of the data, to extract features
    from the data, or to transform the data into a different format.

    See {term}`series-to-series transformation` and {term}`series-to-features
    transformation` for types of transformer.

Collection transformers
    {term}`Time series transformers` that take a {term}`time series collection` as
    input. While these transformers only accept collectons, a wrapper is provided to
    allow them to be used with singular time series datatypes.

Series-to-series transformation
    {term}`Time series transformers` that take a {term}`time series` as input and
    output a (different) time series. An example of this is the Siscrete
    Fourier Transform (DFT).

Series-to-features transformation
    {term}`Time series transformers` that take a {term}`time series` as input and
    output a set of features (in {term}`tabular` format for {term}`time series
    collections`. An example of this is the extraction of the mean and various other
    summary statistics from the series.

Instances
Instance
    A member of the set of entities being studied and which an machine learning
    practitioner wishes to generalize. For example, patients, chemical process runs,
    machines, countries, etc.

    May also be referred to as cases, samples, examples, observations or records
    depending on the discipline and context.

[//]: # (Panel time series)

[//]: # (    A form of time series data where the same time series are observed observed for)

[//]: # (    multiple observational units. The observed series may consist of {term}`univariate)

[//]: # (    time series` or {term}`multivariate time series`. Accordingly, the data varies)

[//]: # (    across time, observational unit and series &#40;i.e. variables&#41;.)

[//]: # ()
[//]: # (Time series collection)

[//]: # (Time series collections)

[//]: # (    todo)

Univariate time series
    A single {term}`time series`.

Multivariate time series
    A {term}`time series` with multiple {term}`channels`. Typically observed for the
    same observational unit. Multivariate time series is typically used to refer to
    cases where the series evolve together over time.

    An example of time series data with multiple channels is data extracted from a
    gyroscope sensor, which can produce different time series data for the x, y and
    z axes of the device.

Reduction
    Reduction refers to decomposing a given learning task into simpler tasks that can
    be composed to create a solution to the original task. In `aeon` reduction is used
    to allow one learning task to be adapted as a solution for an alternative task.

Trend
    When data shows a long-term increase or decrease, this is referred to as a trend.
    Trends can also be non-linear.

Seasonality
    When a {term}`time series` is affected by seasonal characteristics such as the time
    of year or the day of the week, it is called a seasonal pattern.
    The duration of a season is always fixed and known.

Tabular
    A 2 dimensional data structure where the rows of the matrix represent {
    term}`instances` and the columns represent {term}`variables`. This is the most
    common data structure used in `scikit-learn`.

    A {term}`univariate time series` can be formatted in this way, where each
    variable of being measured for each instance are treated as
    features and stored as a primitive data type in the 2d data structure. E.g., there
    are N instances of time series and each has T {term}`timepoint`, this would yield
    a matrix with shape (N, T): N rows, T columns.

[//]: # (random_state)

[//]: # (    https://scikit-learn.org/stable/glossary.html#term-random_state)

[//]: # ()
[//]: # (n_jobs)

[//]: # (    https://scikit-learn.org/stable/glossary.html#term-random_state)
```
