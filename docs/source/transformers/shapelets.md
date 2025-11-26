# Shapelets in RDST Classifier / Transformer

Aeon’s RDST classifier/transformer extracts **shapelets** – small patterns from time series data – used for classification.  
This documentation explains how to access, interpret, and visualize them.

---

## Shapelet Attributes

Each shapelet stored in the transformer has these attributes:

| Field          | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `values`       | Numeric values of the shapelet, padded to the size of the largest shapelet |
| `startpoint`   | Starting index of the shapelet in the original time series                  |
| `length`       | Length of the shapelet                                                      |
| `dilation`     | Sampling interval of the shapelet (skip factor)                             |
| `threshold`    | Threshold used to compute Shapelet Occurrence (SO) feature                  |
| `normalization`| Whether the shapelet was normalized                                         |
| `mean`         | Mean used for normalization                                                 |
| `std`          | Standard deviation used for normalization                                   |
| `class`        | Class label from which the shapelet was extracted                           |

> **Note:** The `values` array is padded. Access only the first `length` elements to avoid `inf` values.


## Accessing Shapelet Values

```python
# Fit RDST classifier
rdstclf.fit(X_train, y_train)

# Access first shapelet
shapelet_0 = rdstclf.transformer.shapelets[0]
values = shapelet_0['values']
startpoint = shapelet_0['startpoint']
length = shapelet_0['length']

# Get actual values
shp_values = values[:, :length]
print("Shapelet 0 values:", shp_values)
print("Startpoint:", startpoint)
print("Length:", length)
```

### Threshold and Shapelet Occurrence (SO)

The `threshold` determines whether a subsequence in a time series is considered a **match** for a given shapelet:

- If the distance between the shapelet and a subsequence **< threshold** → **occurrence = 1**  
- Else → **occurrence = 0**

This binary Shapelet Occurrence (SO) feature is what the classifier ultimately uses.

**Example:**

```python
distance = some_distance_function(time_series_subsequence, shapelet_0)
if distance < shapelet_0['threshold']:
    occurrence = 1
else:
    occurrence = 0
```

### Visual Diagram of a Shapelet (conceptual)
```text
+-------------------------------+
| v1  v2  v3  ...  v_length     |  ← real values (use only these)
| ...  inf  inf  inf            |  ← padding → ignore!
+-------------------------------+
| startpoint : 42               |
| length     : 18               |
| dilation   : 2                |
| threshold  : 0.47             |
| class      : 1                |
+-------------------------------+
```

### Loop Through All Shapelets
```python
for i, shp in enumerate(rdstclf.transformer.shapelets):
    real_vals = shp['values'][:, :shp['length']].flatten()
    
    print(f"Shapelet {i} | Class {shp['class']}")
    print(f"   Length     : {shp['length']}")
    print(f"   Startpoint : {shp['startpoint']}")
    print(f"   Dilation   : {shp['dilation']}")
    print(f"   Threshold  : {shp['threshold']:.4f}")
    print(f"   Values     : {real_vals}")
    print("-" * 60)
```


## What we have covered here
- Meaning of every shapelet attribute
- How to extract the real (unpadded) numeric values
- How threshold and Shapelet Occurrence (SO) features work
- How to inspect and visualize all discovered shapelets



