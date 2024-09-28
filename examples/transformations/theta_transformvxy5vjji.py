# %%NBQA-CELL-SEPfc780c
import matplotlib.pyplot as plt

from aeon.datasets import load_airline
from aeon.transformations.series._theta import ThetaTransformer

y = load_airline()
transformer = ThetaTransformer()
transformer.fit(y)
y_thetas = transformer.transform(y)

fig, ax = plt.subplots()
y_thetas.plot(ax=ax, figsize=(12, 7))
plt.legend(["theta=0", "theta=2"])


# %%NBQA-CELL-SEPfc780c
t = ThetaTransformer([0, 0.25, 0.75, 1])
t.fit(y)
y_t = t.transform(y)

fig, ax = plt.subplots()
y_t.plot(ax=ax, figsize=(12, 7))
plt.legend(
    ["theta=0, linear regression", "theta=0.25", "theta=0.75", "theta=1, original ts"]
)
plt.ylim(0, 900)


# %%NBQA-CELL-SEPfc780c
t_1 = ThetaTransformer([0, 1, 2, 2.5])
t_1.fit(y)
y_t1 = t_1.transform(y)

fig, ax = plt.subplots()
y_t1.plot(ax=ax, figsize=(12, 7))
plt.legend(
    ["theta=0, linear regression", "theta=1, original ts", "theta=2", "theta=2.5"]
)
