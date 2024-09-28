# %%NBQA-CELL-SEPfc780c
import warnings

from aeon.datasets import load_airline
from aeon.visualisation import plot_series

warnings.filterwarnings("ignore")

airline = load_airline()
plot_series(airline)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_longley

employment, longley = load_longley()
plot_series(employment)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_lynx

lynx = load_lynx()
plot_series(lynx)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_PBS_dataset

pbs = load_PBS_dataset()
plot_series(pbs)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_shampoo_sales

shampoo = load_shampoo_sales()
print(type(shampoo))
plot_series(shampoo)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_uschange

consumption, others = load_uschange()
print(type(consumption))
plot_series(consumption)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_solar

solar = load_solar()
print(type(solar))
plot_series(solar)


# %%NBQA-CELL-SEPfc780c
import matplotlib.pyplot as plt

from aeon.datasets import load_acsf1

trainX, trainy = load_acsf1(split="train")
testX, testy = load_acsf1(split="test")
print(type(trainX))
print(trainX.shape)
plt.plot(trainX[0][0][:100])
plt.title(
    f"First 100 observations of the first train case of the ACFS1 data, class: "
    f"({trainy[0]})"
)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_arrow_head

arrowhead, arrow_labels = load_arrow_head()
print(arrowhead.shape)
plt.title(
    f"First two cases of the ArrowHead, classes: "
    f"({arrow_labels[0]}, {arrow_labels[1]})"
)

plt.plot(arrowhead[0][0])
plt.plot(arrowhead[1][0])


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_basic_motions

motions, motions_labels = load_basic_motions(split="train")
plt.title(
    f"First and second dimensions of the first train instance in BasicMotions data, "
    f"(student {motions_labels[0]})"
)
plt.plot(motions[0][0])
plt.plot(motions[0][1])


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_gunpoint

gun, gun_labels = load_gunpoint(split="test")
plt.title(
    f"First three cases of the test set for GunPoint, classes"
    f"(actor {gun_labels[0]}, {gun_labels[1]}, {gun_labels[2]})"
)
plt.plot(gun[0][0])
plt.plot(gun[1][0])
plt.plot(gun[2][0])


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_italy_power_demand

italy, italy_labels = load_italy_power_demand(split="train")
plt.title(
    f"First three cases of the test set for ItalyPowerDemand, classes"
    f"( {italy_labels[0]}, {italy_labels[1]}, {italy_labels[2]})"
)
plt.plot(italy[0][0])
plt.plot(italy[1][0])
plt.plot(italy[2][0])


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_japanese_vowels

japan, japan_labels = load_japanese_vowels(split="train")
plt.title(
    f"First channel of three test cases for JapaneseVowels, classes"
    f"({japan_labels[0]}, {japan_labels[10]}, {japan_labels[200]})"
)
print(f" number of cases = " f"{len(japan)}")
print(f" First case shape = " f"{japan[0].shape}")
print(f" Tenth case shape = " f"{japan[10].shape}")
print(f" 200th case shape = " f"{japan[200].shape}")

plt.plot(japan[0][0])
plt.plot(japan[10][0])
plt.plot(japan[200][0])


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_osuleaf

leaf, leaf_labels = load_osuleaf(split="train")
plt.title(
    f"First three cases of the test set for OSULeaf, classes"
    f" ({leaf_labels[0]}, {leaf_labels[1]}, {leaf_labels[2]})"
)
plt.plot(leaf[0][0])
plt.plot(leaf[1][0])
plt.plot(leaf[2][0])


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_plaid

plaid, plaid_labels = load_plaid(split="train")
plt.title(
    f"three train cases for PLAID, classes"
    f"( {plaid_labels[0]}, {plaid_labels[10]}, {plaid_labels[200]})"
)
print(f" number of cases = " f"{len(plaid)}")
print(f" First case shape = " f"{plaid[0].shape}")
print(f" Tenth case shape = " f"{plaid[10].shape}")
print(f" 200th case shape = " f"{plaid[200].shape}")

plt.plot(plaid[0][0])
plt.plot(plaid[10][0])
plt.plot(plaid[200][0])


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_covid_3month

covid, covid_target = load_covid_3month()
print(covid.shape)
plt.title("Response variable for Covid3Months data")
plt.plot(covid_target)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_cardano_sentiment

cardano, cardano_target = load_cardano_sentiment()
print(cardano.shape)
plt.title("Response variable for cardano data")
plt.plot(cardano_target)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_electric_devices_segmentation

data, period, change_points = load_electric_devices_segmentation()
print(" Period = ", period)
print(" Change points = ", change_points)
plot_series(data)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_gun_point_segmentation

data, period, change_points = load_gun_point_segmentation()
print(" Period = ", period)
print(" Change points = ", change_points)
plot_series(data)
