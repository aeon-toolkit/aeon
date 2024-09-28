# %%NBQA-CELL-SEPfc780c
import os

# avoid imports warning for tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

# avoid packages warnings such as sklearn
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# skip future tensorflow warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.decomposition import PCA
from tensorflow.keras import backend as k

from aeon.classification.deep_learning import FCNClassifier, InceptionTimeClassifier
from aeon.clustering.deep_learning import AEFCNClusterer
from aeon.datasets import load_classification, load_regression
from aeon.networks import InceptionNetwork
from aeon.regression.deep_learning import InceptionTimeRegressor


# %%NBQA-CELL-SEPfc780c
xtrain, ytrain = load_classification(name="ArrowHead", split="train")
xtest, ytest = load_classification(name="ArrowHead", split="test")

inc = InceptionTimeClassifier(n_classifiers=5, use_custom_filters=False, n_epochs=3)
inc.fit(X=xtrain, y=ytrain)
ypred = inc.predict(X=xtest)

print("Predictions: ", ypred[0:5])
print("Ground Truth: ", ytest[0:5])


# %%NBQA-CELL-SEPfc780c
xtrain, ytrain = load_classification(name="ArrowHead", split="train")
xtest, ytest = load_classification(name="ArrowHead", split="test")

inc = InceptionTimeClassifier(n_classifiers=5, use_custom_filters=True, n_epochs=3)
inc.fit(X=xtrain, y=ytrain)
ypred = inc.predict(X=xtest)

print("Predictions: ", ypred[0:5])
print("Ground Truth: ", ytest[0:5])


# %%NBQA-CELL-SEPfc780c
xtrain, ytrain = load_regression(name="Covid3Month", split="train")
xtest, ytest = load_regression(name="Covid3Month", split="test")

inc = InceptionTimeRegressor(n_regressors=5, n_epochs=1, use_custom_filters=False)
inc.fit(X=xtrain, y=ytrain)
ypred = inc.predict(X=xtest)

print("Predictions: ", ypred[0:5])
print("Ground Truth: ", ytest[0:5])


# %%NBQA-CELL-SEPfc780c
xtrain, _ = load_classification(name="ArrowHead", split="train")
xtest, ytest = load_classification(name="ArrowHead", split="test")

aefcn = AEFCNClusterer(
    n_clusters=2,
    temporal_latent_space=False,
    clustering_algorithm="kmeans",
    n_epochs=10,
)

aefcn.fit(X=xtrain)
ypred = aefcn.predict(X=xtest)
print("Predictions: ", ypred[0:5])
print("Ground Truth: ", ytest[0:5])
print()
print("Score : ", aefcn.score(X=xtest))


# %%NBQA-CELL-SEPfc780c
xtrain, ytrain = load_classification(name="ArrowHead", split="train")
xtest, ytest = load_classification(name="ArrowHead", split="test")

fcn = FCNClassifier(
    save_best_model=True,
    save_last_model=True,
    file_path="./",
    best_file_name="best_fcn",
    last_file_name="last_fcn",
    n_epochs=2,
)

fcn.fit(X=xtrain, y=ytrain)
ypred = fcn.predict(X=xtest)

# Loading the model ento another FCN model

fcn2 = FCNClassifier()
fcn2.load_model(model_path="./best_fcn.keras", classes=np.unique(ytrain))
# The classes parameter is only needed in the case of classification
ypred2 = fcn2.predict(X=xtest)

print(ypred)
print(ypred2)

os.remove("./best_fcn.keras")
os.remove("./last_fcn.keras")


# %%NBQA-CELL-SEPfc780c
# define self-supervised space dimension

n_dim = 16

# load the data

xtrain, ytrain = load_classification(name="ArrowHead", split="train")
xtest, ytest = load_classification(name="ArrowHead", split="test")

# Flip axis to be handled correctly in tensorflow

xtrain = np.transpose(xtrain, axes=(0, 2, 1))
xtest = np.transpose(xtest, axes=(0, 2, 1))


# %%NBQA-CELL-SEPfc780c
def triplet_loss_function(alpha):
    def temp(ytrue, ypred):
        ref = ypred[:, :, 0]
        pos = ypred[:, :, 1]
        neg = ypred[:, :, 2]

        ref = k.cast(ref, dtype=ref.dtype)
        pos = k.cast(pos, dtype=ref.dtype)
        neg = k.cast(neg, dtype=ref.dtype)

        loss = k.maximum(
            k.sum(
                tf.math.subtract(
                    tf.math.add(k.square(ref - pos), alpha), k.square(ref - neg)
                ),
                axis=1,
            ),
            0,
        )

        return loss

    return temp


# %%NBQA-CELL-SEPfc780c
# Define the triplets input layers


input_ref_layer = tf.keras.layers.Input(xtrain.shape[1:])
input_pos_layer = tf.keras.layers.Input(xtrain.shape[1:])
input_neg_layer = tf.keras.layers.Input(xtrain.shape[1:])

inc_network = InceptionNetwork()
input_layer, gap_layer = inc_network.build_network(input_shape=xtrain.shape[1:])

encoder = tf.keras.models.Model(inputs=input_layer, outputs=gap_layer)

output_layer_ref = tf.keras.layers.Reshape(target_shape=(-1, 1))(
    tf.keras.layers.Dense(units=n_dim)(encoder(input_ref_layer))
)
output_layer_pos = tf.keras.layers.Reshape(target_shape=(-1, 1))(
    tf.keras.layers.Dense(units=n_dim)(encoder(input_pos_layer))
)
output_layer_neg = tf.keras.layers.Reshape(target_shape=(-1, 1))(
    tf.keras.layers.Dense(units=n_dim)(encoder(input_neg_layer))
)

output_layer = tf.keras.layers.Concatenate(axis=-1)(
    [output_layer_ref, output_layer_pos, output_layer_neg]
)

SSL_model = tf.keras.models.Model(
    inputs=[input_ref_layer, input_pos_layer, input_neg_layer], outputs=output_layer
)

SSL_model.compile(loss=triplet_loss_function(alpha=1e-5))


# %%NBQA-CELL-SEPfc780c
def triplet_generation(x):
    w1 = np.random.choice(np.linspace(start=0.6, stop=1, num=1000), size=1)
    w2 = (1 - w1) / 2

    ref = np.random.permutation(x[:])

    n = int(ref.shape[0])

    pos = np.zeros(shape=ref.shape)
    neg = np.zeros(shape=ref.shape)

    all_indices = np.arange(start=0, stop=n)

    for i in range(n):
        temp_indices = np.delete(arr=all_indices, obj=i)
        indice_neg = int(np.random.choice(temp_indices, size=1))

        temp_indices = np.delete(arr=all_indices, obj=[i, indice_neg])
        indice_b = int(np.random.choice(temp_indices, size=1))

        indice_c = int(np.random.choice(temp_indices, size=1))

        indice_b2 = int(np.random.choice(temp_indices, size=1))

        indice_c2 = int(np.random.choice(temp_indices, size=1))

        a = ref[i].copy()

        nota = ref[indice_neg].copy()

        b = ref[indice_b].copy()
        c = ref[indice_c].copy()

        b2 = ref[indice_b2].copy()
        c2 = ref[indice_c2].copy()

        # MixingUp

        pos[i] = w1 * a + w2 * b + w2 * c
        neg[i] = w1 * nota + w2 * b2 + w2 * c2

    return ref, pos, neg


# %%NBQA-CELL-SEPfc780c
xtrain_ref, xtrain_pos, xtrain_neg = triplet_generation(x=xtrain)


# %%NBQA-CELL-SEPfc780c
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
)

file_path = "./best_ssl_model.keras"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=file_path, monitor="loss", save_best_only=True
)

callbacks = [reduce_lr, model_checkpoint]


# %%NBQA-CELL-SEPfc780c
history = SSL_model.fit(
    [xtrain_ref, xtrain_pos, xtrain_neg],
    np.zeros(shape=len(xtrain)),
    epochs=20,
    callbacks=callbacks,
    verbose=False,
)

plt.figure()
plt.plot(history.history["loss"], lw=3, color="blue", label="training loss")
plt.legend()
plt.show()


# %%NBQA-CELL-SEPfc780c
plt.cla()
plt.clf()
plt.close()
plt.figure()

pca = PCA(n_components=2)

# load best model
model = tf.keras.models.load_model("./best_ssl_model.keras", compile=False)
latent_space = model.predict([xtrain, xtrain, xtrain])
latent_space = latent_space[:, :, 0]

pca.fit(latent_space)
latent_space_2d = pca.transform(latent_space)

colors = ["blue", "red", "green"]

for c in range(len(np.unique(ytrain))):
    plt.scatter(
        latent_space_2d[ytrain.astype(np.int64) == c][:, 0],
        latent_space_2d[ytrain.astype(np.int64) == c][:, 1],
        color=colors[c],
        label="class-" + str(c),
    )


plt.legend()
plt.show()


# %%NBQA-CELL-SEPfc780c
os.remove("./best_ssl_model.keras")
