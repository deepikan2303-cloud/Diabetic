"""Transfer-learning CNN: ImageNet-pretrained backbone + small classification head."""
import tensorflow as tf
from tensorflow.keras import layers

from . import config

BACKBONES = {
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
    "resnet50v2": tf.keras.applications.ResNet50V2,
    "densenet121": tf.keras.applications.DenseNet121,
}


def _preprocess(backbone: str, x):
    """Backbone-specific input scaling, built from layers so the saved model reloads cleanly."""
    if backbone == "efficientnetb0":  # rescales internally, expects 0..255
        return x
    if backbone == "resnet50v2":  # scale to -1..1
        return layers.Rescaling(1 / 127.5, offset=-1)(x)
    # densenet: 0..1 then ImageNet mean/std normalisation
    x = layers.Rescaling(1 / 255.0)(x)
    return layers.Normalization(mean=[0.485, 0.456, 0.406],
                                variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])(x)


def build_model(num_classes: int, backbone: str = "efficientnetb0",
                size: int = config.IMG_SIZE, weights: str | None = "imagenet") -> tuple[tf.keras.Model, tf.keras.Model]:
    """Return (model, base). The base starts frozen for head training."""
    base = BACKBONES[backbone](include_top=False, weights=weights, input_shape=(size, size, 3))
    base.trainable = False

    inputs = layers.Input((size, size, 3), name="image")
    x = base(_preprocess(backbone, inputs), training=False)  # keep BatchNorm in inference mode even when fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs, outputs, name=f"dr_{backbone}"), base


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )


def unfreeze_top(base: tf.keras.Model, n_layers: int) -> None:
    """Unfreeze the last `n_layers` of the backbone, keeping BatchNorm layers frozen."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
