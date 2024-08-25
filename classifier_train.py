import tensorflow as tf

color_mode = "grayscale"
number_colour_layers = 1
image_size = (512, 512)
image_shape = image_size + (number_colour_layers,)

image_shape

training_data_path = "./Data/archive/DAGM_dataset/Class1/Train"
#test_data_path = "./casting_data/casting_data/test"
SEED = 42

def get_image_data(data_path, color_mode, image_size, seed = None):

    return tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        color_mode=color_mode,
        image_size=image_size,
        seed=seed,
        batch_size=6
    )


training_ds = get_image_data(
    training_data_path,
    color_mode,
    image_size,
    SEED
)


preprocessing_layers = [
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(512,512,1))
]


def conv_2d_pooling_layers(filters, number_colour_layers):
    return [
        tf.keras.layers.Conv2D(
            filters,
            number_colour_layers,
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D()
    ]
core_layers =     conv_2d_pooling_layers(8, number_colour_layers) +     conv_2d_pooling_layers(16, number_colour_layers) +     conv_2d_pooling_layers(32, number_colour_layers)


class_names = training_ds.class_names
number_classes = len(class_names)
dense_layers = [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(number_classes)
]

model = tf.keras.Sequential(
    preprocessing_layers +
    core_layers +
    dense_layers
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer='adam',
    loss=loss,
    metrics=['accuracy']
)



callback = tf.keras.callbacks.EarlyStopping(
    monitor='acc', patience=3, mode='auto'
)



history = model.fit(
    training_ds,epochs = 4,callbacks = [callback]
)


model.save("./model/2Wobot1_train_model.h5")