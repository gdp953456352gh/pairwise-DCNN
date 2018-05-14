import numpy as np

from keras import backend
from keras.layers import Activation, Add, Dense, Input, Lambda
from keras.models import Model

INPUT_DIM = 1000

# Model.
h_1 = Dense(1000, activation = "relu")
h_2 = Dense(512, activation = "relu")
h_3 = Dense(256, activation = "relu")
h_4 = Dense(128, activation = "relu")
h_5 = Dense(64, activation = "relu")
s = Dense(1)

# Highlight Segment score.
high_array = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_rel = h_1(rel_doc)
h_2_rel = h_2(h_1_rel)
h_3_rel = h_3(h_2_rel)
h_4_rel = h_4(h_3_rel)
h_5_rel = h_5(h_4_rel)
high_score = s(h_5_rel)

# Unhighlight Segment score.
Unhigh_array = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_irr = h_1(irr_doc)
h_2_irr = h_2(h_1_irr)
h_3_irr = h_3(h_2_irr)
h_4_irr = h_4(h_3_irr)
h_5_irr = h_5(h_4_irr)
Unhigh_score = s(h_5_irr)

# Subtract scores.
negated_Unhigh_score = Lambda(lambda x: -1 * x, output_shape = (1, ))(Unhigh_score)
diff = Add()([high_score, negated_Unhigh_score])

# Pass difference through sigmoid function.
prob = Activation("sigmoid")(diff)
pro1 = Lambda(lambda x: 1- x, output_shape = (1, ))(prob)
# Build model.
model = Model(inputs = [high_array, Unhigh_array], outputs = pro1)
model.compile(optimizer = "adadelta", loss = "binary_crossentropy")

# Fake data.
N = 2
X_1 = 2 * np.random.uniform(size = (N, INPUT_DIM))
X_2 = np.random.uniform(size = (N, INPUT_DIM))
y = np.ones((X_1.shape[0], 1))

# Train model.
NUM_EPOCHS = 10
BATCH_SIZE = 10
history = model.fit([X_1, X_2], y, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, verbose = 1)

# Generate scores.
get_score = backend.function([rel_doc], [rel_score])
get_score([X_1])
get_score([X_2])

# Test result
print (model.predict([X_1, X_2], batch_size=100, verbose=0))