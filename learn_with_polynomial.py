import numpy as np  
import math
import random
import matplotlib.pyplot as plt

# usually noise standard deviation is set to a small value, e.g 0.1
noise_sd = 0.4
def generate_training_value():
    input_value = random.uniform(0,2)
    # our data has an underlying pattern of sin(2 * pi * x), with a small gaussian noise
    return (input_value, math.sin(2 * math.pi * input_value) + np.random.normal(0.1,noise_sd,1)[0])

# sample a finite number of points for our training set
training_set_size = 20
training_data = list(zip(*sorted([generate_training_value() for i in range(0,training_set_size)])))
training_inputs = training_data[0]
training_values = training_data[1]

validation_set_size = 10
validation_inputs, validation_values = list(zip(*sorted([generate_training_value() for i in range(0,validation_set_size)])))

min_model_size = 3
max_model_size = 9
fig, axs = plt.subplots(max_model_size - min_model_size, figsize=(12, 9))
fig.tight_layout()

# render the training points
for ax in axs:
 ax.scatter(list(training_inputs), list(training_values), c='red')

# render the underlying sin(2 * pi * x), so we can visualise how close the learning model is to the underlying pattern (as opposed to the noise)
pattern_inputs = np.arange(0,2,0.01)
pattern_values = np.sin(pattern_inputs * 2 * math.pi)
for ax in axs:
    ax.plot(pattern_inputs, pattern_values)


model_errors = []
validation_errors = []


#def root_mean_square(inputs, values):
#    return sum([(prediction_model(training_inputs[i]) - training_values[i]) ** 2 for i in range(0,len(training_inputs))])


for size_increment in range(0, max_model_size - min_model_size):
    model_size = min_model_size + size_increment
    
    # We use least-squares error here, which has a unique minimum for our model. This minimum occurs when the equation Aw = T is satisfied, where
    # w are the coefficients of our polynomial model.
    T = np.array([np.sum(np.array(training_values) * (np.array(training_inputs) ** n)) for n in range(0,model_size)]).T

    A = [[sum((np.array(training_inputs) ** j) * (np.array(training_inputs) ** i)) for j in range(0,model_size)] for i in range(0, model_size)]

    A_inv = np.linalg.inv(A)

    optimal_parameters = np.matmul(A_inv, T)

    # at this point, we construct our polynomial model, using the optimal parameters that minimise the least-squares error function
    def prediction_model(input_value):
        output = 0
        for i in range(0,model_size):
            output += optimal_parameters[i] * input_value ** i
        return output

    prediction_values = list(map(prediction_model, pattern_inputs))
    # plot our polynomial model
    model_plot, = axs[size_increment].plot(pattern_inputs, prediction_values)
    axs[size_increment].legend(handles=[model_plot], labels=[f"M = {model_size}"])

    model_errors.append((model_size, sum([(prediction_model(training_inputs[i]) - training_values[i]) ** 2 for i in range(0,len(training_inputs))])))
    #validation_errors.append()
for ax in axs:
    ax.set_ylim(bottom=-2, top=2)

# stop the suptitle overlapping with the subplots
fig.subplots_adjust(top=0.95)
fig.suptitle('Learning sin(2πx) using a M-order polynomial model')
plt.show()


plt.plot(*list(zip(*model_errors)))
plt.show()





