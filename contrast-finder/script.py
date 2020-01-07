from neural_network import NeuralNetwork

if __name__ == "__main__":

    neural_network = NeuralNetwork([1])

    output = neural_network.feedforward([1])
    print(output)