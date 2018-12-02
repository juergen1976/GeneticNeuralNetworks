package loerrach.toth.geneticneuralnetwork;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;
import java.util.stream.IntStream;

public class EvolutionExecuter {

    public static void main(String[] args) throws Exception {
        int generations = 2;
        int population = 20;

        Hashtable nn_param_choices = new Hashtable();
        nn_param_choices.put("nb_neurons", Arrays.asList(64, 128, 256, 512, 768, 1024));
        nn_param_choices.put("nb_layers", Arrays.asList(1, 2, 3, 4));
        nn_param_choices.put("activation", Arrays.asList(Activation.RELU, Activation.LEAKYRELU, Activation.TANH, Activation.SOFTMAX));
        nn_param_choices.put("optimizer", Arrays.asList(OptimizationAlgorithm.LINE_GRADIENT_DESCENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT));

        EvolutionExecuter executor = new EvolutionExecuter();
        executor.generate(generations, population, nn_param_choices);
    }

    public void generate(int generationCount, int populationCount, Hashtable nn_param_choices) throws IOException {
        int rngSeed = 123; // random number seed for reproducibility
        int batchSize = 64; // batch size for each epoch
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        Optimizer networkOptimizer = new Optimizer(nn_param_choices, 0.4, 0.1, 0.2 );
        List<Network> population = networkOptimizer.createPopulation(populationCount);

        // Evolve multiple generations
        int i=0;
        while (i<generationCount) {
            // Train network population
            trainNetworkPopulation(population, mnistTrain, mnistTest);

            // Evolve generation, except last one
            if (i <= generationCount -1) {
                population = networkOptimizer.evolve(population);
            }
            i++;
        }

        // Sort the final population
        population = networkOptimizer.orderStrongestNetworks(population);
    }

    public void trainNetworkPopulation(List<Network> population, DataSetIterator mnistTrain, DataSetIterator mnistTest) {
        for (Network network : population) {
            network.train(mnistTrain, mnistTest);
        }
    }

}
