package loerrach.toth.geneticneuralnetwork;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.util.stream.IntStream;

/**
 * NetworkTraining will create a neural network with the configuration.
 * Also it provides a train method to do the training.
 */
public class NetworkTraining {

    /**
     * Create a new network with the required configuration.
     * @param netConf Network configuration
     * @return a MultiLayerNetwork network
     */
    public MultiLayerNetwork compileNetwork(Network netConf) {

        int nb_layers = (int) netConf.getConfig().get("nb_layers");
        int nb_neurons = (int) netConf.getConfig().get("nb_neurons");
        Activation activation = (Activation) netConf.getConfig().get("activation");
        OptimizationAlgorithm optimizer = (OptimizationAlgorithm) netConf.getConfig().get("optimizer");
        int rngSeed = 123;
        int outputNum = 10;
        final int numRows = 28;
        final int numColumns = 28;

        MultiLayerConfiguration conf;
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4)
            .optimizationAlgo(optimizer)
            .list()
            .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                .nIn(numRows * numColumns)
                .nOut(nb_neurons)
                .activation(activation)
                .weightInit(WeightInit.XAVIER)
                .build());

        // Add layers between
        if (nb_layers > 1) {
             builder.layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                .nIn(nb_neurons)
                .nOut(nb_neurons)
                .activation(activation)
                .weightInit(WeightInit.XAVIER)
                .build());
        }

        // Output Layer
        builder.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                .nIn(nb_neurons)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build());

        conf = builder.build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration
        return model;
    }

    /**
     * Train the network. A new network is created and trained.
     * @param netConfig The network configuration
     * @param mnistTrain The MNIST training set
     * @param mnistTest The MNIST test set
     * @return a score wth the accuracy between 0.0 and 1.0
     */
    public double train_and_score(Network netConfig, DataSetIterator mnistTrain, DataSetIterator mnistTest) {
        MultiLayerNetwork networkModel = this.compileNetwork(netConfig);
        int numEpochs = 15; // number of epochs to perform

        IntStream.range(0, numEpochs).mapToObj(i -> mnistTrain).forEach(networkModel::fit);

        Evaluation eval = new Evaluation(10); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = networkModel.output(next.getFeatures()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        return eval.accuracy();
    }
}
