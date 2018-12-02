package loerrach.toth.geneticneuralnetwork;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Hashtable;
import java.util.List;
import java.util.Random;


public class Network {

    private double accuracy;
    private Hashtable nn_param_choices;
    private Hashtable network_config;

    /**
     * NetworkConfiguration constructor
     * @param nn_param_choices Hashtable with following structure
     *                         key: "nb_neurons", value: List<Integer>
     *                         key: "nb_layers", value: List<Integer>
     *                         key: "activation", value: List<Activation>
     *                         key: "optimizer", value: List<OptimizationAlgorithm>
     */
    public Network(Hashtable nn_param_choices) {

        this.nn_param_choices = nn_param_choices;
        this.network_config = new Hashtable();
        accuracy = 0.0;
    }

    public void createRandom() {
        List nb_neuronsChoices = (List) this.nn_param_choices.get("nb_neurons");
        List nb_layersChoices = (List) this.nn_param_choices.get("nb_layers");
        List activationChoices = (List) this.nn_param_choices.get("activation");
        List optimizerChoices = (List) this.nn_param_choices.get("optimizer");

        this.network_config.put("nb_neurons", nb_neuronsChoices.get(new Random().nextInt(nb_neuronsChoices.size())));
        this.network_config.put("nb_layers", nb_layersChoices.get(new Random().nextInt(nb_layersChoices.size())));
        this.network_config.put("activation", activationChoices.get(new Random().nextInt(activationChoices.size())));
        this.network_config.put("optimizer", optimizerChoices.get(new Random().nextInt(optimizerChoices.size())));
    }

    public void createSet(Hashtable network_config) {
        this.network_config = network_config;
    }

    public Hashtable getConfig() {
        return this.network_config;
    }

    public double getAccuracy() {
        return this.accuracy;
    }

    public void train(DataSetIterator mnistTrain, DataSetIterator mnistTest) {
        NetworkTraining trainer = new NetworkTraining();
        this.accuracy = trainer.train_and_score(this, mnistTrain, mnistTest );
    }
}
