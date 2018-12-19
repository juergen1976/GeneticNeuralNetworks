package loerrach.toth.geneticneuralnetwork;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Optimizer class with genetic algorithms to evolve neural networks
 */
public class Optimizer {

    private Hashtable nn_param_choices;
    private double retain;
    private double random_select;
    private double mutate_chance;

    /**
     * Constructor
     * @param nn_param_choices Possible configurations for a network, see example is EvolutionExecutor
     * @param retain percent between 0.0 and 1.0, how much in a population should be kept for the next generation
     * @param random_select number between 0.0 and 1.0. used to random keep some low performer networks,
     *                      higher number will increase the possibility
     * @param mutate_chance number between 0.0 and 1.0. used for doing mutation. higher number will increase the possibility
     */
    public Optimizer(Hashtable nn_param_choices, double retain, double random_select, double mutate_chance) {
        this.nn_param_choices = nn_param_choices;
        this.retain = retain;
        this.random_select = random_select;
        this.mutate_chance = mutate_chance;
    }

    /**
     * Create a new population of networks
     * @param count number of populations
     * @return
     */
    public List<Network> createPopulation(int count) {
        List<Network> result = new ArrayList<>();

        IntStream.range(0, count).mapToObj(i -> new Network(this.nn_param_choices)).forEach(netConfig -> {
            netConfig.createRandom();
            result.add(netConfig);
        });

        return result;
    }

    /**
     * Get fitness score, which is the accuracy of the network between 0.0 and 1.0
     * @param network Network instance to evaluate
     * @return fitness between 0.0 and 1.0
     */
    public static double fitness(Network network) {
        return network.getAccuracy();
    }

    /**
     * Grade a whole population. The fitness function
     * @param population a list of networks
     * @return grade between 0.0 and 1.0
     */
    public double grade(ArrayList<Network> population) {
        double result = 0;
        for (Network network : population) {
            result = result + Optimizer.fitness(network);
        }

        return result/population.size();
    }

    /**
     * Make two children as parts of their parents.
     * @param father the father Network
     * @param mother the mother Network
     * @return Two new children of network
     */
    public List<Network> breed(Network father, Network mother) {
        List<Network> twoChildren = new ArrayList<>();

        // Randomly mutate some children
        IntStream.range(0, 2).mapToObj(i -> new Network(this.nn_param_choices)).forEach(child -> {
            Hashtable config = new Hashtable();
            config.put("nb_neurons", getRandomFatherOrMother(father, mother).getConfig().get("nb_neurons"));
            config.put("nb_layers", getRandomFatherOrMother(father, mother).getConfig().get("nb_layers"));
            config.put("activation", getRandomFatherOrMother(father, mother).getConfig().get("activation"));
            config.put("optimizer", getRandomFatherOrMother(father, mother).getConfig().get("optimizer"));
            if (mutate_chance > Math.random()) {
                mutate(child);
            }
            twoChildren.add(child);
        });

        return twoChildren;
    }

    /**
     * Mutate a desired network. Mutate either the number of neuron, number of layers, the activation function or the optimizer
     * @param network a network which should be randomly mutated
     */
    public void mutate(Network network) {
        List<String> keys = Arrays.asList("nb_neurons", "nb_layers", "activation", "optimizer");
        String randomHyperParameterKey = keys.get(new Random().nextInt(keys.size()));
        List hyperParameterValues = (List) this.nn_param_choices.get(randomHyperParameterKey);
        network.getConfig().put(randomHyperParameterKey, hyperParameterValues.get(new Random().nextInt(hyperParameterValues.size())));
    }

    /**
     * Evolve a population for the next generation.  This will include scoring, selection, mutation, breeding
     * @param population a list of networks which will form one population
     * @return the next generation of the population gone trough the genetic algorithm
     */
    public List<Network> evolve(List<Network> population) {
        List<Network> strongestNetworks = orderStrongestNetworks(population);

        // Get the number we want to retain for the next generation
        int retain_length = (int) (population.size() * this.retain);

        // Survivors will go to the next generation
        List<Network> survivors = strongestNetworks.subList(0, retain_length);

        // Keep some random low performers from the end of the strongest
        for (int i=strongestNetworks.size(); i>strongestNetworks.size() - retain_length; i--) {
            if (this.random_select > Math.random()) {
                survivors.add(strongestNetworks.get(i));
            }
        }

        // If there are some free spots for the next generation, fill them up with children
        int newChildrenLength = population.size() - survivors.size();
        List<Network> children = new ArrayList<>();
        while (children.size() < newChildrenLength) {
            // Get random father and mother
            int fatherIndex = new Random().nextInt(survivors.size());
            int motherIndex = new Random().nextInt(survivors.size());
            if (fatherIndex != motherIndex) {
                List<Network> twoKids = breed(survivors.get(fatherIndex), survivors.get(motherIndex));
                for (Network kid : twoKids) {
                    if (children.size() < newChildrenLength) {
                        children.add(kid);
                    }
                }
            }
        }
        survivors.addAll(children);

        return survivors;
    }

    /**
     * Order the network top-down according to the fitness of each network
     * @param networks Networks for sorting
     * @return sorted networks according to the fitness of each network
     */
    public List<Network> orderStrongestNetworks(List<Network> networks) {
        // Create a TreeMap which sorts the entries according to network accuracy top-down
        TreeMap<Double, Network> gradedNetworks = new TreeMap(Collections.reverseOrder());
        networks.forEach(network -> gradedNetworks.put(new Double(network.getAccuracy()), network));
        return new ArrayList(gradedNetworks.values());

    }

    private Network getRandomFatherOrMother(Network father, Network mother) {
        boolean isFather = new Random().nextBoolean();
        return isFather ? father : mother;
    }


}
