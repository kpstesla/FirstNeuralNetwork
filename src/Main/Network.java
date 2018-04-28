package Main;

/*
** Name : Network
** Author : Kyle
** Date : 2/8/18
** Description : 
*/
import java.util.List;
import java.util.ArrayList;
public class Network {
    //the actual network
    List<List<Neuron>>  network = new ArrayList<List<Neuron>>();
    List<Neuron> inputLayer;
    //testing values
    int[] labels;
    List<int[][]> images;
    //learning rate
    double learningRate;
    //used in the input to create the net
    int[] numberOfLayers;
    //used to create random values
    int minimumRandomValue;
    int maximumRandomValue;
    //how bad the current test was
    double costFuncion;
    //the current expected values of the output layer
    double[] expectedValues;
    //stores the current gradient
    List<Double> gradient = new ArrayList<Double>();
    //Stores the all of the gradients for the minibatch
    List<List<Double>> gradients= new ArrayList<List<Double>>();
    //Stores the average gradient for the minibatch
    List<Double> avgGradient = new ArrayList<Double>();
    //Correct guesses and stuff
    int correctGuesses;
    int incorrectGuesses;
    double accuracy;

    //more instnace data for mnist
    int guessedValue;
    int actualValue;


    //constructor
    public Network(int[] paramLabels, List<int[][]> paramImages, double paramLearningRate, int[] paramNumberOfLayers, int paramMinimumRandomValue, int paramMaximumRandomValue){

        //Sets instance data
        labels = paramLabels;
        images = paramImages;
        learningRate = paramLearningRate;
        numberOfLayers = paramNumberOfLayers;
        minimumRandomValue = paramMinimumRandomValue;
        maximumRandomValue = paramMaximumRandomValue;

        //Creates and adds neurons to the network
        Neuron tempNeuron;
        for (int i = 0; i < numberOfLayers.length; i++){
            List<Neuron> tempList = new ArrayList<Neuron>();
            for (int j = 0; j < numberOfLayers[i]; j++){
                if (i == 0)
                    tempNeuron = new Neuron(0, j);
                else
                    tempNeuron = new Neuron(numberOfLayers[i-1], j);
                //adds neurons to the temporary list
                tempList.add(tempNeuron);
            }
            //adds the temporary list to the network
            network.add(tempList);
        }

        //more instance data
        inputLayer = network.get(0);
    }

    //Initializes the values of the weights and biases as determined by the constructor parameter data
    public void initWeightsandBiases(){
        //iterates through layers
        for (int i = 0; i < network.size(); i++){
            //iterates through rows
            for (int j = 0; j < network.get(i).size(); j++){
                //inits each neuron randomly
                network.get(i).get(j).initWeightsandBias(minimumRandomValue, maximumRandomValue);
            }
        }
    }

    //sends the input values to all of the input neurons
    //takes in an array of corresponding values
    //Ex. the value in the array at index 0 will be sent to the 0th input neuron
    public void setInputs(int[] inputArray){
        //iterates through the first layer
        for (int i = 0, n = inputLayer.size(); i < n; i++){
            //sets weighted sum
            inputLayer.get(i).calculateInputWeightedSum(inputArray[i]);
            //calculates activation value
            inputLayer.get(i).calculateActivationValue();
        }
    }

    //uses the input neurons as inputs and calculates the activation value of all of the layers sequentially
    public void feedForward(){
        Neuron currentNeuron;
        //iterates through all of the layers except for the first
        for (int i = 1, n = network.size(); i < n; i++){
            //iterates through all of the neurons in that layer
            for (int j = 0; j < network.get(i).size(); j++){
                //gets the jth neuron in the ith layer
                currentNeuron = network.get(i).get(j);
                //calculates the weighted sum
                currentNeuron.calculateWeightedSum(network.get(i - 1));
                //calculates the activation value
                currentNeuron.calculateActivationValue();
            }
        }
    }

    //calculates and returns the cost function, the sum of the squared differences between the activation value and the expected value for
    // the output neurons
    public double calculateCostFunction(double[] paramExpectedValues){
        expectedValues = paramExpectedValues;
        double sum = 0;
        double[] outputValues = getActivationValues(network.get(network.size() - 1));
        //makes sure they are the same length
        if (expectedValues.length != outputValues.length){
            System.err.println("Error: expectedValues length != outputValues length!");
            return 0.0;
        }
        for (int i = 0; i < expectedValues.length; i++){
            sum += Math.pow(outputValues[i] - expectedValues[i], 2);
        }
        costFuncion = sum;
        return sum;
    }
    //adds values to the results and such
    public void determineResultsMnist(){

        //determines the correct index
        int correctIndex = 0;
        for (int i = 0; i < expectedValues.length; i++){
            if (expectedValues[i] == 1)
                correctIndex = i;
        }
        actualValue = correctIndex;

        //determines the guessed index
        double largestValue = 0;
        int guessedIndex = 0;
        List<Neuron> outputLayer = network.get(network.size() - 1);
        for (int i = 0; i < outputLayer.size(); i++){
            if (outputLayer.get(i).getActivationValue() > largestValue){
                largestValue = outputLayer.get(i).getActivationValue();
                guessedIndex = i;
            }
        }
        guessedValue = guessedIndex;

        //determines if the network is correct
        if (correctIndex == guessedIndex)
            correctGuesses++;
        else
            incorrectGuesses++;
    }

    //clears the results
    public void clearResults(){
        correctGuesses = 0;
        incorrectGuesses = 0;
    }

    //clears the storage for all of the neurons to save memory
    //clears the storage for the gradient as well
    public void clearStorage(){
        Neuron temp;
        //iterates through all of the neurons
        for (int i = 0; i < network.size(); i++){
            for (int j = 0; j < network.get(i).size(); j++){
                //clears the storage of each neuron
                temp = network.get(i).get(j);
                temp.clearStorage();
            }
        }
        //clears all of the gradients storage
        for (int i = 0, n = gradients.size(); i < n; i++){
            gradients.get(i).clear();
        }
    }
    //calculates all of the derivatives for every neuron and stores them locally in the neurons and all together in the network
    public void backpropagate(){

        //Calculates Values
        //temp neuron to store shit
        Neuron currentNeuron;
        //output neurons
        for (int i = 0, n = network.get(network.size() - 1).size(); i < n; i++){
            currentNeuron = network.get(network.size() - 1).get(i);
            currentNeuron.calcAllOutput(network.get(network.size() - 2), expectedValues);
            currentNeuron.storeAll();
        }

        //hidden neurons
        //goes backwards, starting at the second to last layer and ending on the second
        for (int i = network.size() - 2; i > 0; i--){
            //iterates through each neuron in the layer
            for (int j = 0; j < network.get(i).size(); j++){
                currentNeuron = network.get(i).get(j);
                currentNeuron.calcAllHidden(network.get(i - 1), network.get(i + 1));
                currentNeuron.storeAll();
            }
        }

        //input neurons
        //iterates through using the second layer as the next one
        for (int i = 0, n = inputLayer.size(); i < n; i++){
            currentNeuron = inputLayer.get(i);
            currentNeuron.calcAllInput(network.get(1));
            currentNeuron.storeAll();
        }

        /*
        //Stores values in gradient
        //iterates through each layer in the network
        double tempGradient;
        //clears the old gradient
        gradient.clear();
        for (int i = 0; i < network.size(); i++){
            //iterates through the neurons in each layer
            for (int j = 0; j < network.get(i).size(); j++){
                //iterates through each derivative in the gradient
                for (int k = 0; k < network.get(i).get(j).getGradient().size(); k++){
                    //current gradient
                    tempGradient = network.get(i).get(j).getGradient().get(k);
                    gradient.add(tempGradient);
                }
            }

        }
        */
    }

    //Stores the value of a gradient in the gradients 2d list
    public void storeGradient(){
        //the first time things are stored the array is created
        if (gradients.size() == 0){
            for (int i = 0; i < gradient.size(); i++){
                //creates a temporary array and stores it
                List<Double> tempList = new ArrayList<Double>();
                gradients.add(tempList);
            }
        }
        for (int i = 0; i < gradient.size(); i++){
            gradients.get(i).add(gradient.get(i));
        }
    }

    //calculates the average gradient from the gradients list
    public void calculateAvgGradient(){
        //the first time things are stored the array is created
        if (avgGradient.size() == 0) {
            for (int i = 0; i < gradient.size(); i++) {
                avgGradient.add(0.0);
            }
        }
        double tempSum = 0.0;
        double tempAvg = 0.0;
        //iterates through the avgGradient list
        for (int i = 0; i < avgGradient.size(); i++){
            //iterates through the length of each element in the gradients list
            for (int j = 0; j < gradients.get(i).size(); j++){
                //finds the sum of all of the elements in each gradients list
                tempSum += gradients.get(i).get(j);
            }
            //sum of the values divided amount of values
            tempAvg = tempSum / gradients.get(i).size();
            avgGradient.set(i, tempAvg);
            tempSum = 0.0;
        }
    }

    //Calculates all of values locally and in the network
    public void calcAll(){
        calculateAvgGradient();
        for(int i = 0; i < network.size(); i++){
            for (int j = 0; j < network.get(i).size(); j++){
                network.get(i).get(j).calcAvg();
            }
        }
        accuracy = (double) correctGuesses / (correctGuesses + incorrectGuesses);
    }

    //updates all of the neurons
    public void updateAll(){
        for(int i = 0; i < network.size(); i++){
            for (int j = 0; j < network.get(i).size(); j++){
                network.get(i).get(j).updateAll(learningRate);
            }
        }
    }

    public List<Double> getAvgGradient(){
        return avgGradient;
    }

    //returns an array of doubles that contains the activation values for each neuron in the layer
    public double[] getActivationValues(List<Neuron> currentLayer){
        int length = currentLayer.size();
        double[] output = new double[length];
        for (int i = 0; i < currentLayer.size(); i++){
            output[i] = currentLayer.get(i).getActivationValue();
        }
        return output;
    }

    public double[] getWeightedSums (List<Neuron> currentLayer){
        int length = currentLayer.size();
        double[] output = new double[length];
        for (int i = 0; i < currentLayer.size(); i++){
            output[i] = currentLayer.get(i).getWeightedSum();
        }
        return output;
    }

    public double getCostFuncion(){
        return costFuncion;
    }

    public List<Double> getGradient(){
        return gradient;
    }

    public int getGuessedValue(){
        return guessedValue;
    }

    public double getAccuracy(){
        return accuracy;
    }

    public static double sigmoid(double x){
        double value;
        value = 1.0 / (1 + Math.exp(-x));
        return value;
    }

    public static double sigmoidPrime(double x){
        double value;
        value = (Network.sigmoid(x)) * (1 - Network.sigmoid(x));
        return  value;
    }
}
