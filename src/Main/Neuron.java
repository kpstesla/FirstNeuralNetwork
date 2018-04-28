package Main;

/*
** Name : Neuron
** Author : Kyle
** Date : 2/8/18
** Description : 
*/

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class Neuron {

    //Total gradient
    List<Double> gradient = new ArrayList<Double>();

    //bias
    double bias;
    //derivative of cost with respect to bias
    double dCostdBias;
    //stores all of the Biases
    List<Double> dCostdBiases = new ArrayList<Double>();
    //average dCostdBias
    double avgdCostdBias;

    //weights
    List<Double> weights = new ArrayList<Double>();
    //derivative of cost with respect to weights
    List<Double> dCostdWeights = new ArrayList<Double>();
    //list of lists of all of the dWeightdCosts for all of the examples
    List<List<Double>> dCostdWeightsStorage = new ArrayList<List<Double>>();
    //average dCostdWeights
    List<Double> avgdCostdWeights = new ArrayList<Double>();

    //activationValue
    double activationValue;
    //derivative of cost with respect to activation value
    double dCostdActivationValue;

    //other stuff
    int numberOfNeuronsInPrevLayer;
    double weightedSum;
    int locationInLayer;



    //constructor
    //creates weights and bias at 0
    //creates derivative lists as well
    public Neuron(int paramNumberOfNeuronsInPrevLayer, int paramLocationInLayer){
        numberOfNeuronsInPrevLayer = paramNumberOfNeuronsInPrevLayer;

        // creating weights list
        List<Double> tempList;
        for (int i = 0; i < numberOfNeuronsInPrevLayer; i++){
            Double tempWeight = 0.0;
            weights.add(tempWeight);
            dCostdWeights.add(tempWeight);
            avgdCostdWeights.add(tempWeight);
            //creates blank list for the gradient
            gradient.add(tempWeight);
            //Creates a temporary Arraylist and stores it in dCostdWeightsStorage
            tempList = new ArrayList<Double>();
            dCostdWeightsStorage.add(tempList);
        }

        //sets bias
        bias = 0.0;
        dCostdBias = 0.0;
        //creates spot for bias
        gradient.add(0.0);
        //creates spot for activation value
        gradient.add(0.0);

        //sets the location of the neuron
        locationInLayer = paramLocationInLayer;
    }

    //initializes the weights and bias randomly between from and to
    public void initWeightsandBias(int from, int to){
        double tempRandom;
        Random r = new Random();
        for (int i = 0; i < weights.size(); i++){
            tempRandom = from + ((to - from) * r.nextDouble());
            weights.set(i,tempRandom);
        }
        tempRandom = from + ((to - from) * r.nextDouble());
        bias = tempRandom;
    }

    //calculates the weighted sum z of the neuron using the neurons in the previous layer
    public void calculateWeightedSum(List<Neuron> prevLayer){
        weightedSum = 0.0;
        for (int i = 0; i < prevLayer.size(); i++){
            weightedSum += weights.get(i) * prevLayer.get(i).getActivationValue();
        }
        weightedSum += bias;
    }

    //takes in input as a weighted sum for input neurons only
    public void calculateInputWeightedSum(double input){
        weightedSum = input;
    }

    //calculates the activation value using the sigmoid function
    public void calculateActivationValue(){
        activationValue = Network.sigmoid(weightedSum);
    }

    //returns the weighted sum
    public double getWeightedSum(){
        return weightedSum;
    }

    //returns the activation value
    public double getActivationValue(){
        return activationValue;
    }

    //returns the specified weight
    public double getWeight(int index){
        return weights.get(index);
    }

    //returns the dCostdActivationValue
    public double getdCostdActivationValue(){
        return dCostdActivationValue;
    }
    //calculates the activation value with respect to the cost function for the last neuron
    public double calcdCostdActivationValueOutput(double expectedValue){
        dCostdActivationValue = 0.0;
        dCostdActivationValue = 2 * (activationValue - expectedValue);
        return dCostdActivationValue;
    }

    //calculates the derivative of cost with respect to the activation value
    public double calcdCostdActivationValue(List<Neuron> nextLayer){
        //sets variables
        dCostdActivationValue = 0.0;
        double weightLplus1;
        double sigmoidPrimeOfNextWeightedSum;
        double dCDANext;
        //iterates through every neuron in the next layer
        for (int i = 0, n = nextLayer.size(); i < n; i++){

            //value of the weight of the connection between this neuron and the next one
            weightLplus1 = nextLayer.get(i).getWeight(locationInLayer);

            //sigmoid prime of the weighted sum of the next neuron
            sigmoidPrimeOfNextWeightedSum = Network.sigmoidPrime(nextLayer.get(i).getWeightedSum());

            //gets the derivative of the cost function with respect to the activation value of the next neuron
            dCDANext = nextLayer.get(i).getdCostdActivationValue();

            //calculates it
            dCostdActivationValue += weightLplus1 * sigmoidPrimeOfNextWeightedSum * dCDANext;
        }
        return dCostdActivationValue;
    }

    //calculates the derivative of cost with respect to the bias
    public double calcdCostdBias(){
        dCostdBias = 0.0;
        dCostdBias = Network.sigmoidPrime(weightedSum) * dCostdActivationValue;
        return dCostdBias;
    }

    //calculates the derivative of cost with respect to all of the weights
    public void calcdCostdWeights(List<Neuron> prevLayer){
        double a;
        double sigPrimeZ;
        double value;
        for (int i = 0, n = prevLayer.size(); i < n; i++){
            a = prevLayer.get(i).getActivationValue();
            sigPrimeZ = Network.sigmoidPrime(weightedSum);
            value = a * sigPrimeZ * dCostdActivationValue;
            dCostdWeights.set(i,value);
        }
    }

    //Calculates all of the gradients for a hidden neuron
    public void calcAllHidden(List<Neuron> prevLayer, List<Neuron> nextLayer){
        calcdCostdActivationValue(nextLayer);
        calcdCostdBias();
        calcdCostdWeights(prevLayer);
    }

    //Calculates all of the gradients for an output neuron
    public void calcAllOutput(List<Neuron> prevLayer, double[] expectedValues){
        calcdCostdActivationValueOutput(expectedValues[locationInLayer]);
        calcdCostdBias();
        calcdCostdWeights(prevLayer);
    }

    //Calculates all of the derivatives for an input neuron
    public void calcAllInput(List<Neuron> nextLayer){
        calcdCostdActivationValue(nextLayer);
        calcdCostdBias();
    }

    //stores all of the derivative values in the gradient
    //stores all of the derivative values in their respective holders
    public void storeAll(){
        //stores in the gradient
        gradient.set(0,dCostdActivationValue);
        gradient.set(1,dCostdBias);
        for (int i = 0; i < dCostdWeights.size(); i++){
            gradient.set(i+2,dCostdWeights.get(i));
        }
        //stores the bias
        dCostdBiases.add(dCostdBias);
        //stores the weights
        for (int i = 0; i < dCostdWeights.size(); i++){
            dCostdWeightsStorage.get(i).add(dCostdWeights.get(i));
        }
    }

    public void clearStorage(){
        for (int i = 0; i < dCostdWeightsStorage.size(); i++){
            dCostdWeightsStorage.get(i).clear();
        }
        dCostdBiases.clear();
    }

    //Calculates all of the respective average values of the derivatives
    public void calcAvg(){
        double tempSum = 0.0;
        double tempAvg = 0.0;

        //calculates the average bias
        for (int i = 0; i < dCostdBiases.size(); i++){
            //sum of the biases
            tempSum += dCostdBiases.get(i);
        }
        tempAvg = tempSum / dCostdBiases.size();
        avgdCostdBias = tempAvg;

        tempSum = 0.0;
        //calculates all of the average weights
        //iterates through all of the weights lists
        for (int i = 0; i < dCostdWeightsStorage.size(); i++) {
            //iterates through each value in the storagelist
            for (int j = 0; j < dCostdWeightsStorage.get(i).size(); j++) {
                tempSum += dCostdWeightsStorage.get(i).get(j);
            }
            //calculates
            tempAvg = tempSum / dCostdWeightsStorage.get(i).size();
            avgdCostdWeights.set(i,tempAvg);
            tempSum = 0.0;
        }
    }

    //updates the bias
    public void updateBias(double lr){
        double value = lr * avgdCostdBias;
        bias -= value;
    }

    //updates all of the weights
    public void updateWeights(double lr){
        double tempValue;
        for (int i = 0; i < avgdCostdWeights.size(); i++){
            tempValue = avgdCostdWeights.get(i) * lr;
            weights.set(i, weights.get(i) - tempValue);
        }
    }

    //updates all
    public void updateAll(double lr){
        updateBias(lr);
        updateWeights(lr);
    }

    public List<Double> getGradient(){
        return gradient;
    }
}
