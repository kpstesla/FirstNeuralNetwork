//fix public private accessor methods stuff

package Main;

import mnist.MnistReader;

import java.util.*;

public class Main {

    public static void main(String[] args) {

        /* SINGLE TEST DATA
        //gets database of images and labels
        String trainingImagesLocation = "/Users/Kyle/Desktop/Computer Science/train-images-idx3-ubyte";
        String trainingLabelsLocation = "/Users/Kyle/Desktop/Computer Science/train-labels-idx1-ubyte";
        int[] labels = MnistReader.getLabels(trainingLabelsLocation);
        List<int[][]> images = MnistReader.getImages(trainingImagesLocation);

        //some changeable instance data
        int imageNum = 235;
        int[] testArray = mnistImageToArray(images.get(imageNum));
        double[] testExpectedValues = mnistLabelToExpectedValues(labels[imageNum]);
        int[] testLayers = {784,16,16,10};

     x2   //neural net stuff
        Network test = new Network(labels, images, 10,testLayers,-5,5);
        test.initWeightsandBiases();
        test.setInputs(testArray);
        test.feedForward();
        test.calculateCostFunction(testExpectedValues);
        test.backpropagate();

        //prints out image stuff
        System.out.println(mnistImageToString(images.get(imageNum)));
        System.out.println("Digit shown in image: " + labels[imageNum]);
        System.out.println();

        //prints out neural net values
        for (int i = 0; i < testLayers.length; i++){
            System.out.println("Layer: " + i);
            double[] temp = test.getActivationValues(test.network.get(i));
            double[] weightedTemp = test.getWeightedSums(test.network.get(i));
            for (int j = 0; j < temp.length; j++){
                System.out.println(weightedTemp[j] + "\t" + temp[j]);
            }
            System.out.println("");
        }
        System.out.println(test.getCostFuncion());

        //prints out gradient
        for (int i = 0; i < test.getGradient().size(); i++){
            System.out.print(test.getGradient().get(i) + "\t");
            if (i % 10 == 0 && i != 0){
                System.out.println();
            }
        } */

        //Multiple Values test

        //gets database of images and labels
        final double startTime = System.currentTimeMillis();
        String trainingImagesLocation = "/Users/Kyle/Desktop/Computer Science/train-images-idx3-ubyte";
        String trainingLabelsLocation = "/Users/Kyle/Desktop/Computer Science/train-labels-idx1-ubyte";
        int[] labels = MnistReader.getLabels(trainingLabelsLocation);
        List<int[][]> imageList = MnistReader.getImages(trainingImagesLocation);
        //creates list of imageObjects
        List<ImageObject> images = new ArrayList<>();
        //adds objects to the images
        for (int i = 0, n = labels.length; i < n; i++) {
            images.add(new ImageObject(labels[i], imageList.get(i)));
        }
        Collections.shuffle(images);

        //stuff
        int[] testLayers = {784, 30, 10};
        int[] testArray;
        double[] testExpectedValues;
        int epochCorrect = 0;
        int epochSize = 0;
        double learningRate = 0.1;

        //Online Graphing lists
        String accuracyList = "";
        String trialsList = "";
        String errorList = "";
        int counter = 1;

        //Timing stuff

        double initTime;
        double shuffleTime;
        double imageToArrayTime;
        double labelToExpectedValuesTime;
        double setInputsTime;
        double feedForwardTime;
        double calcCostFunctionTime;
        double determineResultsTime;
        double backpropagateTime;
        double storeGradientTime;
        double calcAllTime = 0;
        double updateAllTime = 0;
        double clearResultsTime = 0;
        double clearStorageTime = 0;

        List<Double> imageToArrayTimeList = new ArrayList<Double>();
        List<Double> labelToExpectedValuesList = new ArrayList<Double>();
        List<Double> setInputsTimeList = new ArrayList<Double>();
        List<Double> feedForwardTimeList = new ArrayList<Double>();
        List<Double> calcCostFunctionTimeList = new ArrayList<Double>();
        List<Double> determineResultsTimeList = new ArrayList<Double>();
        List<Double> backpropagateTimeList = new ArrayList<Double>();
        List<Double> storeGradientTimeList = new ArrayList<Double>();
        List<Double> calcAllTimeList = new ArrayList<Double>();
        List<Double> updateAllTimeList = new ArrayList<Double>();
        List<Double> clearResultsTimeList = new ArrayList<Double>();
        List<Double> clearStorageTimeList = new ArrayList<Double>();

        double initEndTime;
        double shuffleEndTime;
        double imageToArrayEndTime;
        double labelToExpectedValuesEndTime;
        double setInputsEndTime;
        double feedForwardEndTime;
        double calcCostFunctionEndTime;
        double determineResultsEndTime;
        double backpropagateEndTime;
        double storeGradientEndTime;
        double calcAllEndTime = 0;
        double updateAllEndTime = 0;
        double clearResultsEndTime = 0;
        double clearStorageEndTime = 0;

        //test cost function
        double costFunction = 0.0;
        List<Double> costFunctionValues = new ArrayList<Double>();
        double avgCostFunction = 0.0;

        //Net Stuff
        Network test = new Network(labels, imageList, learningRate, testLayers, -1, 1);

        initTime = System.currentTimeMillis();
        test.initWeightsandBiases();
        initEndTime = System.currentTimeMillis();

        //parameters for testing stuff
        int numberOfEpochs = 30;
        int numberOfRuns = 5000;
        int imagesPerRun = 10;

        //actual testing loop
        for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
            //starts the clock
            double epochStartTime = System.currentTimeMillis();
            epochSize = numberOfRuns * imagesPerRun;
            //shuffles the list

            shuffleTime = System.currentTimeMillis();
            Collections.shuffle(images);
            shuffleEndTime = System.currentTimeMillis();

            for (int j = 0; j < numberOfRuns; j++) {
                for (int i = j * imagesPerRun; i < (j + 1) * imagesPerRun; i++) {

                    //data
                    imageToArrayTime = System.currentTimeMillis();
                    testArray = mnistImageToArray(images.get(i).getImage());
                    imageToArrayEndTime = System.currentTimeMillis();
                    imageToArrayTimeList.add(imageToArrayEndTime - imageToArrayTime);

                    labelToExpectedValuesTime = System.currentTimeMillis();
                    testExpectedValues = mnistLabelToExpectedValues(images.get(i).getLabel());
                    labelToExpectedValuesEndTime = System.currentTimeMillis();
                    labelToExpectedValuesList.add(labelToExpectedValuesEndTime - labelToExpectedValuesTime);

                    //net stuff pt 2
                    setInputsTime = System.currentTimeMillis();
                    test.setInputs(testArray);
                    setInputsEndTime = System.currentTimeMillis();
                    setInputsTimeList.add(setInputsEndTime - setInputsTime);

                    feedForwardTime = System.currentTimeMillis();
                    test.feedForward();
                    feedForwardEndTime = System.currentTimeMillis();
                    feedForwardTimeList.add(feedForwardEndTime - feedForwardTime);

                    calcCostFunctionTime = System.currentTimeMillis();
                    test.calculateCostFunction(testExpectedValues);
                    calcCostFunctionEndTime = System.currentTimeMillis();
                    calcCostFunctionTimeList.add(calcCostFunctionEndTime - calcCostFunctionTime);

                    //temp cost function stuff
                    costFunction = test.getCostFuncion();
                    costFunctionValues.add(costFunction);
                    ///////

                    determineResultsTime = System.currentTimeMillis();
                    test.determineResultsMnist();
                    determineResultsEndTime = System.currentTimeMillis();
                    determineResultsTimeList.add(determineResultsEndTime - determineResultsTime);

                    backpropagateTime = System.currentTimeMillis();
                    test.backpropagate();
                    backpropagateEndTime = System.currentTimeMillis();
                    backpropagateTimeList.add(backpropagateEndTime - backpropagateTime);

                    storeGradientTime = System.currentTimeMillis();
                    test.storeGradient();
                    storeGradientEndTime = System.currentTimeMillis();
                    storeGradientTimeList.add(storeGradientEndTime - storeGradientTime);

                    //printing stuff
//                System.out.println(mnistImageToString(images.get(imageNum)));
//                System.out.println();
//                System.out.println("Actual Value : " + labels[i]);
//                System.out.println("Guessed Value : " + test.getGuessedValue());
//                System.out.println("Cost function : " + test.getCostFuncion());
//                System.out.println();
                }

                //cost function stuff
                for (int k = 0, n = costFunctionValues.size(); k < n; k++) {
                    avgCostFunction += costFunctionValues.get(k);
                }
                avgCostFunction /= costFunctionValues.size();
                costFunctionValues.clear();

                //net stuff part 3
                calcAllTime = System.currentTimeMillis();
                test.calcAll();
                calcAllEndTime = System.currentTimeMillis();
                calcAllTimeList.add(calcAllEndTime - calcAllTime);

                updateAllTime = System.currentTimeMillis();
                test.updateAll();
                updateAllEndTime = System.currentTimeMillis();
                updateAllTimeList.add(updateAllEndTime - updateAllTime);

                clearResultsTime = System.currentTimeMillis();
                test.clearResults();
                clearResultsEndTime = System.currentTimeMillis();
                clearResultsTimeList.add(clearResultsEndTime - clearResultsTime);

                clearStorageTime = System.currentTimeMillis();
                test.clearStorage();
                clearStorageEndTime = System.currentTimeMillis();
                clearStorageTimeList.add(clearStorageEndTime - clearStorageTime);

                /*
                System.out.println("=============================================================");
                */

                //data collection stuff
                /*
                System.out.println("Accuracy : " + test.getAccuracy());
                System.out.println("Avg Cost Function : " + avgCostFunction);
                */
                accuracyList += test.getAccuracy() + " ";
                trialsList += counter + " ";
                errorList += avgCostFunction + " ";
                counter++;
                epochCorrect += test.getAccuracy() * imagesPerRun;
                avgCostFunction = 0.0;

                //Printing out sample neuron
/*
            System.out.println("Sample Neuron");
            Neuron sample = test.network.get(2).get(0);
            for (int k = 0, n = sample.weights.size(); k < n; k++){
                System.out.println("Weight " + k + ": " + sample.getWeight(k));
                System.out.println("\tAvg Grad of Weight " + k + ":\t" + sample.avgdCostdWeights.get(k));
            }

*/
            }
            System.out.println("Epoch # " + epoch);
            /*
            System.out.println("https://www.rapidtables.com/tools/line-graph.html");
            System.out.println("TRIALS\nACCURACY\nERROR");
            System.out.println(trialsList);
            System.out.println(accuracyList);
            System.out.println(errorList);
            */
            System.out.println("Total Correct : " + epochCorrect + "/" + epochSize);
            double epochEndTime = System.currentTimeMillis();
            System.out.println("Time taken in Milliseconds : " + (epochEndTime - epochStartTime));

            //clears data collection values
            epochCorrect = 0;

            //averaging times and stuff
            double tempsum;
            tempsum = 0;
            for (int j = 0, n = imageToArrayTimeList.size(); j < n; j++){
                tempsum += imageToArrayTimeList.get(j);
            }
            imageToArrayTime = tempsum / imageToArrayTimeList.size();
            imageToArrayTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = labelToExpectedValuesList.size(); j < n; j++){
                tempsum += labelToExpectedValuesList.get(j);
            }
            labelToExpectedValuesTime = tempsum / labelToExpectedValuesList.size();
            labelToExpectedValuesList.clear();

            tempsum = 0;
            for (int j = 0, n = setInputsTimeList.size(); j < n; j++){
                tempsum += setInputsTimeList.get(j);
            }
            setInputsTime = tempsum / setInputsTimeList.size();
            setInputsTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = feedForwardTimeList.size(); j < n; j++){
                tempsum += feedForwardTimeList.get(j);
            }
            feedForwardTime = tempsum / feedForwardTimeList.size();
            feedForwardTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = calcCostFunctionTimeList.size(); j < n; j++){
                tempsum += calcCostFunctionTimeList.get(j);
            }
            calcCostFunctionTime = tempsum / calcCostFunctionTimeList.size();
            calcCostFunctionTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = determineResultsTimeList.size(); j < n; j++){
                tempsum += determineResultsTimeList.get(j);
            }
            determineResultsTime = tempsum / determineResultsTimeList.size();
            determineResultsTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = backpropagateTimeList.size(); j < n; j++){
                tempsum += backpropagateTimeList.get(j);
            }
            backpropagateTime = tempsum / backpropagateTimeList.size();
            backpropagateTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = storeGradientTimeList.size(); j < n; j++){
                tempsum += storeGradientTimeList.get(j);
            }
            storeGradientTime = tempsum / storeGradientTimeList.size();
            storeGradientTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = calcAllTimeList.size(); j < n; j++){
                tempsum += calcAllTimeList.get(j);
            }
            calcAllTime = tempsum / calcAllTimeList.size();
            calcAllTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = updateAllTimeList.size(); j < n; j++){
                tempsum += updateAllTimeList.get(j);
            }
            updateAllTime = tempsum / updateAllTimeList.size();
            updateAllTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = clearResultsTimeList.size(); j < n; j++){
                tempsum += clearResultsTimeList.get(j);
            }
            clearResultsTime = tempsum / clearResultsTimeList.size();
            clearResultsTimeList.clear();

            tempsum = 0;
            for (int j = 0, n = clearStorageTimeList.size(); j < n; j++){
                tempsum += clearStorageTimeList.get(j);
            }
            clearStorageTime = tempsum / clearStorageTimeList.size();
            clearStorageTimeList.clear();

            System.out.println("Init. Time: " + (initEndTime - initTime));
            System.out.println("Shuffle Time: " + (shuffleEndTime - shuffleTime));
            System.out.println("Avg image to array time: " + imageToArrayTime);
            System.out.println("Avg label to expected values time: " + labelToExpectedValuesTime);
            System.out.println("Avg set inputs time: " + setInputsTime);
            System.out.println("Avg feed forward time: " + feedForwardTime);
            System.out.println("Avg calc cost function time: " + calcCostFunctionTime);
            System.out.println("Avg determine Results time: " + determineResultsTime);
            System.out.println("Avg backpropagate time: " + backpropagateTime);
            System.out.println("Avg store gradient time: " + storeGradientTime);
            System.out.println("Calc all time: " + calcAllTime);
            System.out.println("Update all time: " + updateAllTime);
            System.out.println("Clear results time: " + clearResultsTime);
            System.out.println("Clear storage time: " + clearStorageTime);

        }
        final double endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) );
        System.out.println("Number of Epochs : " + numberOfEpochs);
        System.out.println("Learning Rate : " + learningRate);
    }

    //converts a 28 by 28 array to a 784 by 1 array to input
    public static int[] mnistImageToArray(int[][] image){
        int count = 0;
        int[] output = new int[784];
        //iterates 28 times
        for (int i = 0; i < image.length; i++){
            //iterates 28 times
            for (int j = 0; j < image[i].length; j++){
                output[count] = image[i][j];
                count++;
            }
        }
        return output;
    }

    public static String mnistImageToString(int[][] image){
        String output = "";
        for (int i = 0; i < image.length; i++){
            for (int j = 0; j < image[i].length; j++){
                output += image[i][j];
                output += "\t";
            }
            output += "\n";
        }
        return output;
    }

    public static double[] mnistLabelToExpectedValues(int label){
        double[] output = new double[10];
        for (int i = 0; i < output.length; i++){
            output[i] = 0;
        }
        output[label] = 1;
        return output;
    }

    static class ImageObject {
        int label;
        int[][] image;
        public ImageObject(int plabel, int[][] pimage){
            label = plabel;
            image = pimage;
        }
        public int getLabel(){
            return label;
        }
        public int[][] getImage(){
            return image;
        }
    }
}



