import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class NeuralNet {
	double correct = 0;
	NNLayer inputLayer;
	NNLayer [] hiddenLayerList;
	NNLayer outLayer;
	List<String[]> outputList;
	List<String[]> weightList;
	double learningRate;
	int inputSize;
	int hiddenLayerCount; 
	int [] hiddenLayerSize;
	int outputSize;
	String[] outputArr;
	public NeuralNet(int inputSize, int hiddenLayerCount, int [] hiddenLayerSize, int outputSize, String[] outputArr, double learningRate) {
		this.inputSize = inputSize;
		this.hiddenLayerCount = hiddenLayerCount; //number of layers
		this.hiddenLayerSize = hiddenLayerSize;//number of neurons iteratively per hidden layer
		this.outputSize = outputSize;//neuron count
		this.outputArr = outputArr;//string outputs
		this.learningRate = learningRate;
		NetMake();//initializes all the parts
	}
	
	
	public void Guess(double inputData[], String goal) {//prints the test # and the results/guess
		//clearLayerValues();
		inputLayer.BeginInput(inputData);
		System.out.println("Test Goal:"+goal);
		String result = outLayer.Answer();
		System.out.println();
		if(goal.equals(result)){
			correct++;
			System.out.println("Test Guess:"+result+" | Correct");
		}
		else {
			System.out.println("Test Guess:"+result+" | Wrong");
		}
		//outLayer.print();
	}
	

	
	public void Learn(double inputData[], String goal) {
		
		
		//System.out.println("Sample Pre-Run Weight");
		//printLayers();

		inputLayer.BeginInput(inputData);
		for(NNNode n : outLayer.nodes) {
			double y;
			if(n.goal.equals(goal)) {
				y = 1;
			}else {
				y = 0;
			}
			n.gradient = (n.value)*(1-n.value)*(n.value-y);
		}
		backProp();
		clearLayerValues();
		//System.out.println("");
		//System.out.println("Sample After-Run Weight");
		//printLayers();
	}
	
	
	private void clearLayerValues() {
		inputLayer.clearValues();
		for(int i = 0; i<hiddenLayerCount; i++) {
			hiddenLayerList[i].clearValues();
		}
		outLayer.clearValues();
	}

	
	
	private void backProp() {//we identify/calculate all the required data by using backpropagation
		
		for(int i = (hiddenLayerCount-1); i>=0; i--) {
			for(NNNode n : hiddenLayerList[i].nodes) {
				double sigma = 0;
				for(NNConnect con : n.connections) {
					sigma += (con.weight)*(con.to.gradient);
				}
				n.gradient = (n.value)*(1-n.value)*(sigma);
			}
		}
		
		//adjusting weights
		for(NNNode n : inputLayer.nodes) {
			for(NNConnect con : n.connections) {
				double grad = con.to.gradient;
				double value = n.value;
				double changeW = (-learningRate)*(value)*(grad);
				double changeB = (-learningRate)*(grad);
				con.weight+=changeW;
				inputLayer.bias.value+=changeB;
			}
		}
		for(int i = 0; i>hiddenLayerCount; i++) {
			for(NNNode n : hiddenLayerList[i].nodes) {
				for(NNConnect con : n.connections) {
					double grad = con.to.gradient;
					double value = n.value;
					double changeW = (-learningRate)*(value)*(grad);
					double changeB = (-learningRate)*(grad);
					con.weight+=changeW;
					hiddenLayerList[i].bias.value+=changeB;
				}
			}
		}
	}
	
	
	private void makeHiddenLayers() {
		hiddenLayerList = new NNLayer[hiddenLayerCount];
		System.out.println(hiddenLayerCount);
		for(int i = 0; i<hiddenLayerCount; i++) {
			//System.out.println(hiddenLayerSize[i]);
			//System.out.println(i+2);
			NNLayer hiddenLayer = new NNLayer(1,hiddenLayerSize[i],(i+2));
			hiddenLayerList[i] = hiddenLayer;
			if(i != 0) {
				hiddenLayerList[i-1].next = hiddenLayerList[i];
			}
		}
	}

	
	public void NetMake() {//Creates the neural net's structure and assigns responsibility based on parameters in layers
		//(int function,int size, int layer)
		// function key  0=input, 1=hidden, 2=out
		inputLayer = new NNLayer(0,inputSize,1);
		makeHiddenLayers();
		outLayer = new NNLayer(2,outputSize,hiddenLayerCount+2);
		outLayer.type = outputArr;
		
		//connects the layers
		inputLayer.next = hiddenLayerList[0];
		hiddenLayerList[hiddenLayerCount-1].next = outLayer;
		
		
		//indexes nodes to better track and debug issues
		int indexer = 0;
		for(NNNode n : inputLayer.nodes) {
			n.globalIndex = indexer;
			indexer++;
		}
		for(int i = 0; i<hiddenLayerCount; i++) {
			for(NNNode n : hiddenLayerList[i].nodes) {
				n.globalIndex = indexer;
				indexer++;
			}
		}
		for(NNNode n : outLayer.nodes) {
			n.globalIndex = indexer;
			indexer++;
		}
		
		//assigns other functionality depending on function parameter 
		inputLayer.AddFunction();
		for(int i = 0; i<hiddenLayerCount; i++) {
			hiddenLayerList[i].AddFunction();
		}
		outLayer.AddFunction();
		
	}
	
	private void printLayers() {
		inputLayer.printWeights();
		for(int i = 0; i<hiddenLayerCount; i++) {
			hiddenLayerList[i].printWeights();
		}
	}
	
	public void SaveNN() {
		inputLayer.saveWeights();
		inputLayer.saveBias();
		for(int i = 0; i<hiddenLayerCount; i++) {
			hiddenLayerList[i].saveWeights();
			hiddenLayerList[i].saveBias();
		}
	}
	
	public void Load() {
		weightList = LoadWeights("src/weights.txt",5);
		for(String[] weights : weightList) {
			Integer layer = null;
			boolean bias = false;
			Integer n1 = null;
			Integer n2 = null;
			double weightOfBias = 0;
			double weight = 0;
			for(int i = 0; i<5; i++) {
				if(i == 0) {
					layer = Integer.parseInt(weights[i]);
				}
				else if(i == 1) {
					bias = Boolean.parseBoolean(weights[i]);
				}
				else if(i == 2 && !bias) {
					n1 = Integer.parseInt(weights[i]);
				}
				else if(i == 2 && bias) {
					weightOfBias = Double.parseDouble(weights[i]);
				}
				else if(i == 3) {
					n2 = Integer.parseInt(weights[i]);
				}
				else if(i == 4) {
					weight = Double.parseDouble(weights[i]);
				}
			}
			if(bias) {
				setBias(layer,n2,weight,weightOfBias);
			}
			else{
				setWeight(layer,n1,n2,weight);
			}
		}
	}
	
	public void setWeight(int layer,int n1, int n2, double weight) {
		if(layer == 1) {
			inputLayer.LoadWeights(n1,n2, weight);
		}
		else {
			hiddenLayerList[layer-2].LoadWeights(n1, n2, weight);
		}
	}

	public void setBias(int layer, int n2, double value, double weightOfBias) {
		if(layer == 1) {
			inputLayer.LoadBias(layer,n2, value,weightOfBias);
		}
		else {
			hiddenLayerList[layer-2].LoadBias(layer,n2, value,weightOfBias);
		}
	}


	
	public static List<String[]> LoadWeights(String fileName, int param) {//Simple file parser, converts to the list of string arrays
    	List<String[]> myList = new ArrayList<>();
		String line = null;
        try {
            FileReader fileReader = new FileReader(fileName);

            BufferedReader bufferedReader = new BufferedReader(fileReader);

            while((line = bufferedReader.readLine()) != null) {
            	int lineSZ = line.length();
            	int index = 0;
            	String [] data = new String[param];
            	StringBuilder inputTemp = new StringBuilder();
                for(int i = 0; i<lineSZ; i++) {
                	char cTemp = line.charAt(i);
                	if(i+1 == lineSZ) {
                		inputTemp.append(cTemp);
                		data[index] = inputTemp.toString();
                	}
                	else if(cTemp == ',') {
                		data[index] = inputTemp.toString();
                		inputTemp.setLength(0);
                		index++;
                	}
                	else {
                		inputTemp.append(cTemp);
                	}
                }
                myList.add(data);
            }

            bufferedReader.close();
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                "Unable to open file "+fileName);
        }
        catch(IOException ex) {
            System.out.println(
                "Error reading file "+fileName);
        }

        return myList;
	}

}
