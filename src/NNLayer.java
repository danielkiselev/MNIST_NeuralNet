import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class NNLayer {
	int function;//0=input, 1=hidden, 2=out
	int size;//how many nodes
	List<NNNode> nodes = new ArrayList<>();
	NNLayer next = null;
	int layer;//what layer is it?
	NNNode bias;
	String type[];
	
	public NNLayer(int function,int size, int layer){
		this.function = function;
		this.size = size;
		this.layer = layer;
		AddNode();
	}
	
	public void printWeights() {//prints the weight, showing the associated nodes
		for(NNNode local : nodes) {
			for(NNConnect con : local.connections) {
				System.out.println("local->"+local.globalIndex+" ->TO "+ con.to.globalIndex+" WEIGHT"+con.weight);
			}
		}
	}
	
	public void print() {
		for(NNNode local : nodes) {
			System.out.println(local.value);
		}
	}
	
	public String Answer() {//used for prediction checking
		NNNode max = null;
		for(NNNode local : nodes) {
			System.out.println(local.goal+"|"+local.value);
			if(max == null) {
				max = local;
			}
			if(local.value>max.value) {
				max = local;
			}
		}
		return max.goal;
	}
	
	public void clearValues() {
		for(NNNode local : nodes) {
			local.value = 0;
		}
	}
	
	
	public void AddFunction() {//used for initializing the layer
		if(function != 2) {
			bias = new NNNode(layer, true);
			bias.value = 1;
			for(NNNode out : next.nodes) {
				bias.formConnection(out);
			}
			for(NNNode local : nodes) {
				for(NNNode out : next.nodes) {
					local.formConnection(out);
				}
			}
		}else {
			
			int i = 0;
			for(NNNode local : nodes) {
				local.goal = type[i];
				i++;
			}
		}
	}
	
	public void BeginInput(double[] data) {//used for the inputLayer
		//System.out.println();
		for(NNNode local : nodes) {
			//System.out.println("index:"+local.localIndex+"value "+ data[local.localIndex]);
			local.value = data[local.localIndex];
			//System.out.println("value"+ local.value);
		}
		calcNext();
	}
	
	private void calcNext() {//Calculates the value of the connected nodes using weights.
		if(function!=2) {
			//System.out.println("next layer"+next.layer);
			for(NNNode extern : next.nodes) {
				
				double total = 0;
				for(NNNode local : nodes) {
					for(NNConnect con : local.connections) {
						if(con.to.equals(extern)) {
							//System.out.println("con weight"+ con.weight);
							//System.out.println("value"+ local.value);
							total += (con.weight*local.value);
						}
					}
				}
				for(NNConnect c : bias.connections) {
					if(c.to.equals(extern)) {
						total += (c.weight*bias.value);
					}
				}
				if(total<0) {
					total = 0;
				}

				extern.value = total;
				//System.out.println(extern.value);
				////using sigmoid function since it's good with positive data sets such as this one. 
				//System.out.println("Layer:"+(layer+1)+" |  index:"+ extern.localIndex+" |  value:"+extern.value);
			}
			next.calcNext();//goes to the next layer and does this same step
		}
	}
	
	private double sigmoid(double x) {//SIGMOID
	    return (1/( 1 + Math.pow(Math.E,(-1*x))));
	}
	
	
	private void AddNode() {//used to creating the net
		for(int i = 0; i<size; i++) {
			NNNode temp = new NNNode(i, false);
			nodes.add(temp);
		}
	}
	
	public void saveWeights() {//prints the weight, showing the associated nodes
		for(NNNode local : nodes) {
			for(NNConnect con : local.connections) {
				String data = Integer.toString(layer)+","+Boolean.toString(local.bias)+","+Integer.toString(con.n1)+","+Integer.toString(con.n2)+","+con.weight;
				Save(data);
				System.out.println("Data Node: "+data);
			}
		}
	}

	public void saveBias() {//prints the weight, showing the associated nodes
			for(NNConnect con : bias.connections) {
				String data = Integer.toString(layer)+","+Boolean.toString(bias.bias)+","+Double.toString(con.weight)+","+Integer.toString(con.n2)+","+bias.value;
				Save(data);
				System.out.println("Data Bias: "+data);
			}
	}


	private void Save(String data) {

		 String fileName = "src/weights.txt";

	        try {

	        	FileWriter fw = new FileWriter(fileName, true);
	            BufferedWriter bw = new BufferedWriter(fw);
	        	PrintWriter writer = new PrintWriter(bw);
	        	writer.println(data);
	        	writer.close();
	        }
	        catch(IOException ex) {
	            System.out.println(
	                "Error writing to file '"
	                + fileName + "'");
	        }
	    }



	public void LoadWeights(int n1, int n2, double weight) {//prints the weight, showing the associated nodes
		for(NNNode local : nodes) {
			for(NNConnect con : local.connections) {
				if(con.n1 == n1 && con.n2 == n2) {
					con.weight = weight;
					String data = Integer.toString(con.n1)+","+Integer.toString(con.n2)+","+con.weight;
					System.out.println("Data Loaded: "+data);
				}
			}
		}
	}

	public void LoadBias(int n1, int n2, double value, double weight) {//prints the weight, showing the associated nodes
			for(NNConnect con : bias.connections) {
				if(con.n2 == n2) {
					bias.value = value;
					con.weight = weight;
					String data = Integer.toString(con.layer)+","+Integer.toString(con.n2)+","+bias.value;
					System.out.println("Bias Loaded: "+data);
				}
			}
		}

	


}
