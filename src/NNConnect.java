public class NNConnect {
	NNNode to;
	double weight;
	int n1;
	int n2;
	boolean bias = false;
	int layer;




	public NNConnect(NNNode to) {
		//stores the node that you are going to, and the weight.
		//This is because the parent node keeps track of all its connections.
		this.to = to;

	}
}