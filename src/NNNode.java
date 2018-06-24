import java.util.ArrayList;
import java.util.List;

public class NNNode {
	boolean bias = false;
	int layer;
	double gradient;
	int localIndex;
	int globalIndex;
	double value = 0;
	String goal;
	List<NNConnect> connections = new ArrayList<>();//holds the connections
	public NNNode(int localIndex, boolean bias){//index within its layer
		this.bias = bias;
		if(bias) {
			layer = localIndex;
		}
		else {
			this.localIndex = localIndex;
		}

	}


	public void formConnection(NNNode n) {//pass in the node of who you want to connect to.
		//randomly generates a starter weight
		NNConnect con = new NNConnect(n);
		if(bias) {
			con.n2 = n.globalIndex;
			con.layer = layer;
			con.bias = true;
			con.weight = Math.random()*.01;
			connections.add(con);
			//System.out.println();
		}
		else {
			con.weight = Math.random()*.01;
			con.n1 = globalIndex;
			con.n2 = n.globalIndex;
			connections.add(con);
		}
	}

}