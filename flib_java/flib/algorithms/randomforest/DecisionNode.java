package flib.algorithms.randomforest;

public abstract class DecisionNode implements
java.io.Serializable {
	// this is specifiically only for leaf nodes. We need this to determine
	// the exact location in which data points fall
	// Each leaf in a decision tree has its own specific index
	// This is important to measure the "distance" between points
	// according to the tree structure (e.g. for proximities)
	// Equal to -1 if the current node is not a leaf node
	private int leafindex = -1;
	// this indicates the split path within the tree which a datum has to follow
	// to get to the current node
	private int nodeindex;
	// this vector contains the results for classification or regression contained in the node
	private double[] probabilities;
	
	// is the current node a leaf node?
	public boolean isLeafNode(){
		if (leafindex==-1){
			return false;
		}
		else {
			return true;
		}
	}
	// for non-leaf nodes determine if the point is on the left or right of the split
	public abstract boolean evaluateDatum(final double[] datum);
	
	public double[] getProbabilities(){
		return this.probabilities;
	}
	
	public int getLeafIndex(){
		return this.leafindex;
	}
	
	public int getNodeIndex(){
		return this.nodeindex;
	}
	
	// setter functions
	
	public void setLeaf(final double[] probabilities){
		this. probabilities = probabilities.clone();
	}
	
	public void setLeafIndex(int leafindex){
		this.leafindex = leafindex;
	}
	
	public void setNodeIndex(int nodeindex){
		this.nodeindex = nodeindex;
	}
}