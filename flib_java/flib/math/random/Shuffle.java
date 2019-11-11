package flib.math.random;

import java.util.Random;
import java.util.ArrayList;
import flib.math.VectorFun;

// This class is used to generate a random permutation array of integers from 1 through n
public class Shuffle{
	public static int[] randIndices(int l, int n){
		Random random = new Random();
		int[] ind = new int[l];
		for (int i=0; i<l; i++){
			ind[i] = random.nextInt(n);
		}
		return ind;
	}
	
	public static int[] randPerm(int n, Random random){
		int r,temp;
		int[] ind = new int[n];
		for (int i=0; i<n; i++){
			ind[i] = i;
		}
		for (int i=0; i<n; i++){
			r = i+random.nextInt(n-i);
			temp = ind[i];
			ind[i] = ind[r];
			ind[r] = temp;
		}
		return ind;
	}
	
	public static int[] randPerm(int n){
		Random random = new Random();
		return randPerm(n,random);
	}
	
	// weighted shuffle
	public static int[] randPerm(final double[] x){
		int n = x.length;
		Random random = new Random();
		int[] ind = new int[n];
		TreeNode tn = new TreeNode(x);
		for (int i=0; i<n; i++){
			ind[i] = tn.nextInd(random);
		}
		return ind;
	}
	
	// Tree Node class used to build the binary tree for the weighted shuffle
	private static class TreeNode {
		private double weight;
		private int ind;
		private ArrayList<TreeNode> children = new ArrayList<TreeNode>();
		
		// a recursive splitting function
		private void split(int locl, int locr, final double[] x){
			// if we arrive at a leaf
			if (locr-locl==0){
				this.weight = x[locl];
				this.ind = locl;
			}
			// otherwise if we need to split further
			else {
				int locn = (locr-locl)/2+locl;
				this.children.add(new TreeNode());
				this.children.get(0).split(locl,locn,x);
				this.children.add(new TreeNode());
				this.children.get(1).split(locn+1,locr,x);
				this.weight = this.getChildren().get(0).getWeight()+this.getChildren().get(1).getWeight();
			}
		}
		
		public TreeNode() {
		}
		
		// build the whole tree with a given set of weights
		public TreeNode(final double[] weights){
			this.split(0,weights.length-1,weights);
		}
		
		public int getInd(){
			return this.ind;
		}
		
		public double getWeight(){
			return this.weight;
		}
		
		public void setInd(int ind){
			this.ind = ind;
		}
		
		public void setWeight(double weight){
			this.weight = weight;
		}	
		
		private void addChild(TreeNode a){
			this.children.add(a);
		}
		
		public ArrayList<TreeNode> getChildren(){
			return this.children;
		}
		
		// find the next index according to the weight in the tree
		public int nextInd(Random rand){
			ArrayList<Boolean> path = new ArrayList<Boolean>();
			TreeNode temp = this;
			double w;
			while (!temp.getChildren().isEmpty()){
				w = temp.getChildren().get(0).getWeight()+temp.getChildren().get(1).getWeight();
				if (rand.nextDouble()<temp.getChildren().get(0).getWeight()/w){
					path.add(true);
					temp = temp.getChildren().get(0);
				}
				else {
					path.add(false);
					temp= temp.getChildren().get(1);
				}
				
			}
			// here we update the weights in the tree
			// by running through the tree once more
			w = temp.getWeight();
			temp = this;
			temp.setWeight(temp.getWeight()-w);
			for (int i=0; i<path.size(); i++){
				if (path.get(i)){
					temp = temp.getChildren().get(0);
				}
				else {
					temp = temp.getChildren().get(1);
				}
				temp.setWeight(temp.getWeight()-w);
			}			
			return temp.getInd();
		}
	}
}