package flib.math.random;

import java.util.Random;
import java.util.ArrayList;
import flib.math.SortPair;
import flib.math.WRBTree;
//import flib.math.random.Shuffle;

// This class is used to generate a random permutation array of integers from 1 through n
// the shuffler is initiated and can be used for quick draws later on
public class Shuffler implements
java.io.Serializable {
	// the weighted R-B-Tree
	private WRBTree root;
	// an internal random number generator
	private Random rng;
	// total weight of the tree
	private double wt;
	// total number of entries in the tree
	private int s;
	
	// create a balanced tree containing the weights for shuffling
	public Shuffler(final double[] x){
		root = new WRBTree();
		rng = new Random();
		wt = 0;
		s = 0;
		// insert all the weighted entries
		// keep the original sorting indices
		// for test testing
		//int[] r = (new Shuffle()).randPerm(x.length);
		for (int i=0; i<x.length; i++){
			s++;
			wt+=x[i];
			root.insert(new SortPair(x[i],i));
			//root.insert(new SortPair(x[r[i]],i));
		}
	}
	
	// draw n samples according to a weighted shuffle
	public int[] randPerm(int n){
		// we have to delete and reinsert all values drawn for the random permutation
		// the values are stored in a temporary SortPair list
		ArrayList<SortPair> temp = new ArrayList<SortPair>();
		// the list of indices
		int[] ind = new int[n];
		// read out and delete n samples without replacement
		for (int i=0; i<n; i++){
			// roll
			double r = rng.nextDouble()*wt;
			// find the corresponding entry inside the tree and store it
			temp.add(root.weighted_access(r));
			ind[i] = (int)temp.get(i).getOriginalIndex();
			// delete the entry from the tree for further draws
			root.delete(temp.get(i));
			wt-=temp.get(i).getValue();
		}
		// rebuild the tree so it can be used again
		for (int i=0; i<n; i++){
			root.insert(temp.get(i));
			wt+=temp.get(i).getValue();
		}
		return ind;
	}
	
	public WRBTree getRoot(){
		return root;
	}
	
	public double getWeight(){
		return wt;
	}
	
	public int getSize(){
		return s;
	}
	
	public Random getRNG(){
		return rng;
	}
}