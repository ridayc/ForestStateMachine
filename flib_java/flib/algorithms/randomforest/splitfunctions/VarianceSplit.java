package flib.algorithms.randomforest.splitfunctions;

import java.util.ArrayList;
import java.util.Random;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.lang.Math;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.DecisionTree;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.SortPair2;
import flib.math.RankSort;
import flib.math.random.Shuffle;

public class VarianceSplit extends SplitFunction {
	
	@Override protected boolean checkLeaf(final int[] points, final double[] labels, final double[] weights, final double[] parameters, final double[] splitpurity, int depth){
		// check if any of the conditions of a leaf node are fulfilled
		// splitpurity and labels are unused here
		double wt = 0;
		for (int i=0; i<points.length; i++){
			wt+=weights[points[i]];
		}
		// stopping conditions
		if (depth>=parameters[2]||wt<=parameters[3]){
			return true;
		}
		else {
			return false;
		}
	}
	
	@Override protected void setLeaf(TreeNode<DecisionNode> current, final int[] points, final double[] labels, final double[] weights, final boolean[] categorical, final double[] parameters, int leafindex){
		// prepare the node content
		DecisionNode leaf = new DecisionNode();
		// do nothing for the content?
		leaf.setLeaf(new double[1]);
		double wt = 0;
		for (int i=0; i<points.length; i++){
			wt+=weights[points[i]];
		}
		leaf.setWeight(wt);
		leaf.setLeafIndex(leafindex);
		leaf.setPoints(points);
		// set these purity fractions in the current node
		current.setData(leaf);
	}
	
	@Override public boolean split(TreeNode<DecisionNode> current, final int[] points, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final int[] dim, final double[] parameters, final double[] splitpurity, int leafindex){
		double[] m = new double[3];
		m[0] = 0;
		boolean pure = false;
		Random rng = new Random();
		for (int i=0; i<dim.length; i++){
			// find all unique dimension values
			double[] ts = new double[points.length];
			for (int j=0; j<ts.length; j++){
				ts[j] = trainingset[points[j]][dim[i]];
			}
			RankSort r = new RankSort(ts);
			double[] s = r.getSorted();
			int[] o = r.getRank();
			double[] g = new double[2];
			g[0] = 0;
			double ml = 0, mr = 0, wl = 0, wr = 0, m2l = 0, m2r = 0;
			// calculate the total weighted mean and variance for both sides
			for (int j=0; j<s.length; j++){
				double w = weights[points[o[j]]];
				wr+=w;
				double a = s[j];
				mr+=a*w;
				m2r+=a*a*w;
			}
			//double y = wr^2*(m2r/wr-mr/wr*mr/wr);
			double y = (m2r*wr-mr*mr);
			if (y==0){
				pure = true;
			}
			// calculate intermediate variance values
			for (int j=0; j<s.length-1; j++){
				double w = weights[points[o[j]]];
				wl+=w;
				wr-=w;
				double a = s[j];
				ml+=a*w;
				mr-=a*w;
				m2l+=a*a*w;
				m2r-=a*a*w;
				// calculate the double sided weighted variance
				//double x = (m2l/wl-ml/wl*ml/wl)*wl+(m2r/wr-mr/wr*mr/wr)*wr;
				double x = (m2l*wl-ml*ml)+(m2r*wr-mr*mr);
				if (y-x>g[0]){
					g[0] = y-x;
					g[1] = (a+s[j+1])*0.5;
				}
			}
			// compare the splits of the different dimensions
			// choose the split which maximizes the difference in variance
			if (g[0]>m[0]){
				m[0] = g[0];
				m[1] = dim[i];
				int a = rng.nextInt(s.length-1);
				m[2] = (s[a]+s[a+1])*0.5;
			}
			if (m[0]>0){
				pure = false;
			}
		}
		if (pure){
			this.setLeaf(current,points,labels,weights,categorical,parameters,leafindex);
			return false;
		}
		// generate the branching
		DecisionNode branch = new DecisionNode();
		branch.setDim((int)m[1]);
		branch.setCategorical(categorical[(int)m[1]]);
		branch.setSplit(m[2]);
		current.setData(branch);
		current.setLeft(new TreeNode<DecisionNode>());
		current.setRight(new TreeNode<DecisionNode>());
		return true;
	}
	
	@Override public double[] oobe(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters){
		return new double[1];
	}
	
	@Override public double[][] oobeConvergence(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters){
		return new double[1][1];
	}
	
	@Override public double[][] variableImportance(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters){
		return new double[1][1];
	}
	
	@Override public double predict(final double[] point, final double[] tree_weights, ArrayList<DecisionTree> forest){
		double[] votes = applyForest(point,tree_weights,forest);
		return votes[0];
	}
	
	public double[] applyForest(final double[] point, final int[] missing, final ArrayList<DecisionTree> forest, int num_class){
		return new double[0];
	}
	
	@Override public double[] sample(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels, final double[] weights){
		Random rng = new Random();
		int n = trainingset.length;
		int dim = trainingset[0].length;
		int l = missing.length;
		double[] samp = point.clone();
		int[] randind = Shuffle.randPerm(l);
		for (int i=0; i<l; i++){
			int[] miss = new int[l-i];
			for (int j=0; j<l-i;j++){
				miss[j] = missing[randind[i+j]];
			}
			Arrays.sort(miss);
			TreeSet<SortPair2> list = new TreeSet<SortPair2>();
			for (int j=0; j<forest.size(); j++){
				forest.get(j).neighbors(samp,miss,list,weights);
			}
			int[] neighbors = new int[list.size()];
			double[] w = new double[list.size()];
			Iterator<SortPair2> itr = list.iterator();
			SortPair2 sp = itr.next();
			neighbors[0] = (int)sp.getValue();
			double a = label[0]-labels[neighbors[0]];
			w[0] = sp.getOriginalIndex();
			int counter = 1;
			while (itr.hasNext()){
				sp = itr.next();
				neighbors[counter] = (int)sp.getValue();
				w[counter] = w[counter-1]+sp.getOriginalIndex();
				counter++;
			}
			int b = VectorFun.binarySearch(w,rng.nextDouble()*(w[counter-1]));
			samp[missing[randind[i]]] = trainingset[neighbors[b]][missing[randind[i]]];
		}
		return samp;
	}
	
	public double[] nearestNeighbor(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels){
		TreeSet<SortPair2> list = new TreeSet<SortPair2>();
		for (int i=0; i<forest.size(); i++){
			forest.get(i).neighbors(point,missing,list);
		}
		double[] m = new double[2];
		Iterator<SortPair2> itr = list.iterator();
		while (itr.hasNext()){
			SortPair2 sp = itr.next();
			double b = sp.getOriginalIndex();
			if (b>m[1]){
				m[1] = b;
				m[0] = sp.getValue();
			}
		}
		return trainingset[(int)m[0]].clone();
	}
	
	public double[] sampledNeighbor(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels, int type, double p){
		TreeSet<SortPair2> list = new TreeSet<SortPair2>();
		for (int i=0; i<forest.size(); i++){
			forest.get(i).neighbors(point,missing,list);
		}
		double[] cs = new double[list.size()];
		int[] o = new int[list.size()];
		Iterator<SortPair2> itr = list.iterator();
		SortPair2 sp = itr.next();
		double b = sp.getOriginalIndex();
		cs[0] = Math.pow(b,p);
		o[0] = (int)sp.getValue();
		int counter = 1;
		while (itr.hasNext()){
			sp = itr.next();
			b = sp.getOriginalIndex();
			if (type%2==0){
				cs[counter] = cs[counter-1]+Math.pow(b,p);
			}
			else {
				cs[counter] = b;
			}
			o[counter] = (int)sp.getValue();
			counter++;
		}
		int a2 = 0;
		if (type%2==0){
			a2 = o[VectorFun.binarySearch(cs,(new Random()).nextDouble()*cs[counter-1])];
		}
		else {
			int[] o2 = (new RankSort(cs)).getRank();
			int b2 = (int)p;
			if (b2>o2.length){
				b2 = o2.length;
			}
			a2 = o[o2[o2.length-1-(new Random()).nextInt(b2)]];
		}
		if (type/2==0){
			return trainingset[a2].clone();
		}
		else {
			double[] temp = new double[1];
			temp[0] = a2;
			return temp;
		}
	}	
}