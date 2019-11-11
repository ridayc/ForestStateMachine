package flib.algorithms.randomforest.splitfunctions;

import java.util.Random;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.splitfunctions.RegressionSplit;
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;
import flib.algorithms.randomforest.splitfunctions.SplittingFunctions;
import ij.IJ;

public class CrossingRegression extends RegressionSplit {

	protected double crossings(final double[][] loc, boolean cat){
		// the location object is build of three equally sized component
		// 1. starting location in the original sorted vector and the original value along the current dimension
		// 2. classes contained at the value
		// 3. weights of the contained classes at the specific value
		int l = loc.length/3;
		// crossing sum
		double c = 0;
		// squared difference within all values
		for (int i=0; i<l; i++){
			double m = 0;
			double m2 = 0;
			double w = 0;
			int l2 = loc[i+l].length;
			for (int j=0; j<l2; j++){
				m+=loc[i+l][j];
				m2+=loc[i+l][j]*loc[i+l][j];
				w+=loc[i+2*l][j];
			}
			c+=4*(m2/w-m/w*m/w)*(w-1);
		}
		// for non-categorical dimensions
		if (!cat){
			double m0 = 0;
			double w0 = 0;
			for (int i=0; i<loc[l].length; i++){
				m0+= loc[l][i];
				w0+= loc[2*l][i];
			}
			m0/=w0;
			for (int i=1; i<l; i++){
				double m1 = 0;
				double w1 = 0;
				for (int j=0; j<loc[i+l].length; j++){
					m1+= loc[i+l][j];
					w1+= loc[i+2*l][j];
				}
				m1/=w1;
				c+=(m1-m0)*(m1-m0);
				m0 = m1;
			}
		}
		// categorical case
		else {
			double m = 0;
			double m2 = 0;
			for (int i=0; i<l; i++){
				double m1 = 0;
				double w1 = 0;
				for (int j=0; j<loc[i+l].length; j++){
					m1+= loc[i+l][j];
					w1+= loc[i+2*l][j];
				}
				m1/=w1;
				m+=m1;
				m2+=m1*m1;
			}
			c+=(m2/(l-1)-m/(l-1)*m/(l-1))*4;
		}
		return c;
	}
	
	@Override public boolean split(TreeNode<DecisionNode> current, final int[] points, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final int[] dim, final double[] parameters, final double[] splitpurity, int leafindex){
		// go through all proposed dimensions and based on zeros crossing
		// choose one to be the split dimension
		// randomly choose a splitting approach cluster or classic
		Random rng = new Random();
		// for the dimension with the fewest zero crossings
		double[] m = new double[3];
		m[0] = Double.MAX_VALUE;
		m[1] = dim[0];
		if (rng.nextDouble()>parameters[5]){
			for (int i=0; i<dim.length; i++){
				double[][] loc = SortingFunctions.regressionLabels(labels,points,trainingset,weights,dim[i]);
				// find the values fluctuation squared sums
				double c = crossings(loc,categorical[dim[i]]);
				if (m[0]>c){
					m[0] = c;
					m[1] = dim[i];
				}
			}
			double[][] loc = SortingFunctions.regressionLabels(labels,points,trainingset,weights,(int)m[1]);
			m[2] = SplittingFunctions.varReduc(loc,categorical[(int)m[1]],parameters[10])[1];
		}
		else {
			for (int i=0; i<dim.length; i++){
				// find all unique dimension values
				double[][] loc = SortingFunctions.regressionLabels(labels,points,trainingset,weights,dim[i]);
				double[] g = SplittingFunctions.varReduc(loc,categorical[dim[i]],parameters[10]);
				if (g[0]<m[0]){
					m[0] = g[0];
					m[1] = dim[i];
					m[2] = g[1];
				}
			}
		}
		int counter = 0;
		if (!categorical[(int)m[1]]){
			for (int i=0; i<points.length; i++){
				if (trainingset[points[i]][(int)m[1]]<m[2]){
					counter++;
				}
			}
		}
		else {
			for (int i=0; i<points.length; i++){
				if (trainingset[points[i]][(int)m[1]]==m[2]){
					counter++;
				}
			}
		}
		if (counter==points.length||counter==0){
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
}