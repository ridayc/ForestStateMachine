package flib.algorithms.randomforest.splitfunctions;

import java.util.Random;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.splitfunctions.RegressionSplit;
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;
import flib.algorithms.randomforest.splitfunctions.SplittingFunctions;

public class MixedRegression extends RegressionSplit {
	
	@Override public boolean split(TreeNode<DecisionNode> current, final int[] points, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final int[] dim, final double[] parameters, final double[] splitpurity, int leafindex){
		double[] m = new double[3];
		m[0] = Double.MAX_VALUE;
		// bad dimension value. Needs to be overwritten
		m[1] = Double.MAX_VALUE;
		double[] p = new double[3];
		double w = parameters[5]+parameters[6];
		p[0] = parameters[5]/w;
		p[1] = 1;
		int type = 0;
		double r = (new Random()).nextDouble();
		if (r>p[0]){
			type = 1;
		}
		for (int i=0; i<dim.length; i++){
			// find all unique dimension values
			double[][] loc = SortingFunctions.regressionLabels(labels,points,trainingset,weights,dim[i]);
			double[] g = splitMethod(loc,categorical[dim[i]],type,parameters[10]);
			if (g[0]<m[0]){
				m[0] = g[0];
				m[1] = dim[i];
				m[2] = g[1];
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
			//System.out.println("Non-optimal split");
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

	double[] splitMethod(double[][] loc, boolean cat, int type, double alpha){
		if (type==0){
			return SplittingFunctions.varReduc(loc,cat,1);
		}
		else{
			int l = loc.length/3;
			if (!cat){
				l--;
			}
			if (l>0){
				l = (new Random()).nextInt(l);
			}
			return SplittingFunctions.var_flip(loc,cat,l);
		}
	}
}