package flib.algorithms.randomforest.splitfunctions;

import java.util.Random;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.splitfunctions.ClassificationSplit;
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;
import flib.algorithms.randomforest.splitfunctions.SplittingFunctions;

public class MixedSplit extends ClassificationSplit {
	
	@Override public boolean split(TreeNode<DecisionNode> current, final int[] points, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final int[] dim, final double[] parameters, final double[] splitpurity, int leafindex){
		double[] m = new double[3];
		m[0] = Double.MAX_VALUE;
		// bad dimension value. Needs to be overwritten
		m[1] = Double.MAX_VALUE;
		// get the unique labels
		int[] ul = SortingFunctions.uniqueLabels(labels,points);
		// find the number of classes
		int ncl = 0;
		for (int i=0; i<ul.length; i++){
			if (ncl<ul[i]){
				ncl = ul[i];
			}
		}
		ncl++;
		double[] p = new double[3];
		double w = parameters[5]+parameters[6]+parameters[7];
		p[0] = parameters[5]/w;
		p[1] = parameters[6]/w+p[0];
		p[2] = 1;
		int type = 0;
		double r = (new Random()).nextDouble();
		if (r>p[0]&&r<p[1]){
			type = 1;
		}
		else if (r>=p[1]){
			type = 2;
		}
		for (int i=0; i<dim.length; i++){
			// find all unique dimension values
			double[][] loc = SortingFunctions.labelList(ul,points,trainingset,weights,dim[i]);
			double[] g = splitMethod(loc,categorical[dim[i]],ncl,type,parameters[10]);
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

	double[] splitMethod(double[][] loc, boolean cat, int num_class, int type, double alpha){
		if (type==0){
			return SplittingFunctions.gini(loc,cat,num_class,1);
		}
		else if (type==1){
			int l = loc.length/3;
			if (!cat){
				l--;
			}
			if (l>0){
				l = (new Random()).nextInt(l);
			}
			return SplittingFunctions.randSplit(loc,cat,num_class,l);
		}
		else {
			int l = loc.length/3;
			if (!cat){
				l--;
			}
			if (l>0){
				l = (new Random()).nextInt(l);
			}
			return SplittingFunctions.gini_flip(loc,cat,num_class,l);
		}
	}
}