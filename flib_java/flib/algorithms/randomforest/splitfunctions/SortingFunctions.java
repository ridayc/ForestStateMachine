package flib.algorithms.randomforest.splitfunctions;

import flib.math.RankSort;

public class SortingFunctions {
	public static int[] uniqueLabels(final double[] labels){
		// sort according to the labels
		RankSort r = new RankSort(labels);
		// get the sorted values
		double[] s = r.getSorted();
		// order of the original indices
		int[] o = r.getRank();
		// create a vector to store the integer labels in
		int[] l = new int[s.length];
		// count the number of unique labels
		int count = 0;
		l[o[0]] = count;
		for (int i=1; i<s.length; i++){
			if(s[i]>s[i-1]){
				count++;
			}
			l[o[i]] = count;
		}
		return l;
	}
	
	public static int[] uniqueLabels(final double[] labels, final int[] points){
		double[] l = new double[points.length];
		for (int i=0; i<points.length; i++){
			l[i] = labels[points[i]];
		}
		return uniqueLabels(l);
	}
	
	public static double[][] labelList(final int[] labels, final double[] val, final double[] weights){
		// sort according to the values
		RankSort r = new RankSort(val);
		// get the sorted values
		double[] s = r.getSorted();
		// order of the original indices
		int[] o = r.getRank();
		// number of unique values in the value list
		int count = 1;
		for (int i=1; i<s.length; i++){
			if(s[i]>s[i-1]){
				count++;
			}
		}
		int n = count;
		// counter for the number of items per value
		int[] counter = new int[n];
		// contains all weighted and sorted values according to unique labels
		double[][] loc = new double[3*n][];
		count = 0;
		int count2 = 1;
		loc[count] = new double[2];
		loc[count][0] = 0;
		loc[count][1] = s[0];
		for (int i=1; i<s.length; i++){
			if(s[i]>s[i-1]){
				counter[count] = count2;
				count++;
				count2 = 1;
				loc[count] = new double[2];
				loc[count][0] = i;
				loc[count][1] = s[i];
			}
			else {
				count2++;
			}
		}
		counter[count] = count2;
		// looping over all unique values
		for (int i=0; i<n; i++){
			// if there is only a single value we only need to list a single class
			// and the weight of the point at that location
			if(counter[i]==1){
				// present class
				loc[n+i] = new double[1];
				loc[n+i][0] = labels[o[(int)loc[i][0]]];
				// class weight
				loc[2*n+i] = new double[1];
				loc[2*n+i][0] = weights[o[(int)loc[i][0]]];
			}
			// otherwise we need to go through the labels of the single value
			else {
				// there is only one single class
				boolean sc = true;
				// the first seen class value
				double v = labels[o[(int)loc[i][0]]];
				for (int j=0; j<counter[i]; j++){
					if (v!=labels[o[(int)loc[i][0]+j]]){
						sc = false;
						break;
					}
				}
				// again the case of a single class
				if (sc){
					// present class
					loc[n+i] = new double[1];
					loc[n+i][0] = labels[o[(int)loc[i][0]]];
					// class weight
					loc[2*n+i] = new double[1];
					for (int j=0; j<counter[i]; j++){
						loc[2*n+i][0]+=weights[o[(int)loc[i][0]+j]];
					}
				}
				// otherwise there's quite a bit of work to be done for this value...
				else {
					double[] l = new double[counter[i]];
					double[] w = new double[counter[i]];
					for (int j=0; j<counter[i]; j++){
						l[j] = labels[o[(int)loc[i][0]+j]];
						w[j] = weights[o[(int)loc[i][0]+j]];
					}
					RankSort r2 = new RankSort(l);
					// get the sorted values
					double[] s2 = r2.getSorted();
					// order of the original indices
					int[] o2 = r2.getRank();
					// number of class labels for this specific value
					count = 1;
					for (int j=1; j<s2.length; j++){
						if(s2[j]>s2[j-1]){
							count++;
						}
					}
					loc[n+i] = new double[count];
					loc[2*n+i] = new double[count];
					count = 0;
					loc[n+i][count] = l[o2[0]];
					loc[2*n+i][count]+=w[o2[0]];
					for (int j=1; j<s2.length; j++){
						if(s2[j]>s2[j-1]){
							count++;
							loc[n+i][count] = l[o2[j]];
						}
						loc[2*n+i][count]+=w[o2[j]];
					}
				}
			}
		}
		return loc;
	}
	
	public static double[][] labelList(final int[] labels, final int[] points, final double[][] trainingset, final double[] weights, int dim){
		double[] val = new double[points.length];
		double[] w = new double[points.length];
		for (int i=0; i<points.length; i++){
			val[i] = trainingset[points[i]][dim];
			w[i] = weights[points[i]];
		}
		return labelList(labels,val,w);
	}
	
	public static double[][] regressionLabels(final double[] labels, final double[] val, final double[] weights){
		// sort according to the values
		RankSort r = new RankSort(val);
		// get the sorted values
		double[] s = r.getSorted();
		// order of the original indices
		int[] o = r.getRank();
		// number of unique values in the value list
		int count = 1;
		for (int i=1; i<s.length; i++){
			if(s[i]>s[i-1]){
				count++;
			}
		}
		int n = count;
		// counter for the number of classes per value
		int[] counter = new int[count];
		// contains all weighted and sorted values according to unique labels
		double[][] loc = new double[3*n][];
		count = 0;
		int count2 = 1;
		loc[count] = new double[2];
		loc[count][0] = 0;
		loc[count][1] = s[0];
		for (int i=1; i<s.length; i++){
			if(s[i]>s[i-1]){
				counter[count] = count2;
				count++;
				count2 = 1;
				loc[count] = new double[2];
				loc[count][0] = i;
				loc[count][1] = s[i];
			}
			else {
				count2++;
			}
		}
		counter[n-1] = count2;
		// looping over all unique values
		count2 = 0;
		for (int i=0; i<n; i++){
			loc[i+n] = new double[counter[i]];
			loc[i+2*n] = new double[counter[i]];
			for (int j=0; j<counter[i]; j++){
				loc[i+n][j] = labels[o[(int)loc[i][0]+j]];
				loc[i+2*n][j] = weights[o[(int)loc[i][0]+j]];
			}
		}
		return loc;
	}
	
	public static double[][] regressionLabels(final double[] labels, final int[] points, final double[][] trainingset, final double[] weights, int dim){
		double[] val = new double[points.length];
		double[] w = new double[points.length];
		double[] l = new double[points.length];
		for (int i=0; i<val.length; i++){
			val[i] = trainingset[points[i]][dim];
			w[i] = weights[points[i]];
			l[i] = labels[points[i]];
		}
		return regressionLabels(l,val,w);
	}
	
	public static double[][] convertLabels2(final double[][] labels, final int[] cc){
		// take a partioned set of labels and convert all labels and weights
		// to the binary labels
		int l = labels.length/3;
		double[][] clabels = new double[2*l][2];
		for (int i=0; i<l; i++){
			clabels[i] = labels[i].clone();
			// go through the previous labels
			for (int j=0; j<labels[l+i].length; j++){
				clabels[i+l][cc[(int)labels[l+i][j]]]+=labels[2*l+i][j];
			}
		}
		return clabels;
	}
	
	public static double[][] convertLabels(final double[][] labels, final int[] cc){
		int l = labels.length/3;
		double[][] clabels = new double[3*l][2];
		for (int i=0; i<l; i++){
			clabels[i] = labels[i].clone();
			// go through the previous labels
			for (int j=0; j<labels[l+i].length; j++){
				clabels[i+l][1] = 1;
				clabels[i+2*l][cc[(int)labels[l+i][j]]]+=labels[2*l+i][j];
			}
			if (clabels[i+2*l][1]==0){
				clabels[i+l] = new double[1];
				clabels[i+l][0] = 0;
				double a = clabels[i+2*l][0];
				clabels[i+2*l] = new double[1];
				clabels[i+2*l] [0] = a;
			}
			if (clabels[i+2*l][0]==0){
				clabels[i+l] = new double[1];
				clabels[i+l][0] = 1;
				double a = clabels[i+2*l][1];
				clabels[i+2*l] = new double[1];
				clabels[i+2*l] [0] = a;
			}
		}
		return clabels;
	}
}