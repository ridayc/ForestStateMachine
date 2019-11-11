package flib.math;

import flib.math.SortPair;
import java.util.Arrays;

public class RankSort {
	private SortPair[] sorted;
	private int l;
	
	public RankSort(final double[] x){
		this.sort(x);
	}
	
	public RankSort(final double[] x, final int[] y){
		this.sort(x,y);
	}
	
	public RankSort(final double[] x, final double[] y){
		this.sort(x,y);
	}
	
	private void sort(final double[] x){
		l = x.length;
		sorted = new SortPair[l];
		for (int i=0; i<l; i++){
			sorted[i] = new SortPair(x[i],(double)i);
		}
		Arrays.sort(sorted);
	}
	
	private void sort(final double[] x,final int[] y){
		l = x.length;
		sorted = new SortPair[l];
		for (int i=0; i<l; i++){
			sorted[i] = new SortPair(x[i],(double)y[i]);
		}
		Arrays.sort(sorted);
	}
	
	private void sort(final double[] x,final double[] y){
		l = x.length;
		sorted = new SortPair[l];
		for (int i=0; i<l; i++){
			sorted[i] = new SortPair(x[i],y[i]);
		}
		Arrays.sort(sorted);
	}
	
	public double[] getSorted(){
		double[] s = new double[l];
		for (int i=0; i<l; i++){
			s[i] = sorted[i].getValue();
		}
		return s;
	}
	
	public int[] getRank(){
		int[] r = new int[l];
		for (int i=0; i<l; i++){
			r[i] = (int)sorted[i].getOriginalIndex();
		}
		return r;
	}
	
	public double[] getDRank(){
		double[] r = new double[l];
		for (int i=0; i<l; i++){
			r[i] = sorted[i].getOriginalIndex();
		}
		return r;
	}
}