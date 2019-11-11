package flib.ij.celldetection;

import ij.ImagePlus;
import java.lang.Math;
import java.util.Arrays;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.ij.celldetection.RoundnessLevels;

public class RoundnessEval {
	private RoundnessLevels RL;
	private int nclust,w,h;
	private int[] n;
	private int[][] size;
	private double[][] rg;
	
	public RoundnessEval(final ImagePlus imp, int pps, int nclust, int maxit){
		this.nclust = nclust;
		this.RL = new RoundnessLevels(imp,pps,nclust,maxit);
		this.size = this.RL.getSizes();
		this.n = new int[this.nclust-1];
		for (int i=0; i<this.nclust-1; i++){
			this.n[i] = this.size[i].length;
		}
		this.rg = this.RL.getRG();
	}
	
	public ImagePlus getLevelImage(int m){
		return this.RL.getBinaryStack(m);
	}
	
	public RoundnessLevels getRoundnessLevels(){
		return this.RL;
	}
	
	public double[] averageRoundness(){
		double[] rdn = new double[this.nclust-1];
		for (int i=0; i<this.nclust-1; i++){
			rdn[i] = VectorFun.sum(VectorFun.mult(VectorFun.div(VectorConv.int2double(this.size[i]),VectorFun.add(this.rg[i],0.166)),1/(2*Math.PI)))/this.size[i].length;
		}
		return rdn;
	}
	
	public double[] weightAverageRoundness(){
		double[] rdn = new double[this.nclust-1];
		for (int i=0; i<this.nclust-1; i++){
			rdn[i] = VectorFun.sum(VectorFun.mult(VectorFun.mult(VectorFun.div(VectorConv.int2double(this.size[i]),VectorFun.add(this.rg[i],0.166)),1/(2*Math.PI)),VectorConv.int2double(this.size[i])))/VectorFun.sum(VectorConv.int2double(this.size[i]));
		}
		return rdn;
	}
	
	public double[] logWeightAverageRoundness(){
		double[] rdn = new double[this.nclust-1];
		for (int i=0; i<this.nclust-1; i++){
			rdn[i] = VectorFun.sum(VectorFun.mult(VectorFun.mult(VectorFun.div(VectorConv.int2double(this.size[i]),VectorFun.add(this.rg[i],0.166)),1/(2*Math.PI)),VectorFun.log(VectorFun.add(VectorConv.int2double(this.size[i]),1))))/VectorFun.sum(VectorFun.log(VectorFun.add(VectorConv.int2double(this.size[i]),1)));
		}
		return rdn;
	}
	
	public double[] powWeightAverageRoundness(double p){
		double[] rdn = new double[this.nclust-1];
		for (int i=0; i<this.nclust-1; i++){
			rdn[i] = VectorFun.sum(VectorFun.mult(VectorFun.mult(VectorFun.div(VectorConv.int2double(this.size[i]),VectorFun.add(this.rg[i],0.166)),1/(2*Math.PI)),VectorFun.pow(VectorConv.int2double(this.size[i]),p)))/VectorFun.sum(VectorFun.pow(VectorConv.int2double(this.size[i]),p));
		}
		return rdn;
	}
	
	public double[] medianRoundness(){
		double[] rdn = new double[this.nclust-1];
		double[] temp;
		for (int i=0; i<this.nclust-1; i++){
			temp = VectorFun.mult(VectorFun.div(VectorConv.int2double(this.size[i]),VectorFun.add(this.rg[i],0.166)),1/(2*Math.PI));
			Arrays.sort(temp);
			rdn[i] = temp[(int)(temp.length/2)];
		}
		return rdn;
	}
	
	public double[] pixelMedianRoundness(){
		double[] rdn = new double[this.nclust-1];
		double[] temp, temp2;
		int count;
		for (int i=0; i<this.nclust-1; i++){
			count = 0;
			temp = VectorFun.mult(VectorFun.div(VectorConv.int2double(this.size[i]),VectorFun.add(this.rg[i],0.166)),1/(2*Math.PI));
			temp2 = new double[(int)VectorFun.sum(VectorConv.int2double(this.size[i]))];
			for (int j=0; j<this.size[i].length; j++){
				Arrays.fill(temp2,count,count+this.size[i][j],temp[j]);
				count+=this.size[i][j];
			}
			Arrays.sort(temp2);
			rdn[i] = temp2[(int)(temp.length/2)];
		}
		return rdn;
	}
}