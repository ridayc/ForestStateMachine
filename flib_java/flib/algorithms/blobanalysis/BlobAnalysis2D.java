package flib.algorithms.blobanalysis;

import java.util.ArrayList;
import flib.math.VectorFun;
import flib.math.VectorConv;

public class BlobAnalysis2D {
	private  int[][] bloblist;
	private int[] blobsize;
	private  double[][][] blobcoord;
	private double[][] blobcenters;
	private int w,n;
	
	public BlobAnalysis2D(int w, final ArrayList<ArrayList<Integer>> bloblist, double calx, double caly){
		this.n = bloblist.size();
		this.bloblist = new int[this.n][];
		this.blobsize = new int[this.n];
		this.blobcoord = new double[this.n][2][];
		for (int i=0; i<this.n; i++){
			this.bloblist[i] = new int[bloblist.get(i).size()];
			for (int j=0; j<bloblist.get(i).size(); j++){
				this.bloblist[i][j] = bloblist.get(i).get(j);
			}
			this.blobsize[i] = this.bloblist[i].length;
			this.blobcoord[i][0] = VectorFun.mult(VectorFun.mod(VectorConv.int2double(this.bloblist[i]),(double)w),calx);
			this.blobcoord[i][1] = VectorFun.mult(VectorConv.int2double(VectorConv.double2int(VectorFun.div(VectorConv.int2double(this.bloblist[i]),(double)w))),caly);
		}
		this.blobCenters();
	}
	
	public BlobAnalysis2D(int w, ArrayList<ArrayList<Integer>> bloblist){
		this(w,bloblist,1,1);
	}
	
	public void blobCenters(){
		this.blobcenters = new double[2][this.n];
		for (int i=0; i<this.n; i++){
			this.blobcenters[0][i] = VectorFun.sum(this.blobcoord[i][0])/((double)this.blobsize[i]);
			this.blobcenters[1][i] = VectorFun.sum(this.blobcoord[i][1])/((double)this.blobsize[i]);
		}
	}
	
	public double[] radiusOfGyration(){
		double[] rg = new double[this.n];
		double[] diffx, diffy;
		for (int i=0; i<this.n; i++){
			diffx = VectorFun.mult(this.blobcoord[i][0],-1);
			VectorFun.addi(diffx,this.blobcenters[0][i]);
			diffy = VectorFun.mult(this.blobcoord[i][1],-1);
			VectorFun.addi(diffy,this.blobcenters[1][i]);
			rg[i] = (VectorFun.sum(VectorFun.mult(diffx,diffx))+VectorFun.sum(VectorFun.mult(diffy,diffy)))/this.blobsize[i];
		}
		return rg;
	}
	
	public int[] getBlobSizes(){
		return this.blobsize.clone();
	}
}