package flib.algorithms.regions;

import flib.math.VectorFun;
import flib.math.BV;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.algorithms.Watershed;
import flib.algorithms.BLabel;
import flib.algorithms.AssignToRandomNeighbor;

public class RegionPartitioning {
	private int[][] regions;
	private double[] reg;
	private int w,h;
	
	public RegionPartitioning(int w, int h, final double[] x){
		this.w = w;
		this.h = h;
		double[] wsp = VectorConv.int2double((new Watershed(this.w,this.h,VectorFun.mult(x,-1),8)).getRegionNumber());
		double[] wsn = VectorConv.int2double((new Watershed(this.w,this.h,x,8)).getRegionNumber());
		wsp = VectorConv.bool2double(BV.gt(wsp,0));
		wsn = VectorConv.bool2double(BV.gt(wsn,0));
		double[] binim = VectorConv.bool2double(BV.gt(x,0));
		double[] blp = VectorFun.add((new BLabel(this.w,this.h,VectorFun.mult(binim,wsp),4)).getBlobNumber(),1);
		blp = VectorFun.mult((new AssignToRandomNeighbor(this.w,this.h,blp,4)).getImage(),binim);
		double[] bln = VectorFun.add((new BLabel(this.w,this.h,VectorFun.mult(VectorFun.sub(1,binim),wsn),4)).getBlobNumber(),1);
		bln = VectorFun.mult((new AssignToRandomNeighbor(this.w,this.h,bln,4)).getImage(),VectorFun.sub(1,binim));
		int m = (int)VectorFun.max(blp)[0];
		VectorFun.addi(bln,m);
		this.reg = bln.clone();
		int[] subset = VectorAccess.subset(binim);
		VectorAccess.write(this.reg,subset,VectorAccess.access(blp,subset));
		// here we need to assign some single pixels which were on the watershed boundaries to all other regions
		//this.reg = (new AssignToRandomNeighbor(this.w,this.h,this.reg,4)).getImage();
		//this.reg = VectorFun.sub(this.reg,1);
		this.reg = (new BLabel(this.w,this.h,VectorFun.add(this.reg,1),8)).getBlobNumber();
		m = (int)VectorFun.max(this.reg)[0]+1;
		int[] count = new int[m];
		this.regions = new int[m][];
		for (int i=0; i<x.length; i++){
			count[(int)this.reg[i]]++;
		}
		for (int i=0; i<m; i++){
			this.regions[i] = new int[count[i]];
		}
		count = new int[(int)m];
		for (int i=0; i<x.length; i++){
			this.regions[(int)this.reg[i]][count[(int)this.reg[i]]] = i;
			count[(int)this.reg[i]]++;
		}
	}
	
	public int[][] getRegions(){
		int[][] temp = new int[this.regions.length][];
		for (int i=0; i<this.regions.length; i++){
			temp[i] = this.regions[i].clone();
		}
		return temp;
	}
	
	public double[] getImage(){
		return this.reg.clone();
	}
}