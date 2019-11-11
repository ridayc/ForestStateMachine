package flib.algorithms.graph;

import java.util.ArrayList;
import java.util.LinkedList;
import flib.math.VectorAccess;
import flib.algorithms.graph.NeighborConnectivity;

public class LineSearch {
	private ArrayList<Integer> island = new ArrayList<Integer>();
	private ArrayList<Integer> terminal = new ArrayList<Integer>();
	private ArrayList<Integer> inter = new ArrayList<Integer>();
	private ArrayList<Integer> branch = new ArrayList<Integer>();
	// segment contains for each segment all the pixels which belong to this segment
	private ArrayList<ArrayList<Integer>> segment = new ArrayList<ArrayList<Integer>>();
	// segment number contains for each image pixel the segments this pixel belongs to
	private ArrayList<ArrayList<Integer>> segmentnumber;
	
	public LineSearch(int w, int h, final double[] x){
		this.findSegments(w,h,x);
	}
	
	public void findSegments(int w, int h, final double[] x){
		int n = x.length;
		int[] d = new int[8];
		d[0] = -1;
		d[1] = 1;
		d[2] = -w;
		d[3] = w;
		d[4] = -w-1;
		d[5] = -w+1;
		d[6] = w-1;
		d[7] = w+1;
		double[] indarray = new double[n];
		for (int i=0; i<n; i++){
			indarray[i] = i;
		}
		double[][] connected = NeighborConnectivity.neighbors(w,h,x,8,0);
		connected = NeighborConnectivity.breakDiagonals(connected);
		double[][] neighborind = NeighborConnectivity.neighbors(w,h,indarray,8,-1);
		int[] wsind = VectorAccess.subset(x);
		int l = wsind.length;
		int[] sN = NeighborConnectivity.sumNeighbors(connected);
		for (int i=0; i<l; i++){
			if (sN[wsind[i]]==0){
				this.island.add(wsind[i]);
			}
			else if (sN[wsind[i]]==1){
				this.terminal.add(wsind[i]);
			}
			else if (sN[wsind[i]]==2){
				this.inter.add(wsind[i]);
			}
			
			else if (sN[wsind[i]]>2){
				this.branch.add(wsind[i]);
			}
		}
		this.segmentnumber = new ArrayList<ArrayList<Integer>>(n);
		for (int i=0; i<n; i++){
			this.segmentnumber.add(new ArrayList<Integer>());
		}
		boolean[] visited = new boolean[n];
		// visit all single points
		for (int i=0; i<this.island.size(); i++){
			this.segment.add(new ArrayList<Integer>());
			int s = this.segment.size()-1;
			this.segment.get(s).add(this.island.get(i));
			this.segmentnumber.get(this.island.get(i)).add(s);
		}
		// visit all terminal points
		for (int i=0; i<this.terminal.size(); i++){
			if (!visited[this.terminal.get(i)]){
				int a = this.terminal.get(i);
				visited[a] = true;
				this.segment.add(new ArrayList<Integer>());
				int s = this.segment.size()-1;
				this.segment.get(s).add(a);
				this.segmentnumber.get(a).add(s);
				int next = -1;
				int prev;
				for (int j=0; j<8; j++){
					if (connected[a][j]==1){
						next = (int)neighborind[a][j];
						break;
					}
				}
				prev = a;
				while (next>=0){
					this.segment.get(s).add(next);
					this.segmentnumber.get(next).add(s);
					if (sN[next]<3){
						visited[next] = true;
					}
					if (sN[next]!=2){
						next = -1;
						break;
					}
					else{
						for (int j=0; j<8; j++){
							if (connected[next][j]==1&&(int)neighborind[next][j]!=prev){
								prev = next;
								next = (int)neighborind[next][j];
								break;
							}
						}
					}
				}
			}
		}
		// visit all branching points
		for (int i=0; i<this.branch.size(); i++){
			int a = this.branch.get(i);
			visited[a] = true;
			for (int k=0; k<8; k++){
				if (connected[a][k]==1){
					if(!visited[(int)neighborind[a][k]]){
						this.segment.add(new ArrayList<Integer>());
						int s = this.segment.size()-1;
						this.segment.get(s).add(a);
						this.segmentnumber.get(a).add(s);
						int next, prev;
						next = (int)neighborind[a][k];
						prev = a;
						while (next>=0){
							this.segment.get(s).add(next);
							this.segmentnumber.get(next).add(s);
							if (sN[next]<3){
								visited[next] = true;
							}
							if (sN[next]!=2){
								next = -1;
								break;
							}
							else{
								for (int j=0; j<8; j++){
									if (connected[next][j]==1&&(int)neighborind[next][j]!=prev){
										prev = next;
										next = (int)neighborind[next][j];
										break;
									}
								}
							}
						}
					}
				}
			}
		}
		// visit all inter points to find remaining loops
		for (int i=0; i<this.inter.size(); i++){
			if (!visited[this.inter.get(i)]){
				int a = this.inter.get(i);
				visited[a] = true;
				this.segment.add(new ArrayList<Integer>());
				int s = this.segment.size()-1;
				this.segment.get(s).add(a);
				this.segmentnumber.get(a).add(s);
				int next = -1;
				int prev;
				for (int j=0; j<8; j++){
					if (connected[a][j]==1){
						next = (int)neighborind[a][j];
						break;
					}
				}
				prev = a;
				while (next>=0){
					this.segment.get(s).add(next);
					this.segmentnumber.get(next).add(s);
					visited[next] = true;
					for (int j=0; j<8; j++){
						if (connected[next][j]==1){
							if (visited[(int)neighborind[next][j]]){
								next=-1;
								break;
							}
							else if ((int)neighborind[next][j]!=prev){
								prev = next;
								next = (int)neighborind[next][j];
								break;
							}
						}
					}
				}
			}
		}
	}
	
	public double[] removeSegments(final double[] x, double c){
		boolean rem;
		double[] y = new double[x.length];
		int l;
		for (int i=0; i<this.segment.size(); i++){
			rem = false;
			for (int j=0; j<this.segment.get(i).size(); j++){
				y[this.segment.get(i).get(j)] = 1;
				if (x[this.segment.get(i).get(j)]<=c){
					rem = true;
				}
			}
			if (rem){
				l = this.segment.get(i).size()-1;
				for (int j=1; j<l; j++){
					y[this.segment.get(i).get(j)] = 0;
				}
				if (x[this.segment.get(i).get(0)]<=c){
					y[this.segment.get(i).get(0)] = 0;
				}
				if (x[this.segment.get(i).get(l)]<=c){
					y[this.segment.get(i).get(l)] = 0;
				}
			}
		}
		return y;
	}
	
	public ArrayList<ArrayList<Integer>> getSegmentNumber(){
		return this.segmentnumber;
	}
	
	public ArrayList<ArrayList<Integer>> getSegments(){
		return this.segment;
	}
	
	public boolean[] loopSegments(){
		boolean[] loopy = new boolean[this.segmentnumber.size()];
		for (int i=0; i<loopy.length; i++){
			loopy[i] = true;
		}
		// islands
		for (int i=0; i<this.island.size(); i++){
			loopy[this.island.get(i)] = false;
		}
		// terminals
		for (int i=0; i<this.terminal.size(); i++){
			loopy[this.terminal.get(i)] = false;
		}
		//branches
		for (int i=0; i<this.branch.size(); i++){
			loopy[this.branch.get(i)] = false;
		}
		return loopy;
	}
}