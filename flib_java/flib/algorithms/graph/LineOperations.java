package flib.algorithms.graph;

import java.util.ArrayList;

public class LineOperations {
	private ArrayList<ArrayList<Integer>> segment;
	private ArrayList<ArrayList<Integer>> segmentnumber;
	
	public LineOperations(final ArrayList<ArrayList<Integer>> segment, final ArrayList<ArrayList<Integer>> segmentnumber, final double[] x, double percentage,  final boolean[] loops, boolean nodes){
		this.getSegments(segment,segmentnumber,x,percentage, loops, nodes);
	}
	
	public LineOperations(final ArrayList<ArrayList<Integer>> segment, final ArrayList<ArrayList<Integer>> segmentnumber, final double[] x, double percentage){
		this(segment,segmentnumber,x,percentage, new boolean[0], false);
	}
	
	public LineOperations(final ArrayList<ArrayList<Integer>> segment, final ArrayList<ArrayList<Integer>> segmentnumber, final double[] x){
		this(segment,segmentnumber,x,1);
	}
	
	public void getSegments(final ArrayList<ArrayList<Integer>> segment, final ArrayList<ArrayList<Integer>> segmentnumber, final double[] x, double percentage, final boolean[] loops, boolean nodes){
		this.segment = new ArrayList<ArrayList<Integer>>();
		this.segmentnumber = new ArrayList<ArrayList<Integer>>();		
		// create a list for each pixel containing the segments this pixel belongs to
		for (int i=0; i<segmentnumber.size(); i++){
			this.segmentnumber.add(new ArrayList<Integer>());
		}
		// go through all  segments and check if they're pixels fulfill the keeping criteria
		int count;
		for (int i=0; i<segment.size(); i++){
			count = 0;
			// count the number of pixels which are larger than 0
			for (int j=0; j<segment.get(i).size(); j++){
				if (x[segment.get(i).get(j)]>0){
					count++;
				}
			}
			// the number of larger than 0 pixels exceeds a certain percentage
			// store these pixels in a list
			if ((double)count/segment.get(i).size()>=percentage){
				this.segment.add(new ArrayList<Integer>());
				for (int j=0; j<segment.get(i).size(); j++){
					int a = segment.get(i).get(j);
					int b = this.segment.size()-1;
					this.segment.get(b).add(a);
					this.segmentnumber.get(a).add(b);
				}
			}
		}
		if (nodes){
			// in the end we go through all pixels and check if there were ending points
			// we should be kept as individual pixel/line segments
			for (int i=0; i<segment.size(); i++){
				int a = segment.get(i).get(0);
				int b = segment.get(i).get(segment.get(i).size()-1);
				if (this.segmentnumber.get(a).size()<1&&x[a]>0&&!loops[a]){
					this.segment.add(new ArrayList<Integer>());
					int c = this.segment.size()-1;
					this.segment.get(c).add(a);
					this.segmentnumber.get(a).add(c);
				}
				if (this.segmentnumber.get(b).size()<1&&x[b]>0&&!loops[b]){
					this.segment.add(new ArrayList<Integer>());
					int c = this.segment.size()-1;
					this.segment.get(c).add(b);
					this.segmentnumber.get(b).add(c);
				}
			}
		}
	}
		
	public ArrayList<ArrayList<Integer>> getSegmentNumber(){
		return this.segmentnumber;
	}
	
	public ArrayList<ArrayList<Integer>> getSegments(){
		return this.segment;
	}
	
	public static boolean[] graphPixel(final ArrayList<ArrayList<Integer>> segmentnumber){
		boolean[] graphp = new boolean[segmentnumber.size()];
		for (int i=0; i<graphp.length; i++){
			if (segmentnumber.get(i).size()>0){
				graphp[i] = true;
			}
		}
		return graphp;
	}
}