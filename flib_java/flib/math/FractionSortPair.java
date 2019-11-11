package flib.math;

public class FractionSortPair implements Comparable<FractionSortPair> {
	private int ind;
	private int v1;
	private int v2;

	public FractionSortPair(int v1, int v2, int ind){
		this.v1 = v1;
		this.v2 = v2;
		this.ind = ind;
	}

	@Override public int compareTo(FractionSortPair o){
		int a;
		if ((getV1()==o.getV1())&&(getV2()==o.getV2())){
			a = 0;
		}
		else {
			a = Double.compare(getFraction(), o.getFraction());
		}
		if (a==0){
			a = compare(this.ind, o.getInd());
		}
		return a;
	}

	public int getInd(){
		return this.ind;
	}

	public double getFraction(){
		return (double)this.v1/this.v2;
	}
	
	public int getV1(){
		return this.v1;
	}
	
	public int getV2(){
		return this.v2;
	}
	
	public static int compare(int x, int y) {
		return (x<y)?-1:((x==y)?0:1);
	}
}