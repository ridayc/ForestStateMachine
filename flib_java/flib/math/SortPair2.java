package flib.math;

// this function creates a key value pair for sorting purposes. 
// if the values of to SortPairs are equal, this equality will be returned

public class SortPair2 implements Comparable<SortPair2>,  java.io.Serializable{
	private double originalIndex;
	private double value;

	public SortPair2(double value, double originalIndex){
		this.value = value;
		this.originalIndex = originalIndex;
	}

	@Override public int compareTo(SortPair2 o){
		int a = Double.compare(value, o.getValue());
		return a;
	}

	public double getOriginalIndex(){
		return originalIndex;
	}

	public double getValue(){
		return value;
	}
	
	public void setOriginalIndex(double i){
		originalIndex = i;
	}
	
	public void setValue(double v){
		value = v;
	}
}