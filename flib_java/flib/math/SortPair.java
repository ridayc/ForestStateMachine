package flib.math;

public class SortPair implements Comparable<SortPair>,  java.io.Serializable{
	private double originalIndex;
	private double value;

	public SortPair(double value, double originalIndex){
		this.value = value;
		this.originalIndex = originalIndex;
	}

	@Override public int compareTo(SortPair o){
		int a = Double.compare(value, o.getValue());
		if (a==0){
			a = Double.compare(originalIndex, o.getOriginalIndex());
		}
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