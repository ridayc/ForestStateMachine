package flib.math;

public class SortGroup implements Comparable<SortGroup>,  java.io.Serializable{
	private double[] group;
	private double value;

	public SortGroup(double value, double[] group){
		this.value = value;
		this.group = group.clone();
	}

	@Override public int compareTo(SortGroup o){
		int a = Double.compare(value, o.getValue());
		return a;
	}

	public double[] getGroup(){
		return group;
	}

	public double getValue(){
		return value;
	}
}