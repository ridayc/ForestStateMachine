package flib.datastructures;

import java.util.ArrayList;
import java.lang.reflect.Array;

public class TypeConversion {
	// deep copy a primitve ArrayList to a primitive array
	// we'll just be assuming that the arraylist given fulfills all our desired properties
	// integer
	public static Object cpArrayList(ArrayList<?> alist, int dim, String className){
		// Prepare a list of all subdimensions which will be encountered
		ArrayList<Class<?>> c = createArrayClassTypes(dim,className);
		return copyMultiArrayList(c,alist,dim-1,className);
	}
	
	private static ArrayList<Class<?>> createArrayClassTypes(int dim, String className){
		ArrayList<Class<?>> c = new ArrayList<Class<?>>();
		try {
			c.add(Class.forName(className));
		}
		catch(ClassNotFoundException e){
			// too bad...
		}
		for (int i=0; i<dim-1; i++){
			c.add((Array.newInstance(c.get(i),1)).getClass());
		}
		return c;
	}
			
	public static Object copyMultiArrayList(ArrayList<Class<?>> class_names, ArrayList<?> alist, int depth, String className){
		Object copy = new Object();
		if (depth>0){
			copy = Array.newInstance(class_names.get(depth-1),alist.size());
			for (int i=0; i<alist.size(); i++){
				Array.set(copy,i,class_names.get(depth-1).cast(copyMultiArrayList(class_names,(ArrayList<?>)alist.get(i),depth-1,className)));
				//Array.set(copy,i,copyMultiArrayList(class_names,(ArrayList<?>)alist.get(i),depth-1));
			}
		}
		else {
			if (className.equals("[Z")){
				copy = copyMultiArrayListBoolean(class_names,(ArrayList<Boolean>)alist);
			}
			else if (className.equals("[B")){
				copy = copyMultiArrayListByte(class_names,(ArrayList<Byte>)alist);
			}
			else if (className.equals("[C")){
				copy = copyMultiArrayListCharacter(class_names,(ArrayList<Character>)alist);

			}
			else if (className.equals("[D")){
				copy = copyMultiArrayListDouble(class_names,(ArrayList<Double>)alist);
			}
			else if (className.equals("[F")){
				copy = copyMultiArrayListFloat(class_names,(ArrayList<Float>)alist);
			}
			else if (className.equals("[I")){
				copy = copyMultiArrayListInteger(class_names,(ArrayList<Integer>)alist);
			}
			else if (className.equals("[J")){
				copy = copyMultiArrayListLong(class_names,(ArrayList<Long>)alist);
			}
			else if (className.equals("[S")){
				copy = copyMultiArrayListShort(class_names,(ArrayList<Short>)alist);
			}
			else {
				for (int i=0; i<alist.size(); i++){
					Array.set(copy,i,alist.get(i));
					//copy[i] = (copy[i].getClass())alist.get(i);
				}
			}				
		}
		return copy;
	}
	
	public static boolean[] copyMultiArrayListBoolean(ArrayList<Class<?>> class_names, ArrayList<Boolean> alist){
		boolean[] copy = new boolean[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (boolean)alist.get(i);
		}
		return copy;
	}
	
	public static byte[] copyMultiArrayListByte(ArrayList<Class<?>> class_names, ArrayList<Byte> alist){
		byte[] copy = new byte[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (byte)alist.get(i);
		}
		return copy;
	}
	
	public static char[] copyMultiArrayListCharacter(ArrayList<Class<?>> class_names, ArrayList<Character> alist){
		char[] copy = new char[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (char)alist.get(i);
		}
		return copy;
	}
	
	public static double[] copyMultiArrayListDouble(ArrayList<Class<?>> class_names, ArrayList<Double> alist){
		double[] copy = new double[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (double)alist.get(i);
		}
		return copy;
	}
	
	public static float[] copyMultiArrayListFloat(ArrayList<Class<?>> class_names, ArrayList<Float> alist){
		float[] copy = new float[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (float)alist.get(i);
		}
		return copy;
	}
	
	public static int[] copyMultiArrayListInteger(ArrayList<Class<?>> class_names, ArrayList<Integer> alist){
		int[] copy = new int[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (int)alist.get(i);
		}
		return copy;
	}
	
	public static long[] copyMultiArrayListLong(ArrayList<Class<?>> class_names, ArrayList<Long> alist){
		long[] copy = new long[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (long)alist.get(i);
		}
		return copy;
	}
	
	public static short[] copyMultiArrayListShort(ArrayList<Class<?>> class_names, ArrayList<Short> alist){
		short[] copy = new short[alist.size()];
		for (int i=0; i<alist.size(); i++){
			copy[i] = (short)alist.get(i);
		}
		return copy;
	}
	
	public static Object copyMultiArrayObject(Object arr) {  
		Class clazz = arr.getClass();  
		if (!clazz.isArray()){
			throw new IllegalArgumentException("not an array: " + arr);
		}			
		Class componentType = clazz.getComponentType();  
		int length = Array.getLength(arr);  
		Object copy = Array.newInstance(componentType, length);  
		if (componentType.isArray()){
			for (int i = 0; i < length; i++) {
				Array.set(copy, i, copyMultiArrayObject(Array.get(arr, i)));
			}
		}
		else {
			System.arraycopy(arr, 0, copy, 0, length);
		}
		return copy;  
	}  
}
	
	