package flib.math;

public class VectorAccess {
	public static int[] subset(final boolean[] x){
		int n = x.length;
		int l = 0;
		for (int i=0; i<n; i++){
			if (x[i]) l++;
		}
		int[] sub = new int[l];
		int count = 0;
		for (int i=0; i<n; i++){
			if (x[i]){
				sub[count] = i;
				count++;
			}
		}
		return sub;
	}
	
	public static int[] subset(final double[] x){
		int n = x.length;
		int l = 0;
		for (int i=0; i<n; i++){
			if (x[i]==1) l++;
		}
		int[] sub = new int[l];
		int count = 0;
		for (int i=0; i<n; i++){
			if (x[i]==1) {
				sub[count] = i;
				count++;
			}
		}
		return sub;
	}
	
	public static double[] access(final double[] x, final int[] sub){
		int l = sub.length;
		double[] a = new double[l];
		for (int i=0; i<l; i++){
			a[i] = x[sub[i]];
		}
		return a;
	}
	
	public static double[][] access2(final double[][] x, final int[] sub){
		int l = sub.length;
		double[][] a = new double[l][];
		for (int i=0; i<l; i++){
			a[i] = x[sub[i]].clone();
		}
		return a;
	}
	
	public static int[] access(final int[] x, final int[] sub){
		int l = sub.length;
		int[] a = new int[l];
		for (int i=0; i<l; i++){
			a[i] = x[sub[i]];
		}
		return a;
	}
	
	public static double[] access(final double[] x, int start, int end){
		double[] a = new double[end-start];
		for (int i=start; i<end; i++){
			a[i-start] = x[i];
		}
		return a;
	}
	
	public static int[] access(final int[] x, int start, int end){
		int[] a = new int[end-start];
		for (int i=start; i<end; i++){
			a[i-start] = x[i];
		}
		return a;
	}
	
	public static double[][] access(final double[][] x, final int[] sub){
		int l = sub.length;
		int d = x.length;
		double[][] a = new double[d][l];
		for (int i=0; i<d; i++){
			a[i] = access(x[i],sub);
		}
		return a;
	}
	
	public static void write(final double[] x, final int[] sub, final double[] y){
		int l = sub.length;
		for (int i=0; i<l; i++){
			x[sub[i]] = y[i];
		}
	}
	
	public static void write(final double[] x, final int[] sub, double y){
		int l = sub.length;
		for (int i=0; i<l; i++){
			x[sub[i]] = y;
		}
	}
	
	public static void write(final double[] x, final double[] y, int start){
		int l = y.length;
		for (int i=start; i<start+l; i++){
			x[i] = y[i-start];
		}
	}
	
	/*public static void write(final int[] x, final int[] sub, int y){
		int l = sub.length;
		for (int i=0; i<l; i++){
			x[sub[i]] = y;
		}
	}*/
	
	public static void write(final int[] x, final int[] sub, final int[] y){
		int l = sub.length;
		for (int i=0; i<l; i++){
			x[sub[i]] = y[i];
		}
	}
	
	public static void write(final int[] x, final int[] y, int start){
		int l = y.length;
		for (int i=start; i<start+l; i++){
			x[i] = y[i-start];
		}
	}
	
	public static double[] vertCat2(final double[]... arrays){
		int l = 0;
		for (double[] array : arrays){
			l+=array.length;
		}
		double[] y = new double[l];
		int pos = 0;
		for (double[] array : arrays){
			for (int i=0; i<array.length; i++){
				y[pos] = array[i];
				pos++;
			}
		}
		return y;
	}
	
	public static int[] vertCat2(final int[]... arrays){
		int l = 0;
		for (int[] array : arrays){
			l+=array.length;
		}
		int[] y = new int[l];
		int pos = 0;
		for (int[] array : arrays){
			for (int i=0; i<array.length; i++){
				y[pos] = array[i];
				pos++;
			}
		}
		return y;
	}
	
	public static double[] vertCat(final double[][] x){
		int l = 0;
		for (int i=0; i<x.length; i++){
			l+=x[i].length;
		}
		double[] y = new double[l];
		int pos = 0;
		for (int i=0; i<x.length; i++){
			for (int j=0; j<x[i].length; j++){
				y[pos] = x[i][j];
				pos++;
			}
		}
		return y;
	}
	
	public static int[] vertCat(final int[][] x){
		int l = 0;
		for (int i=0; i<x.length; i++){
			l+=x[i].length;
		}
		int[] y = new int[l];
		int pos = 0;
		for (int i=0; i<x.length; i++){
			for (int j=0; j<x[i].length; j++){
				y[pos] = x[i][j];
				pos++;
			}
		}
		return y;
	}
	
	// make sure ahead of time that all arrays have the same length
	public static double[][] horzCat(final double[]... arrays){
		int n = arrays[0].length;
		int d = arrays.length;
		double[][] y = new double[n][d];
		for (int i=0; i<d; i++){
			for (int j=0; j<n; j++){
				y[j][i] = arrays[i][j];
			}
		}
		return y;
	}
	
	public static double[][] horzCat(final double[][]... arrays){
		int n = arrays[0].length;
		int d = 0;
		for (int i=0; i<arrays.length; i++){
			d+=arrays[i][0].length;
		}
		double[][] y = new double[n][d];
		d = 0;
		for (int i=0; i<arrays.length; i++){
			for (int k=0; k<n; k++){
				for (int j=0; j<arrays[i][0].length; j++){
					y[k][d+j] = arrays[i][k][j];
				}
			}
			d+=arrays[i][0].length;
		}
		return y;
	}
	
	public static double[][] flip(final double[][] x){
		int d = x[0].length;
		int n = x.length;
		double[][] y = new double[d][n];
		for (int i=0; i<d; i++){
			for (int j=0; j<n; j++){
				y[i][j] = x[j][i];
			}
		}
		return y;
	}
	
	// convert a list of labels to the indice vectors for each label
	public static int[][] labels2Indices(final int[] x, int numlab){
		int l = x.length;
		int[] count = new int[numlab];
		for (int i=0; i<l; i++){
			if (x[i]>=0){
				count[x[i]]++;
			}
		}
		int[][] ind = new int[numlab][];
		for (int i=0; i<numlab; i++){
			ind[i] = new int[count[i]];
			count[i] = 0;
		}
		for (int i=0; i<l;i++){
			if (x[i]>=0){
				ind[x[i]][count[x[i]]] = i;
				count[x[i]]++;
			}
		}
		return ind;
	}
}