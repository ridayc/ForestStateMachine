package flib.math;

public class VectorConv {
	// A vector type and value conversions
	
	// Convert a boolean vector to a double vector
	//
	public static double[] bool2double(final boolean[] b){
		int l = b.length;
		double[] d = new double[l];
		for (int i=0; i<l; i++){
			if (b[i]) {
				d[i] = 1;
			}
			else {
				d[i] = 0;
			}
		}
		return d;
	}
	
	// Convert an integer vector to a double vector
	//
	public static double[] int2double(final int[] fi){
		int l = fi.length;
		double[] d = new double[l];
		for (int i=0; i<l; i++){
			d[i] = (double)fi[i];
		}
		return d;
	}
	
	// Convert a float vector to a double vector
	public static double[] float2double(final float[] f){
		int l = f.length;
		double[] d = new double[l];
		for (int i=0; i<l; i++){
			d[i] = (double)f[i];
		}
		return d;
	}
	
	// Convert a double vector to a float vector
	public static float[] double2float(final double[] d){
		int l = d.length;
		float[] f = new float[l];
		for (int i=0; i<l; i++){
			f[i] = (float)d[i];
		}
		return f;
	}
	
	// Convert a float vector to an integer vector
	public static int[] float2int(final float[] f){
		int l = f.length;
		int[] d = new int[l];
		for (int i=0; i<l; i++){
			d[i] = (int)f[i];
		}
		return d;
	}
	
	// Convert a double vector to an integer vector
	//
	public static int[] double2int(final double[] d){
		int l = d.length;
		int[] fi = new int[l];
		for (int i=0; i<l; i++){
			fi[i] = (int)d[i];
		}
		return fi;
	}
	
	// set all NaN values
	//
	public static double[] nan2value(final double[] x, double v){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i];
			if (Double.isNaN(x[i])) z[i] = v;
		}
		return z;
	}
	
	public static double[] nan2value(final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i];
			if (Double.isNaN(x[i])) z[i] = 0;
		}
		return z;
	}
	
	// set all infinite values
	public static double[] inf2value(final double[] x, double v){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i];
			if (Double.isInfinite(x[i])) z[i] = v;
		}
		return z;
	}
	
	public static double[] inf2value(final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i];
			if (Double.isInfinite(x[i])) z[i] = 0;
		}
		return z;
	}
	
	public static byte[] double2byte(double[] x){
		byte[] b = new byte[x.length*8];
		for (int i=0; i<x.length; i++){
			long l = Double.doubleToRawLongBits(x[i]);
			int pos = i*8;
			b[pos+7] = (byte)((l>>0) & 0xff);
			b[pos+6] = (byte)((l>>8) & 0xff);
			b[pos+5] = (byte)((l>>16) & 0xff);
			b[pos+4] = (byte)((l>>24) & 0xff);
			b[pos+3] = (byte)((l>>32) & 0xff);
			b[pos+2] = (byte)((l>>40) & 0xff);
			b[pos+1] = (byte)((l>>48) & 0xff);
			b[pos]   = (byte)((l>>56) & 0xff);
		}
		return b;
	}
	
	public static double[] byte2double(byte[] b){
		// be divisible by eight please...
		double[] x = new double[b.length/8];
		for (int i=0; i<x.length; i++){
			int pos = i*8;
			long v = (long)((long)(0xff & b[pos]) << 56 |
				(long)(0xff & b[pos+1]) << 48 |
				(long)(0xff & b[pos+2]) << 40 |
				(long)(0xff & b[pos+3]) << 32 |
				(long)(0xff & b[pos+4]) << 24 |
				(long)(0xff & b[pos+5]) << 16 |
				(long)(0xff & b[pos+6]) << 8 |
				(long)(0xff & b[pos+7]) << 0 );
			x[i] = Double.longBitsToDouble(v);
		}
		return x;
	}		

	public static double[][] int2double(int[][] x){
		double[][] y = new double[x.length][];
		for (int i=0; i<x.length; i++){
			y[i] = int2double(x[i]);
		}
		return y;
	}
}