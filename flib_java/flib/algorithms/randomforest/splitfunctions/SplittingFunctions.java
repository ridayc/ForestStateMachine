package flib.algorithms.randomforest.splitfunctions;

import java.lang.Math;
import flib.math.VectorFun;

public class SplittingFunctions {
	public static double[] binaryGini(final double[][] clabels, boolean cat){
		// minimum gini impurity and it's location
		double[] m = new double[2];
		double wt,fl0=0, f0=0, fl1=0, f1=0;
		int l = clabels.length/2;
		// initialize all values
		for (int i=0; i<l; i++){
			f0+=clabels[i+l][0];
			f1+=clabels[i+l][1];
		}
		wt = f0+f1;
		m[0] = Double.MAX_VALUE;
		int st;
		if (!cat){
			st = l-1;
		}
		else {
			st = l;
		}
		// go through all values
		for (int i=0; i<st; i++){
			// standard case
			if (!cat){
				fl0+=clabels[i+l][0];
				fl1+=clabels[i+l][1];
			}
			// categorical case
			else {
				fl0 = clabels[i+l][0];
				fl1 = clabels[i+l][1];
			}
			double wl = fl0+fl1;
			double wr = wt-wl;
			double pl0 = fl0/wl;
			double pl1 = fl1/wl;
			double pr0 = (f0-fl0)/wr;
			double pr1 = (f1-fl1)/wr;
			double g = wl*(1-pl0*pl0-pl1*pl1)+wr*(1-pr0*pr0-pr1*pr1);
			if (g<m[0]){
				// store the split value
				m[0] = g;
				// store the split location
				if (!cat){
					m[1] = (clabels[i][1]+clabels[i+1][1])*0.5;
				}
				else {
					m[1] = clabels[i][1];
				}
			}
		}
		return m;
	}
	
	public static double[] binaryInfGain(final double[][] clabels, boolean cat){
		// minimum information gain and it's location
		double[] m = new double[2];
		double wt,fl0=0, f0=0, fl1=0, f1=0;
		int l = clabels.length/2;
		// initialize all values
		for (int i=0; i<l; i++){
			f0+=clabels[i+l][0];
			f1+=clabels[i+l][1];
		}
		wt = f0+f1;
		m[0] = Double.MAX_VALUE;
		int st;
		if (!cat){
			st = l-1;
		}
		else {
			st = l;
		}
		// go through all values
		for (int i=0; i<st; i++){
			// standard case
			if (!cat){
				fl0+=clabels[i+l][0];
				fl1+=clabels[i+l][1];
			}
			// categorical case
			else {
				fl0 = clabels[i+l][0];
				fl1 = clabels[i+l][1];
			}
			double wl = fl0+fl1;
			double wr = wt-wl;
			double pl0 = fl0/wl;
			double pl1 = fl1/wl;
			double pr0 = (f0-fl0)/wr;
			double pr1 = (f1-fl1)/wr;
			double lpl0=0;
			if (pl0>0){
				lpl0 = Math.log(pl0);
			}
			double lpl1=0;
			if (pl1>0){
				lpl1 = Math.log(pl1);
			}
			double lpr0=0;
			if (pr0>0){
				lpr0 = Math.log(pr0);
			}
			double lpr1=0;
			if (pr1>0){
				lpr1 = Math.log(pr1);
			}
			double g = wl*(-pl0*lpl0-pl1*lpl1)+wr*(-pr0*lpr0-pr1*lpr1);
			if (g<m[0]){
				// store the split value
				m[0] = g;
				// store the split location
				if (!cat){
					m[1] = (clabels[i][1]+clabels[i+1][1])*0.5;
				}
				else {
					m[1] = clabels[i][1];
				}
			}
		}
		return m;
	}
	
	public static double[] binaryMedian(final double[][] clabels, boolean cat){
		int l = clabels.length/2;
		double[] m = new double[2];
		// count the total number of transitions
		double t = 0;
		double[] tl = new double[l];
		double w, w2;
		// between value transitions
		if(!cat){
			for (int i=0; i<l-1; i++){
				// inside value transitions
				if (clabels[l+i][0]>0&&clabels[l+i][1]>0){
					w = clabels[l+i][0]+clabels[l+i][1];
					t+=clabels[l+i][0]/w*clabels[l+i][1]/w*(w-1);
				}
				// single class transitions
				if (!(clabels[l+i][0]>0&&clabels[l+i][1]>0)&&!(clabels[l+i+1][0]>0&&clabels[l+i+1][1]>0)){
					if ((clabels[l+i][0]>0&&clabels[l+i+1][1]>0)||(clabels[l+i][1]>0&&clabels[l+i+1][0]>0)){
						t++;
					}
				}
				else {
					w = clabels[l+i][0]+clabels[l+i][1];
					w2 = clabels[l+i+1][0]+clabels[l+i+1][1];
					t+=clabels[l+i][0]/w*clabels[l+i+1][1]/w2+clabels[l+i][1]/w*clabels[l+i+1][0]/w2;
				}
				tl[i] = t;
			}
			if (clabels[2*l-1][0]>0&&clabels[2*l-1][1]>0){
				w = clabels[2*l-1][0]+clabels[2*l-1][1];
				t+=clabels[2*l-1][0]/w*clabels[2*l-1][1]/w*(w-1);
			}
			tl[l-1] = t;
			// the median value
			m[0] = t/2;
			for (int i=1; i<l; i++){
				if (m[0]>tl[i]){
					m[1] = (clabels[i][1]+clabels[i-1][1])*0.5;
					break;
				}
			}
		}
		return m;
	}
	
	public static double[] gini(final double[][] clabels, boolean cat, int num_class, double alpha){
		// minimum gini impurity and it's location
		double[] m = new double[2];
		double wt;
		double[] f = new double[num_class], fl = new double[num_class];
		int l = clabels.length/3;
		// initialize all values
		for (int i=0; i<l; i++){
			for (int j=0; j<clabels[i+l].length; j++){
				f[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
			}
		}
		wt = VectorFun.sum(f);
		m[0] = Double.MAX_VALUE*0.5;
		// stopping value
		int st;
		if (!cat){
			st = l-1;
		}
		else {
			st = l;
		}
		// we'll have problems if there's only a single dimension
		if (l==1){
			m[0] = Double.MAX_VALUE*0.5;
			m[1] = clabels[0][1];
			return m;
		}
		// go through all values
		for (int i=0; i<st; i++){
			// standard case
			if (!cat){
				for (int j=0; j<clabels[i+l].length; j++){
					fl[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
				}
			}
			// categorical case
			else {
				fl = new double[num_class];
				for (int j=0; j<clabels[i+l].length; j++){
					fl[(int)clabels[i+l][j]] = clabels[i+2*l][j];
				}
			}
			double wl = VectorFun.sum(fl);
			double wr = wt-wl;
			double[] pl = new double[num_class], pr = new double[num_class];
			for (int j=0; j<num_class; j++){
				pl[j] = fl[j]/wl;
				if (Double.isNaN(pl[j])){
					pl[j] = 0;
				}
				pr[j] = (f[j]-fl[j])/wr;
				if (Double.isNaN(pr[j])){
					pr[j] = 0;
				}
			}
			double gl = 1, gr = 1;
			for (int j=0; j<num_class; j++){
				gl-=pl[j]*pl[j];
				gr-=pr[j]*pr[j];
			}
			double g = wl*gl+wr*gr;
			if (g<m[0]){
				// store the split value
				m[0] = g;
				// store the split location
				if (!cat){
					m[1] = (clabels[i][1]+clabels[i+1][1])*0.5;
				}
				else {
					m[1] = clabels[i][1];
				}
			}
		}
		return m;
	}
	
	public static double[] gini_flip(final double[][] clabels, boolean cat, int num_class, int alpha){
		double[] m = new double[2];
		if (!cat){
			m[1] = (clabels[alpha][1]+clabels[alpha+1][1])*0.5;
		}
		else {
			m[1] = clabels[alpha][1];
		}
		int l = clabels.length/3;
		double[] f = new double[num_class], fl = new double[num_class];
		if (l<=1){
			m[0] = Double.MAX_VALUE*0.5;
			return m;
		}
		if (cat){
			// we need the weight of each class over all categories
			for (int i=0; i<l; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					f[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
				}
			}
			double wt = VectorFun.sum(f);
			// for the flip counting
			double t = 0;
			// compare each indivual class with itself, and with the rest
			for (int i=0; i<l; i++){
				double s = 0;
				for (int j=0; j<num_class; j++){
					fl[j] = 0;
				}
				for (int j=0; j<clabels[i+l].length; j++){
					fl[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
					s+=clabels[i+2*l][j];
				}
				for (int j=0; j<num_class; j++){
					t+=fl[j]/s*(1-fl[j]/s)*s;
				}
			}
			// times wt to adjust for flips between categories
			m[0] = t;
		}
		else {
			double s1 = 0;
			double s2 = 0;
			double t = 0;
			for (int j=0; j<clabels[l].length; j++){
				f[(int)clabels[l][j]]+=clabels[2*l][j];
				s1+=clabels[2*l][j];
			}
			for (int j=0; j<num_class; j++){
				t+=f[j]/s1*(1-f[j]/s1)*s1;
			}
			for (int i=1; i<l; i++){
				if (clabels[i-1+l].length<2&&clabels[i+l].length<2){
					if (clabels[i-1+l][0]!=clabels[i+l][0]){
						t++;
						s1 = clabels[i+2*l][0];
						f[(int)clabels[i-1+l][0]] = 0;
						f[(int)clabels[i+l][0]] = clabels[i+2*l][0];
						fl[(int)clabels[i+l][0]] = 0;
					}
				}
				else {
					s2 = 0;
					for (int j=0; j<clabels[i+l].length; j++){
						fl[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
						s2+=clabels[i+2*l][j];
					} 
					double w = (s1+s2)*0.5;
					for (int j=0; j<num_class; j++){
						//within point
						t+=f[j]/s1*(1-f[j]/s1)*s1;
						// between points
						t+=f[j]/s1*(1-fl[j]/s2)*w;
						f[j] = fl[j];
						fl[j] = 0;
					}
					s1 = s2;
				}
			}
			m[0] = t;
		}
		return m;
	}
	
	public static double[] infGain(final double[][] clabels, boolean cat, int num_class, double alpha){
		// minimum information gain and it's location
		double[] m = new double[2];
		double wt;
		double[] f = new double[num_class], fl = new double[num_class];
		int l = clabels.length/3;
		// initialize all values
		for (int i=0; i<l; i++){
			for (int j=0; j<clabels[i+l].length; j++){
				f[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
			}
		}
		wt = VectorFun.sum(f);
		m[0] = Double.MAX_VALUE*0.5;
		// we'll have problems if there's only a single dimension
		if (l==1){
			m[0] = Double.MAX_VALUE*0.5;
			m[1] = clabels[0][1];
			return m;
		}
		int st;
		if (!cat){
			st = l-1;
		}
		else {
			st = l;
		}
		// go through all values
		for (int i=0; i<st; i++){
			// standard case
			if (!cat){
				for (int j=0; j<clabels[i+l].length; j++){
					fl[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
				}
			}
			// categorical case
			else {
				fl = new double[num_class];
				for (int j=0; j<clabels[i+l].length; j++){
					fl[(int)clabels[i+l][j]] = clabels[i+2*l][j];
				}
			}
			double wl = VectorFun.sum(fl);
			double wr = wt-wl;
			double[] pl = new double[num_class], pr = new double[num_class];
			for (int j=0; j<num_class; j++){
				pl[j] = fl[j]/wl;
				pr[j] = (f[j]-fl[j])/wr;
			}
			double gl = 0, gr = 0;
			for (int j=0; j<num_class; j++){
				if (pl[j]>0){
					gl-=pl[j]*Math.log(pl[j]);
				}
				if (pr[j]>0){
					gr-=pr[j]*Math.log(pr[j]);
				}
			}
			double g = wl*gl+wr*gr;
			if (g<m[0]){
				// store the split value
				m[0] = g;
				// store the split location
				if (!cat){
					m[1] = (clabels[i][1]+clabels[i+1][1])*0.5;
				}
				else {
					m[1] = clabels[i][1];
				}
			}
		}
		return m;
	}
	
	public static double[] randSplit(final double[][] clabels, boolean cat, int num_class, int alpha){
		// minimum gini impurity and it's location
		double[] m = new double[2];
		if (!cat){
			m[1] = (clabels[alpha][1]+clabels[alpha+1][1])*0.5;
		}
		else {
			m[1] = clabels[alpha][1];
		}
		m[0] = Double.MAX_VALUE*0.5;
		int l = clabels.length/3;
		double[] f = new double[num_class], fl = new double[num_class];
		if (l<=1){
			m[0] = Double.MAX_VALUE*0.5;
			return m;
		}
		double wt;
		wt = VectorFun.sum(f);
		// stopping value
		int st;
		if (!cat){
			st = l-1;
			// initialize all values
			for (int i=0; i<alpha; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					f[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
				}
			}
			// other partition
			for (int i=alpha; i<st; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					fl[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
				}
			}
		}
		else {
			st = l;
			// initialize all values
			for (int i=0; i<st; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					f[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
				}
			}
			// other partition
			for (int i=alpha; i<alpha+1; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					fl[(int)clabels[i+l][j]]+=clabels[i+2*l][j];
				}
			}
		}
		double wl = VectorFun.sum(fl);
		double wr = VectorFun.sum(f);
		double[] pl = new double[num_class], pr = new double[num_class];
		for (int j=0; j<num_class; j++){
			pl[j] = fl[j]/wl;
			if (Double.isNaN(pl[j])){
				pl[j] = 0;
			}
			pr[j] = (f[j])/wr;
			if (Double.isNaN(pr[j])){
				pr[j] = 0;
			}
		}
		double gl = 0, gr = 0;
		for (int j=0; j<num_class; j++){
			if (pl[j]>0){
				gl-=pl[j]*Math.log(pl[j]);
			}
			if (pr[j]>0){
				gr-=pr[j]*Math.log(pr[j]);
			}
		}
		double g = wl*gl+wr*gr;
		if (g<m[0]){
			// store the split value
			m[0] = g;
		}
		return m;
	}
	
	public static double[] varReduc(final double[][] clabels, boolean cat, double alpha){
		// minimum total variance
		double[] m = new double[2];
		double mt=0, m2t=0, ml=0, m2l=0, wt=0, wl=0;
		int l = clabels.length/3;
		// variable initialization
		for (int i=0; i<l; i++){
			for (int j=0; j<clabels[i+l].length; j++){
				mt+=clabels[i+l][j]*clabels[i+2*l][j];
				m2t+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
				wt+=clabels[i+2*l][j];
			}
		}
		m[0] = Double.MAX_VALUE*0.5;
		if (l<=1){
			return m;
		}
		int st;
		if (!cat){
			st = l-1;
		}
		else {
			st = l;
		}
		// go through all values
		for (int i=0; i<st; i++){
			if (!cat){
				for (int j=0; j<clabels[i+l].length; j++){
					ml+=clabels[i+l][j]*clabels[i+2*l][j];
					m2l+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
					wl+=clabels[i+2*l][j];
				}
			}
			else {
				for (int j=0; j<clabels[i+l].length; j++){
					wl = clabels[i+2*l][j];
					ml = clabels[i+l][j]*clabels[i+2*l][j];
					m2l = clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
				}
			}
			double wr = wt-wl;
			double vl = m2l/wl-(ml*ml)/(wl*wl);
			double vr = (m2t-m2l)/wr-(mt-ml)*(mt-ml)/(wr*wr);
			double v = wl*vl+wr*vr;
			if (v<m[0]){
				// store the split value
				m[0] = v;
				// store the split location
				if (!cat){
					m[1] = (clabels[i][1]+clabels[i+1][1])*0.5;
				}
				else {
					m[1] = clabels[i][1];
				}
			}
		}
		return m;
	}
	
	public static double[] var_flip(final double[][] clabels, boolean cat, int alpha){
		double[] m = new double[2];
		if (!cat){
			m[1] = (clabels[alpha][1]+clabels[alpha+1][1])*0.5;
		}
		else {
			m[1] = clabels[alpha][1];
		}
		int l = clabels.length/3;
		double mr=0, mr2=0, ml=0, ml2=0, wr=0, wl=0;
		if (l<=1){
			m[0] = Double.MAX_VALUE*0.5;
			return m;
		}
		if (cat){
			double t = 0;
			// we need the weight of each class over all categories
			for (int i=0; i<l; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					ml+=clabels[i+l][j]*clabels[i+2*l][j];
					ml2+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
					wl+=clabels[i+2*l][j];
				}
				t+=(ml2/wl-ml*ml/(wl*wl))*wl;
			}
			m[0] = t;
		}
		else {
			double t = 0;
			for (int j=0; j<clabels[l].length; j++){
				ml+=clabels[l][j]*clabels[2*l][j];
				ml2+=clabels[l][j]*clabels[l][j]*clabels[2*l][j];
				wl+=clabels[2*l][j];
			}
			for (int i=1; i<l; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					mr+=clabels[i+l][j]*clabels[i+2*l][j];
					mr2+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
					wr+=clabels[i+2*l][j];
				}
				double me = ml+mr;
				double m2 = ml2+mr2;
				double w = wl+wr;
				// w is to give a weight to this variance
				t+=(m2/w-(me*me)/(w*w))*w/2;
				ml2 = mr2;
				ml = mr;
				wl = wr;
				mr2 = 0;
				mr = 0;
				wr = 0;
			}
			m[0] = t;
		}
		return m;
	}
	
		public static double[] varSplit(final double[][] clabels, boolean cat, int alpha){
		// minimum total variance
		double[] m = new double[2];
		if (!cat){
			m[1] = (clabels[alpha][1]+clabels[alpha+1][1])*0.5;
		}
		else {
			m[1] = clabels[alpha][1];
		}
		int l = clabels.length/3;
		double mr=0, m2r=0, ml=0, m2l=0, wr=0, wl=0;
		if (l<=1){
			m[0] = Double.MAX_VALUE*0.5;
			return m;
		}
		int st;
		if (!cat){
			st = l-1;
			// variable initialization
			for (int i=0; i<alpha; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					mr+=clabels[i+l][j]*clabels[i+2*l][j];
					m2r+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
					wr+=clabels[i+2*l][j];
				}
			}
			for (int i=alpha; i<st; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					ml+=clabels[i+l][j]*clabels[i+2*l][j];
					m2l+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
					wl+=clabels[i+2*l][j];
				}
			}
		}
		else {
			st = l;
			// variable initialization
			for (int i=0; i<st; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					mr+=clabels[i+l][j]*clabels[i+2*l][j];
					m2r+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
					wr+=clabels[i+2*l][j];
				}
			}
			for (int i=alpha; i<alpha+1; i++){
				for (int j=0; j<clabels[i+l].length; j++){
					ml+=clabels[i+l][j]*clabels[i+2*l][j];
					m2l+=clabels[i+l][j]*clabels[i+l][j]*clabels[i+2*l][j];
					wl+=clabels[i+2*l][j];
				}
			}
		}
		// go through all values
		double vl = m2l/wl-(ml*ml)/(wl*wl);
		double vr = (m2r)/wr-(mr)*(mr)/(wr*wr);
		double v = wl*vl+wr*vr;
		if (v<m[0]){
			// store the split value
			m[0] = v;
		}
		return m;
	}
}